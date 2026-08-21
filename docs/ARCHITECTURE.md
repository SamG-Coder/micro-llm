# Architecture

Target: Qwen 27B (3.6 / 3.8, 64-layer hybrid) on an RTX 5080.

## Hardware budget

| What | Number |
| --- | --- |
| Card | RTX 5080, 16GB GDDR7 |
| Usable headless | ~15.2GB |
| Usable with display | ~14.5GB |
| CUDA + decode scratch | 0.8-1.0GB before weights |
| Full model Q4_K_M | ~17.1GB, does not fit |
| Full model Q3 | ~13.4GB, almost no KV left |
| Remnant ceiling, 8-16k | <=12GB weights |
| Remnant ceiling, 32k + vision | <=10GB weights |

KV is cheaper than a normal 27B. Only 16 of 64 layers do full GQA (4 KV heads, d=256). That is 64KB per token at FP16: 0.5GB at 8k, 2GB at 32k. DeltaNet state is basically free.

Host for the trace hour: full GGUF in RAM or a fast SSD mmap, call it 32GB system RAM. Prefill is chunked (512-1024) so activations do not blow the budget.

## Model spine (do not prune)

- All 16 Gated Attention layers
- The 4 KV heads
- First two and last two layers

These are not scored for deletion.

## Sections we can cut

A **section** is one of three things:

1. **FFN channel** — one of the 17408 intermediate slots in a layer. This is the big cut. Real weight lives in the FFNs (~60%).
2. **DeltaNet pack** — one of the three linear-attention blocks in each 4-layer group. Keep or drop as a unit. 8 packs dropped is safe without recovery. 16 needs a short recover.
3. **Vocab row** — one embed row plus its `lm_head` twin. Vocab is 248k. Embed plus `lm_head` is about 2.5B params if untied. First thing to strip for a job.

Free cuts when the job allows: vision tower (text-only), unused vocab rows, MTP heads if we skip speculative decode.

## Score

Session energy, not weight magnitude.

The prompt is the job prior. The hour is the evidence. We do not prune mid-session. We emit a keep-mask after.

### FFN channels

While the layer is still on chip, keep three running numbers per channel:

- `n_fired`
- sum of squares
- max abs

Dead = never fired. Weak = bottom energy. Cut dead first, then weak, until the remnant ceiling.

Floor: a channel that fired even once on a special or high-loss token stays. An hour can miss a rare-but-critical path.

Width-cut from a job: 25% is clean, 35-40% if we recover.

### DeltaNet packs

Residual influence. If hidden-in and hidden-out stay near-identical all hour, that pack is a no-op for this job.

### Vocab

Keep anything that showed up in the prompt, the output, or the top-k logits, plus a small reserved core so we do not lock the tokenizer. Tokenizer IDs stay original. Remap on the embed gather and the `lm_head` write.

## Two loaders

### Trace path (host prunes)

Streaming run, not a resident one.

Pin: CUDA, the 16 Gated Attention layers, KV, DeltaNet state, embed.

Stream: FFNs (the 8+GB) from host RAM or NVMe mmap, double-buffered over PCIe 5.0. A Q4 FFN layer is about 150MB. Prefetch layer n+1 while n computes.

Peak weight VRAM around 3-4GB plus growing KV. 32k during the trace is fine. Expect single-digit to low-teens tok/s, 20k+ tokens in an hour.

Do not load the vision tower unless the session actually sends an image.

Trace at Q3 if more of the stack should stay resident. Serve later at Q4 from the packed remnant.

v1: stream all, score, cut after. No mid-session shrink.

Hooks fire while the FFN is still on chip, before evict. Per-channel stats via a warp reduce into a host ring. Hour ends, dump one prune table.

### Serve path (reuse the remnant)

Packed remnant only. Surviving tensors, densely laid out. No offload. One GGUF: new tensor shapes plus a KV block for keep-channels, keep-packs, and the vocab remap.

Zeros still sit in VRAM if you load a full GGUF and mask. Do not do that.

## Export

Hour-end dump in, one packed GGUF out. Prune table baked into the file so the remnant and the map cannot drift.

- Attention copies through
- FFN packs down to keep-channels
- Dead DeltaNet packs get dropped
- Embed and `lm_head` get remapped
- Tokenizer IDs stay original

Q4_K superblocks will not line up with arbitrary channels, so dequant, gather, then requant.

## v1 product shape

1. User gives a job prompt.
2. Host runs the hour (streamed).
3. Emit keep-mask / prune table.
4. Export packed GGUF.
5. Reuse that remnant for the job on the 5080.
