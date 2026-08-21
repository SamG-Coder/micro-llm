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
| Remnant ceiling, 8-16k | <=12GB **weights** |
| Remnant ceiling, 32k + vision | <=10GB **weights** |

12GB / 10GB is weights only. CUDA plus KV still sit on top.

KV is cheaper than a normal 27B. Only 16 of 64 layers do full GQA (4 KV heads, d=256). That is 64KB per token at FP16: 0.5GB at 8k, 2GB at 32k. DeltaNet state is basically free.

FP8 KV on Blackwell cuts the 32k bill from 2GB to 1GB. That is cheaper than cutting more FFN.

Host for the trace hour: full GGUF in RAM or a fast SSD mmap, call it 32GB system RAM. Prefill is chunked (512-1024) so activations do not blow the budget.

Packed tensors need 256-byte alignment or the 5080 kernels stall.

## Model spine (do not prune)

Spine is the Gated Attention **block** (QKVO plus the 4 KV heads), not the whole layer.

- All 16 Gated Attention blocks
- The 4 KV heads
- First two and last two layers (floor)

The FFN after those 16 still gets scored and width-cut. Skip those FFNs and we leave about 4B params on the table. The 12GB remnant gets a lot harder.

These spine pieces are not scored for deletion.

## Sections we can cut

A **section** is one of three things:

1. **FFN channel** ? one index across gate, up, and down in a layer (17408 intermediate slots). One index so Export's gather stays aligned. This is the big cut. Real weight lives in the FFNs (~60%), including the FFNs after the 16 Gated Attention blocks.
2. **DeltaNet pack** ? one of the three linear-attention blocks in each 4-layer group. Keep or drop as a unit. 8 packs dropped is safe without recovery. 16 needs a short recover.
3. **Vocab row** ? one embed row plus its `lm_head` twin. Vocab is 248k. Embed plus `lm_head` is about 2.5B params if untied. First thing to strip for a job.

Free cuts when the job allows: vision tower (text-only), unused vocab rows, MTP heads if we skip speculative decode.

## Score

Session energy, not weight magnitude.

The prompt is the job prior. The hour is the evidence. We do not prune mid-session. We emit a keep-mask after.

### FFN channels

While the layer is still on chip, keep three running numbers per channel:

- `n_fired`
- sum of squares
- max abs

**Fired** means `|SiLU(gate) * up|` above eps. Dead = never fired. Weak = bottom energy. Cut dead FFN, then dead packs, then unused vocab, then weak FFN until the remnant ceiling. Weak cap is 25% (keep >= 13056 of 17408); recover allows 40% (keep >= 10445). A layer with total `n_fired == 0` and no floor bits is a missing hook: keep all 17408.

Floor: a channel that fired even once on a special or high-loss token stays. An hour can miss a rare-but-critical path.

Width-cut from a job: 25% is clean, 35-40% if we recover.

### DeltaNet packs

Residual influence. If hidden-in and hidden-out stay near-identical all hour, that pack looks like a no-op.

A pack is **not** dead if it spiked even a few times, even if the hour average looks like identity.

### Vocab

Keep anything that showed up in the prompt, the output, or the top-k logits, plus a small reserved core so we do not lock the tokenizer. Tokenizer IDs stay original. Remap on the embed gather and the `lm_head` write.

## Two loaders

### Trace path (host prunes)

Streaming run, not a resident one.

Pin: CUDA, the 16 Gated Attention blocks (QKVO + KV), KV cache, DeltaNet state, embed.

Stream: FFNs (the 8+GB), including FFNs after the 16 Gated Attention blocks, from host RAM or NVMe mmap, double-buffered over PCIe. A Q4 FFN layer is about 150MB. Double-buffer peak is 300MB, not 150. Prefetch layer n+1 while n computes.

Prefetch only overlaps if the host pages are pinned. Unpinned pages serialize the copy.

A PCIe 4 board makes the hour slower. It does not OOM.

Peak weight VRAM around 3-4GB plus growing KV. 32k during the trace is fine. Expect single-digit to low-teens tok/s, 20k+ tokens in an hour.

Do not load the vision tower unless the session actually sends an image.

Trace at Q3 if more of the stack should stay resident. Serve later at Q4 from the packed remnant.

v1: stream all, score, cut after. No mid-session shrink.

Hooks fire while the FFN is still on chip, before evict. Per-channel stats via a warp reduce into a host ring. Hour ends, dump one prune table.

### Serve path (reuse the remnant)

Packed remnant only. Surviving tensors, densely laid out, 256-byte aligned. No offload. One GGUF: new tensor shapes plus a KV block for keep-channels, keep-packs, and the vocab remap.

12GB is the weight budget. CUDA plus KV still sit on top. Prefer FP8 KV on Blackwell before cutting more FFN.

Zeros still sit in VRAM if you load a full GGUF and mask. Do not do that.

## Export

Hour-end dump in, one packed GGUF out. Prune table baked into the file so the remnant and the map cannot drift.

- Attention copies through
- FFN packs down to keep-channels (one index across gate, up, and down)
- Dead DeltaNet packs get dropped
- Embed and `lm_head` get remapped
- Tokenizer IDs stay original

Q4_K superblocks will not line up with arbitrary channels, so dequant, gather, then requant Q4_K. Default remnant stays Q4 (`micro_llm.serve_ok=true`). `--q4-k-to-f16` is host debug only — F16 of a 75% FFN is ~25GB and must not load on the 5080.

## v1 product shape

1. User gives a job prompt.
2. Host runs the hour (streamed).
3. Emit keep-mask / prune table.
4. Export packed GGUF.
5. Reuse that remnant for the job on the 5080.

## Streamer traps (do not skip)

- Tap |SiLU(gate)*up| from the live forward. Do not dequant the whole FFN to FP16. Do not materialize a chunk-by-17408 scratch. Per-token warp reduce, then evict.
- lm_head only at logits. Do not pin it next to embed.
- Prefetch only overlaps if host pages are pinned. Double-buffer peak is 300MB, not 150.

## Scoring traps (do not skip)

- Spine is the Gated Attention block (QKVO plus the 4 KV heads), not the whole layer. The FFN after those 16 still width-cuts.
- Pack ids are global 0..47, not 0..2 inside the group. See PRUNE_TABLE.pack-id-fix.md.
- First two and last two means do not drop the layer. The FFN there still width-cuts.
- High-loss floor is a per-token fired bitset (~140KB), OR into the floor after logits if that token was special or high-loss.
- A DeltaNet pack is not dead if n_spike > 0, even if the hour average looks like identity.

## MLPT dump

Hour-end file is little-endian, magic MLPT, 80-byte header, then 64x17408 channel rows, packs 0..47, floor bitset, vocab bitset. Exact layout in PRUNE_TABLE.md. C++ dumps scores only. Export cuts and packs.
