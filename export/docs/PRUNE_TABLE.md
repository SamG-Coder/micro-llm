# Prune table

Hour-end dump. One file. This is the contract between the trace hooks and the packed GGUF writer.

v1: stream all, score, cut after. No mid-session shrink.

File is little-endian. Magic `MLPT`. Version `1`. Product 27B size is **17,997,328** bytes.

## Binary layout (v1)

### Header (80 bytes)

| Offset | Type | Field | Value |
| ---: | --- | --- | --- |
| 0 | char[4] | magic | `MLPT` |
| 4 | u32 | version | `1` |
| 8 | u32 | n_layers | `64` |
| 12 | u32 | n_ffn_channels | `17408` |
| 16 | u32 | n_packs | `48` |
| 20 | u32 | vocab_size | `248320` |
| 24 | u32 | tensor_align | `256` |
| 28 | u32 | header_size | `80` |
| 32 | f32 | fire_eps | fired if `\|SiLU(gate)*up\| > eps` |
| 36 | f32 | spike_eps | spike if `\|hidden_out-hidden_in\| > eps` |
| 40 | u64 | n_tokens | tokens that reached `after_logits` |
| 48 | u32 | flags | bit0 = floor bitset present |
| 52 | u32[7] | reserved | zero |

`tensor_align` is the packed-tensor alignment Export must use (256 bytes).

### Channel stats

Immediately after the header: `n_layers * n_ffn_channels` records, row-major
`index = layer * 17408 + channel`.

Each record is **16 bytes**:

| Offset | Type | Field |
| ---: | --- | --- |
| 0 | u64 | `n_fired` |
| 8 | f32 | `sumsq` of `SiLU(gate)*up` |
| 12 | f32 | `maxabs` of that activation |

`channel` is **one index** across gate, up, and down. Export gathers that
same index from all three.

Fired means `|SiLU(gate)*up| > fire_eps`. Dead = `n_fired == 0`.
Weak = lowest energy among survivors.

Floor: if it fired even once on a special or high-loss token, it stays
(bitset below), even if energy is zero.

The 16 Gated Attention **blocks** (QKVO + 4 KV heads) are not in this table.
The FFN after those 16 **is** (layers 3, 7, ..., 63 still have channel rows).

### Pack stats

Next: **48 records**, index = **global pack id 0..47**.

Each record is **24 bytes**:

| Offset | Type | Field |
| ---: | --- | --- |
| 0 | u32 | `pack` (must equal the index, 0..47) |
| 4 | u32 | `layer` = `4*group + slot` |
| 8 | u64 | `n_spike` |
| 16 | f64 | `sumsq_residual` of `\|hidden_out-hidden_in\|` |

`n_spike > 0` means not dead. Pack `p` lives on layer `4 * (p // 3) + (p % 3)`.
Do **not** use `0..2` inside a group ? that collides 16 rows onto the same id.

A pack is not dead if `n_spike > 0`, even if the hour average looks like identity.

Export drops every DeltaNet mixer tensor on a layer whose global pack id is
absent from `keep_packs`. Dead packs are omitted entirely, not zero-filled.

### Floor bitset

Next: `64 * 17408 / 8 = 139264` bytes (~140KB) when flags bit0 is set.

Bit `layer * 17408 + channel` is 1 if that channel **MUST keep** (fired on a
special or high-loss token). Bit `i` lives in byte `i >> 3`, bit `(i & 7)`.

A floor bit forces keep even if the channel is weak or `n_fired == 0`.

### Vocab bitset

Next: `(248320 + 7) / 8 = 31040` bytes.

Bit `token_id` is 1 if the original tokenizer ID showed up in the prompt, the
sampled output, or top-k logits, or is in the reserved core.

Tokenizer IDs stay original. Remap is embed gather + `lm_head` write only.

### Sizes (27B product)

```
header          80
channels        64 * 17408 * 16  = 17,825,792
packs           48 * 24          =      1,152
floor bitset                     =    139,264
vocab bitset                     =     31,040
---------------------------------------------
total                            = 17,997,328 bytes
```

## What the hooks write (semantics)

### FFN channels

One row per `(layer, channel)`.

| Field | Meaning |
| --- | --- |
| `layer` | Layer index, 0..63 |
| `channel` | Intermediate slot, 0..17407 |
| `n_fired` | Count of tokens where `|SiLU(gate) * up| > eps` |
| `sumsq` | Sum of squares of that activation |
| `maxabs` | Max abs of that activation |

### DeltaNet packs

One row per pack. Keep or drop as a unit. Pack ids are **global 0..47**.

| Field | Meaning |
| --- | --- |
| `pack` | Global pack id, `0..47`. Maps to layer `4 * group + slot` with `group = pack // 3` and `slot = pack % 3`. |
| `n_spike` | Times residual `|hidden_out - hidden_in|` exceeded eps |
| `sumsq_residual` | Sum of squares of that residual |

### Vocab

Bitset, original tokenizer IDs. Reserved core is whatever bits the hooks set.

## What the cut emits

After the hour, collapse scores to a keep-mask against the remnant ceiling
(12GB weights for 8-16k, 10GB if 32k plus vision). CUDA and KV sit on top.

```
keep_channels[layer] = sorted unique channel ids
keep_packs[]         = global pack ids to keep (0..47)
vocab_remap          = old_id -> dense remnant row
```

`keep_packs` is the global id space. Layer for pack `p` is `4 * (p // 3) + (p % 3)`.

Cut dead FFN first (`n_fired == 0` AND not floor), then dead DeltaNet packs
(`n_spike == 0`), then unused vocab (bitset 0), then weak FFN (lowest energy
among survivors) only if still over the ceiling. Weak cap is 25% (keep >=
13056 of 17408); `--recover` allows 40% (keep >= 10496). Raise
`CutCeilingError` if still over. Never hollow a layer to 1 channel. If a
layer's total `n_fired` is 0 and it has no floor bits, keep all 17408
(missing hook — never `kept=[0]`). bytes/param is `17.1/28 ≈ 0.61`, or
`source_file_size / n_params` when `--model` is given. Never drop first two
or last two layers as layers (FFN still width-cuts). Never drop the 16
Gated Attention blocks.

### Ranking rule (weak channels)

`energy = sumsq`. If `sumsq == 0` and `n_fired > 0`, `energy = n_fired * 1e-12`.
Weak cut drops lowest energy among **non-floor** survivors. Ties:
`(energy, n_fired, maxabs, layer, channel)`.

Floor bit forces keep even if energy is zero. Dead non-floor is always cut.

## What Export bakes into the GGUF

One packed file. The keep-mask lives in a KV block so remnant and map cannot drift.

| KV key | Type | Meaning |
| --- | --- | --- |
| `micro_llm.version` | u32 | `1` |
| `micro_llm.tensor_align` | u32 | `256` |
| `micro_llm.keep_packs` | i32[] | global pack ids |
| `micro_llm.keep_channels.n` | i32[] | `len(keep_channels[layer])` |
| `micro_llm.keep_channels.ids` | i32[] | concatenated channel ids |
| `micro_llm.vocab_remap.old_ids` | i32[] | original tokenizer ids, row order |
| `micro_llm.vocab_remap.rows` | i32[] | dense remnant rows (0..N-1) |
| `micro_llm.keep_vision` | bool | v1 default false |
| `micro_llm.keep_mtp` | bool | v1 default false |
| `micro_llm.serve_ok` | bool | `false` on `--q4-k-to-f16` (host debug); `true` on a real Q4 remnant. C++ serve refuses unless present and true. |
| `micro_llm.cuda_scratch_bytes` | u64 | `0.9` GiB as integer bytes |
| `micro_llm.kv_bytes_per_token_fp16` | u64 | `65536` |
| `micro_llm.kv_bytes_per_token_fp8` | u64 | `32768` |
| `micro_llm.serve_usable_bytes` | u64 | `15.2` GiB headless, integer bytes |
| `micro_llm.weight_bytes` | u64 | written first as `cut.estimate_weight_bytes` (KV is before tensors); patched after stream-write to on-disk file size. Serve prefers file size over this estimate. |

- Attention / Gated Attention (QKVO + 4 KV heads) copy through
- FFN packed to `keep_channels` (same index, gate / up / down)
- Dead DeltaNet packs dropped (not zero-filled)
- Embed and `lm_head` remapped (both if untied)
- Vision tower and MTP heads omitted unless those flags are set
- Every packed tensor 256-byte aligned

Serve path reads new shapes plus that KV block. No full-GGUF-plus-mask.

Q4_K: do **not** slice quantized blocks in place. Default path is dequant →
gather → real Q4_K requant (`export.quant.requantize_q4_k`, llama.cpp
`quantize_row_q4_K_ref`) → write Q4_K with `micro_llm.serve_ok=true`. The
serve remnant stays Q4 so the 12GB / 0.61 estimator still holds.
`--q4-k-to-f16` is host debug only (F16 after gather, `serve_ok=false`).
C++ serve refuses unless `micro_llm.serve_ok` is present and true. F16 of a
75% FFN is ~25GB — do not load that on the 5080.
