# Hook ring (per token)

Live flash for the hotspot view. One record per token. C++ writes. Hotspot paints. Not an MLPT. Not a keep-mask. Not a budget.

Session heat, `n_fired`, `sumsq`, and `maxabs` stay in the MLPT dump. The keep-mask stays in Export. Gigabytes stay on Memory's 15.2 bar. This file is this-token only.

## What it is

This token's fired FFN channels, pack residuals, and vocab ids. Paint as a flash, then fade into session heat. Session heat is the MLPT. Do not accumulate this ring into a second heat store.

## Hard cap (Memory)

Ring depth is 64 (~8.7MB). Do not keep 20k records (that is 2.8GB). Session heat is the MLPT. The ring is the live window only. Gigabytes stay on the budget bar.

## Record

Little-endian. One record after a small header if you batch, or a single struct on a ring of 64.

Header (optional, 16 bytes) if dumped as a file:

- magic `HTR1` (4 bytes)
- u32 version = 1
- u32 n_layers = 64
- u32 n_ffn = 17408

Per-token record:

| Field | Type | Meaning |
| --- | --- | --- |
| `token_index` | u32 | tokens that have reached `after_logits` |
| `sampled_id` | u32 | original tokenizer id |
| `flags` | u32 | bit0 = `special_or_high_loss` |
| `n_topk` | u32 | count of top-k ids, max 64 |
| `topk` | u32[64] | original tokenizer ids; unused slots 0 |
| `fire_eps` | f32 | same as MLPT, default 1e-6 |
| `spike_eps` | f32 | 0.02 on relative residual |
| `ffn_fired` | bitset 64*17408 bits = 139264 bytes | bit `layer*17408+channel` is 1 if `|SiLU(gate)*up| > fire_eps` on THIS token |
| `pack_rel_residual` | f32[48] | `|out-in|/|in|` for packs 0..47 this token |
| `pack_spike` | u64 | bit p is 1 if `pack_rel_residual[p] > spike_eps` |

Sizes:

- bitset 139264
- residuals 192
- spike mask 8
- rest ~280
- record ~139.7 KB
- ring of 64 ~ 8.7 MB

## IDs

- Channel is one index across gate/up/down, 0..17407
- Pack is global 0..47, layer = `4*group+slot`
- Tokenizer IDs stay original

## What Hotspot paints from this

- Flash FFN heat-strip bins that have any fired bit this token
- Pulse pack nodes whose spike bit is set
- Spark the vocab strip on `sampled_id` and top-k (rarer ids brighter)
- Spine (16 Gated Attention blocks) always lit, not from this ring
- Unwired is NOT decided here. Unwired is a layer whose MLPT total `n_fired` is 0. During a live token, a quiet layer is just not-this-token.

## What does not belong

- `n_fired`, `sumsq`, `maxabs` (MLPT)
- `keep_channels` / `keep_packs` / `vocab_remap` (Export after cut)
- file bytes, CUDA, KV GB (Memory 15.2 bar)
- 20k-token archive (2.8GB). Ring depth 64 is the cap
- 17408 cubes
- fake glow
- a sample `setInterval` while live frames are arriving

## Defaults

`fire_eps` 1e-6. `spike_eps` 0.02. top-k 64 max, 50 is fine. Ring depth 64 (hard cap, ~8.7MB).
