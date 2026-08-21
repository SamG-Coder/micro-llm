# Prune table dump format (v1)

Hour-end dump. **One file.** Contract between the trace hooks and LLM Export.

v1: stream all, score, cut after. No mid-session shrink.

File is little-endian. Magic `MLPT`. Version `1`.

C++ API: `micro_llm::save_prune_table` / `micro_llm::load_prune_table`
in `include/micro_llm/prune_table.hpp`.

## Header (80 bytes)

| Offset | Type | Field | Value |
| ---: | --- | --- | --- |
| 0 | char[4] | magic | `M L P T` |
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
The same constant lives on the C++ types as `kPackedAlign` / `kTensorAlign`.

## Channel stats

Immediately after the header: `n_layers * n_ffn_channels` records, row-major
`index = layer * 17408 + channel`.

Each record is 16 bytes:

| Offset | Type | Field |
| ---: | --- | --- |
| 0 | u64 | `n_fired` |
| 8 | f32 | `sumsq` of `SiLU(gate)*up` |
| 12 | f32 | `maxabs` of that activation |

`channel` is **one index** across gate, up, and down. Export gathers that
same index from all three.

Fired means `|SiLU(gate)*up| > fire_eps`. Dead = `n_fired == 0`.

The 16 Gated Attention **blocks** (QKVO + 4 KV heads) are not in this table.
The FFN after those 16 **is** (layers 3,7,...,63 still have channel rows).

First two and last two layers (0, 1, 62, 63): do not DROP the layer. FFN
still width-cuts; their channel rows are scored like every other layer.

## Pack stats

Next: 48 records, index = **global pack id 0..47**.

Each record is 24 bytes:

| Offset | Type | Field |
| ---: | --- | --- |
| 0 | u32 | `pack` (must equal the index, 0..47) |
| 4 | u32 | `layer` = `4*group + slot` |
| 8 | u64 | `n_spike` |
| 16 | f64 | `sumsq_residual` of `\|hidden_out-hidden_in\|` |

`n_spike > 0` means not dead. See [PRUNE_TABLE.pack-id-fix.md](PRUNE_TABLE.pack-id-fix.md).

## Floor bitset

Next: `64 * 17408 / 8 = 139264` bytes (~140KB).

Bit `layer * 17408 + channel` is 1 if that channel fired on a special or
high-loss token. The FFN hook cannot decide this (loss is unknown). The
hooks keep a per-token fired bitset of the same size and **OR it into this
floor after logits** when the token was special or high-loss.

A floor bit means Export must keep the channel even if energy is weak.

## Vocab bitset

Next: `(248320 + 7) / 8 = 31040` bytes.

Bit `token_id` is 1 if the original tokenizer ID showed up in the prompt,
the sampled output, or top-k logits, or is in the reserved core.

Tokenizer IDs stay original. Remap is embed gather + `lm_head` write only.

## Sizes

```
header          80
channels        64 * 17408 * 16  = 17,825,792
packs           48 * 24          =      1,152
floor bitset                     =    139,264
vocab bitset                     =     31,040
---------------------------------------------
total                            = 17,997,328 bytes
```

## What Export does with this file (not this tree)

Collapse scores to a keep-mask against the remnant ceiling. Cut dead FFN
first, then weak, then dead DeltaNet packs (`n_spike == 0`), then unused
vocab. Never drop first/last two layers. Never drop the 16 Gated Attention
blocks. Bake the keep-mask into the packed GGUF KV block. Tensors 256-byte
aligned.
