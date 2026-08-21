# export ? packed GGUF writer

Host-side cut + pack. Reads one MLPT prune table and one full GGUF, writes one
packed remnant GGUF with the keep-mask baked into KV metadata.

```
python -m export.pack --model full.gguf --prune-table dump.mlpt --out remnant.gguf
python -m export.pack --model full.gguf --prune-table dump.mlpt --out remnant.gguf --ceiling-gb 10
```

`--prune-table` may also be a JSON keep-mask (already cut) for debugging.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--ceiling-gb` | `12` | remnant *weight* ceiling in GiB (use `10` for 32k+vision) |
| `--keep-vision` | off | v1 omits the vision tower unless set |
| `--keep-mtp` | off | v1 omits MTP / nextn heads unless set |
| `--q4-k-to-f16` | off | after Q4_K dequant+gather, emit F16 (no fake requant) |
| `--table-json` | ? | dump the cut keep-mask as JSON |

## Layout

```
export/
  mlpt.py         little-endian MLPT reader / writer
  cut.py          scores ? keep_channels / keep_packs / vocab_remap
  pack.py         CLI
  writer.py       gather + 256-byte-aligned GGUF write
  gather.py       same channel index on gate / up / down
  names.py        llama.cpp QWEN35 tensor names + pack-id map
  quant.py        F16/F32 pack; Q4_K dequant + requant hook
  prune_table.py  keep-mask + GGUF KV encode/decode
```

## Ranking rule (weak FFN channels)

Documented in `export/cut.py` as `RANKING_RULE`:

```
energy = sumsq
if sumsq == 0 and n_fired > 0:
    energy = n_fired * 1e-12
```

Cut order: dead FFN (`n_fired == 0` AND not floor) ? lowest-energy non-floor
survivors (only if still over the ceiling) ? dead DeltaNet packs (`n_spike == 0`,
except protected layers 0, 1, n-2, n-1) ? unused vocab (bitset 0).

Floor bit forces keep even if energy is zero. Ties:
`(energy, n_fired, maxabs, layer, channel)`.

## Tensor names (llama.cpp `LLM_ARCH_QWEN35`)

C++ serve loader must match these strings. Names follow
`gguf.constants.TENSOR_NAMES` for `qwen35` / Qwen3.5 / 3.6 hybrid GGUF.

64-layer 27B: groups of 4 = 3 DeltaNet + 1 Gated Attention.

```
layer = 4 * group + slot
slot 0,1,2  ? DeltaNet pack, global id group*3 + slot   (0..47)
slot 3      ? Gated Attention (never dropped)
pack p      ? layer 4*(p//3) + (p%3)
```

| Role | GGUF name | Pack behaviour |
| --- | --- | --- |
| embed | `token_embd.weight` | gather vocab rows |
| lm_head | `output.weight` | gather vocab rows if untied |
| output norm | `output_norm.weight` | copy |
| FFN | `blk.{i}.ffn_gate.weight` | gather `keep_channels[i]` on n_ff axis |
| FFN | `blk.{i}.ffn_up.weight` | same index |
| FFN | `blk.{i}.ffn_down.weight` | same index (typically axis 1) |
| Gated Attn QKVO | `blk.{i}.attn_q.weight` | copy through (layers 3,7,...,63) |
| | `blk.{i}.attn_k.weight` | copy |
| | `blk.{i}.attn_v.weight` | copy |
| | `blk.{i}.attn_output.weight` | copy |
| | `blk.{i}.attn_q_norm.weight` | copy |
| | `blk.{i}.attn_k_norm.weight` | copy |
| norms | `blk.{i}.attn_norm.weight` | copy (layer stays) |
| | `blk.{i}.post_attention_norm.weight` | copy |
| DeltaNet mixer | `blk.{i}.attn_qkv.weight` | drop whole pack if id ? `keep_packs` |
| | `blk.{i}.attn_gate.weight` | drop with pack |
| | `blk.{i}.ssm_in.weight` | drop with pack |
| | `blk.{i}.ssm_ba.weight` | drop with pack |
| | `blk.{i}.ssm_conv1d.weight` | drop with pack |
| | `blk.{i}.ssm_dt.bias` | drop with pack |
| | `blk.{i}.ssm_a` | drop with pack (no `.weight` suffix) |
| | `blk.{i}.ssm_beta.weight` | drop with pack |
| | `blk.{i}.ssm_alpha.weight` | drop with pack |
| | `blk.{i}.ssm_norm.weight` | drop with pack |
| | `blk.{i}.ssm_out.weight` | drop with pack |
| Vision | `v.*` / `mm.*` / `vision.*` / `clip.*` | omit unless `keep_vision` |
| MTP | `*.nextn.*` / `shared_head*` / `blk.{i}` with i ? n_layer | omit unless `keep_mtp` |

`export/names.py` is the name helper. `is_deltanet_mixer` treats any
`ssm_*`, `attn_qkv*`, `attn_gate*` suffix as pack-owned.

GGUF storage (numpy view after `GGUFReader`):

```
ffn_gate / ffn_up : (n_ff, n_embd)   ? gather axis 0
ffn_down          : (n_embd, n_ff)   ? gather axis 1
token_embd / output : (n_vocab, n_embd) ? gather axis 0
```

`gather.py` locates the `n_ff` / `n_vocab` axis rather than hard-coding, so
either layout works. Gate, up, and down **must** use the same channel ids.

## Alignment

Every packed tensor payload starts on a 256-byte boundary (`general.alignment = 256`
and `micro_llm.tensor_align = 256`).

## Q4_K

Do not slice Q4_K superblocks in place. v1:

- F16 / F32: gather and write, fully supported
- Q4_K copy-through (attention, no slice): raw bytes copied
- Q4_K FFN / vocab: `dequantize` ? gather ? `requantize_q4_k` hook
- Hook raises `Q4KRequantNotImplemented`. Pass `--q4-k-to-f16` to emit F16
  after gather instead of faking a Q4_K write.

## Tests

From export/, with numpy, gguf, and pytest:

```
pytest -q
```
