# Prune table

Hour-end dump. One file. This is the contract between the trace hooks and the packed GGUF writer.

v1: stream all, score, cut after. No mid-session shrink.

## What the hooks write

### FFN channels

One row per `(layer, channel)`.

`channel` is one index across gate, up, and down. Export gathers that same index from all three.

| Field | Meaning |
| --- | --- |
| `layer` | Layer index, 0..63 |
| `channel` | Intermediate slot, 0..17407 |
| `n_fired` | Count of tokens where `\|SiLU(gate) * up\| > eps` |
| `sumsq` | Sum of squares of that activation |
| `maxabs` | Max abs of that activation |

Fired means `|SiLU(gate) * up|` above eps. Dead = `n_fired == 0`. Weak = bottom energy among survivors.

Floor: if it fired even once on a special or high-loss token, it stays.

The 16 Gated Attention **blocks** (QKVO + 4 KV heads) are not in this table. The FFN after those 16 **is**.

### DeltaNet packs

One row per pack. Keep or drop as a unit.

| Field | Meaning |
| --- | --- |
| `pack` | Pack id in the 4-layer group |
| `n_spike` | Times residual `\|hidden_out - hidden_in\|` exceeded eps |
| `sumsq_residual` | Sum of squares of that residual |

A pack is not dead if `n_spike > 0`, even if the hour average looks like identity.

### Vocab

Bitset, original tokenizer IDs.

Keep if the id showed up in the prompt, the sampled output, or top-k logits, plus a reserved core.

Tokenizer IDs stay original. Remap is embed gather + `lm_head` write only.

## What the cut emits

After the hour, collapse scores to a keep-mask against the remnant ceiling (12GB weights for 8-16k, 10GB if 32k plus vision). CUDA and KV sit on top of that.

```
keep_channels[layer] = sorted unique channel ids
keep_packs[]         = pack ids to keep
vocab_remap          = old_id -> dense remnant row
```

Cut dead FFN first, then weak, then dead DeltaNet packs (no spikes), then unused vocab. Never drop first two or last two layers. Never drop the 16 Gated Attention blocks.

## What Export bakes into the GGUF

One packed file. Prune table lives in a KV block so remnant and map cannot drift.

- Attention copies through
- FFN packed to `keep_channels` (same index, gate/up/down)
- Dead DeltaNet packs dropped
- Embed and `lm_head` remapped
- Tensors 256-byte aligned

Serve path reads new shapes plus that KV block. No full-GGUF-plus-mask.
