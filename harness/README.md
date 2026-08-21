# micro-llm-harness (v1 trace streamer)

C++ trace streamer + prune-table dump for [micro-llm](https://github.com/SamG-Coder/micro-llm).
This tree is the host-side scorer. It is **not** a 27B inference engine and it
does not fork llama.cpp.

Give a future live forward these hooks. Stream all, score, cut after. Hour-end
writes **one** prune table for LLM Export.

MIT.

## What this tree owns

- Per-channel FFN energy: `n_fired`, `sumsq`, `maxabs` where
  fired = `|SiLU(gate) * up| > eps`
- Per-token fired bitset (~140KB). OR into the high-loss floor **after logits**
- DeltaNet packs, **global** id `0..47` (`n_spike`, `sumsq_residual`)
- Vocab bitset of original tokenizer IDs (248320)
- Trace streamer control plane: pin CUDA + 16 Gated Attention blocks + KV +
  DeltaNet state + embed; two FFN scratch buffers; async prefetch of n+1;
  pinned host pages; `lm_head` only at logits; no mid-session shrink
- Hour-end binary prune table + C++ load/save API

It does **not** dequant a whole FFN to FP16, does **not** allocate a
`[chunk x 17408]` scratch, and does **not** pin `lm_head` next to embed.

## Layout

```
include/micro_llm/
  types.hpp          constants, pack-id map, 256-byte align
  prune_table.hpp    table + load/save
  trace_hooks.hpp    what a future forward calls
  streamer.hpp       pin / double-buffer / prefetch / logits
  ffn_reduce.hpp     per-token reduce (CPU, CUDA optional)
  micro_llm.hpp      umbrella
src/
  prune_table.cpp
  trace_hooks.cpp
  streamer.cpp
  ffn_reduce.cpp     CPU fallback
  ffn_reduce.cu      CUDA stub (not required to build)
  cli_dump.cpp       synthetic traffic ? one prune table
tests/               serialize, pack id, channel align, floor OR, dead/spike, serve_ok
docs/PRUNE_TABLE.md  exact dump format for LLM Export
docs/PRUNE_TABLE.pack-id-fix.md
```

## Build and test

Needs C++17. CUDA / nvcc is optional.

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CPU-only is the default. To try the CUDA kernel when `nvcc` is present:

```bash
cmake -S . -B build -DMICRO_LLM_CUDA=ON
```

CLI (synthetic hooks, no weights):

```bash
./build/micro-llm-dump --out prune_table.bin --tokens 8
```

## Where a future forward calls the hooks

The engine you plug this into still owns matmuls, attention, sampling.
Call the harness like this, **one token at a time**:

```cpp
#include "micro_llm/micro_llm.hpp"

micro_llm::TraceStreamer streamer;   // two ~150MB FFN scratches, peak ~300MB
micro_llm::TraceHooks hooks;         // fire_eps / spike_eps default 1e-6
streamer.begin_session();            // pins CUDA, 16 GA blocks, KV, DeltaNet, embed
hooks.mark_reserved_core(256);

for (token) {
    hooks.begin_token(t);
    for (layer in 0..63) {
        streamer.prefetch_ffn(layer + 1);          // async n+1, host pages pinned
        streamer.bind_ffn(layer);
        // live forward: compute this token's gate[17408], up[17408]
        // tap activations ? do not dequant the whole FFN to FP16
        hooks.on_ffn_activations(layer, gate, up); // warp-reduce, then evict
        streamer.evict_ffn(layer);
        if (micro_llm::is_delta_net_layer(layer)) {
            const uint32_t pack = micro_llm::pack_id_from_delta_layer(layer);
            hooks.on_delta_hidden(pack, hidden_in, hidden_out, 5120);
        }
    }
    hooks.on_vocab_id(sampled_id);
    hooks.on_topk_ids(topk, k);
    streamer.enter_logits();                       // lm_head only here
    // compute loss / special-token flag
    hooks.after_logits(t, special_or_high_loss);   // OR per-token bitset into floor
    streamer.leave_logits();
}

streamer.end_session();
micro_llm::save_prune_table(hooks.table(), "prune_table.bin");
```

Spine: Gated Attention **block** (QKVO + 4 KV heads) is not scored for drop.
The FFN after those 16 **is**. First two and last two layers are a no-drop
**layer** floor; their FFNs still width-cut.

Pack ids are global `0..47`. Do not use `0..2` inside a group
(see `docs/PRUNE_TABLE.pack-id-fix.md`).

## Dump format

`docs/PRUNE_TABLE.md` is the file format for LLM Export: header, channel
stats, 48 pack rows, floor bitset, vocab bitset. Packed tensors later need
256-byte alignment; that constant is `micro_llm::kTensorAlign` and
`kPackedAlign` on the types.
