# micro-llm-harness (v1 trace streamer)

C++ trace streamer + prune-table dump for [micro-llm](https://github.com/SamG-Coder/micro-llm).
Host-side scorer plus a llama.cpp attach for a **real** Qwen 27B (3.6/3.8 hybrid) hour.

MIT.

## What this tree owns

- Per-channel FFN energy: `n_fired`, `sumsq`, `maxabs` where
  fired = `|SiLU(gate) * up| > eps`
- Device tap: `on_ffn_activations_device(layer, d_gate, d_up)` on live CUDA
  pointers. Persistent reduce context. **No per-token cudaMalloc/H2D of gate/up.**
- Per-token fired bitset (~140KB). OR into the high-loss floor **after logits**
- DeltaNet packs, **global** id `0..47`. Score
  `r = ||out-in||_2 / (||in||_2 + 1e-12)`, default `spike_eps = 0.02`.
  Identity must not spike.
- MLPT flags bit1 + `u64 layer_hooked` trailer after vocab (header stays 80B).
  Unwired = `n_tokens>0 && bit unset`. Dead = hooked && all `n_fired==0`.
  Do not fake `n_fired`.
- Serve gate: GGUF KV `micro_llm.serve_ok`. `remnant_may_serve` is true only
  if the key is present and true. False = F16 host dump, refuse.
  Packed FFN keep width must be a multiple of 256 (Q4_K). 13056 and 10496
  are valid; 10445 is not.
- Vocab bitset of original tokenizer IDs (248320)
- Trace streamer control plane: pin CUDA + 16 Gated Attention blocks + KV +
  DeltaNet state + embed; two FFN scratch buffers; async prefetch of n+1;
  pinned host pages; `lm_head` only at logits; no mid-session shrink
- Live forward: `LlamaCppLiveForwardBackend` attaches to llama.cpp `cb_eval`
  on the `qwen35` graph. `StubLiveForwardBackend` is tests-only.
- CLI `micro-llm-trace` for the coding-assistant hour
- Native sample hotspot window: no args or `--ui` opens the committed
  `ui/` map. Does not load a GGUF. Does not start the hour.

It does **not** dequant a whole FFN to FP16, does **not** allocate a
`[chunk x 17408]` scratch, and does **not** pin `lm_head` next to embed.
It does **not** invent a from-scratch 27B engine.

## Layout

```
include/micro_llm/
  types.hpp          constants, pack-id map, 256-byte align, remnant_may_serve
  prune_table.hpp    table + load/save + layer_hooked
  trace_hooks.hpp    host + device FFN taps, relative DeltaNet
  streamer.hpp       pin / double-buffer / prefetch / logits
  ffn_reduce.hpp     persistent CUDA context + CPU fallback
  gguf_meta.hpp      KV probe (architecture, serve_ok, vision)
  serve.hpp          remnant_may_serve from a remnant GGUF
  graph_hooks.hpp    llama.cpp tensor-name matcher (compile-tested)
  live_forward.hpp   stub + llama.cpp backends
  hotspot_ui.hpp     locate committed UI; Windows WebView2 host
  micro_llm.hpp      umbrella
src/
  cli_dump.cpp       synthetic traffic → one prune table
  cli_trace.cpp      micro-llm-trace (window by default; hour with --model)
  cli_view.cpp       micro-llm-view (window only; WIN32 GUI on Windows)
  hotspot_ui.cpp     UI path lookup
  hotspot_ui_win32.cpp  WebView2 (MSVC / Windows)
  llama_forward.cpp  llama.cpp attach (ifdef MICRO_LLM_HAS_LLAMA)
ui/                  committed static hotspot map (no npm at runtime)
tests/               serialize, pack id, floor, dead/spike, hooked, serve, live, ui
```

## Build and test

Needs C++17. CUDA / nvcc and llama.cpp are optional. Default `cmake` + `ctest`
must pass **without** a 17GB GGUF.

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CPU-only is the default. CUDA kernel when `nvcc` is present:

```bash
cmake -S . -B build -DMICRO_LLM_CUDA=ON
```

## Hotspot window (sample map, no npm)

Opening the built exe with no args or `--ui` shows the sample hotspot map.
That path does **not** load a GGUF and does **not** start the hour.
`--n-predict` is unchanged (default 64) and only applies to `--model`.

Windows (MSVC) hosts `ui/` in WebView2 (`WebView2Loader.dll` is copied next
to the exe). The page is a 3D volume: tokens fly through the tower, spine
sparks then heat, packs pulse, FFN bins flash from fired bits on kept
channels only.

**Evergreen WebView2 runtime** is required. Windows 11 and any machine with
current Microsoft Edge usually already have it. If the window fails to
create, install the Evergreen bootstrapper:

https://go.microsoft.com/fwlink/p/?LinkId=2124703

Docs: https://developer.microsoft.com/en-us/microsoft-edge/webview2/

```text
micro-llm-trace.exe
micro-llm-trace.exe --ui
micro-llm-view.exe
micro-llm-trace.exe --ui-check
```

`ui/` is a committed Vite build of `viewer/`. Do not run `npm install` or
`npm start` to open the map. Rebuild the snapshot only if you change
`viewer/` (`viewer/scripts/sync-harness-ui.sh`).

On Linux the same flags locate the files and print that the native window
is Windows + WebView2.

## How to run the hour (local coding assistant)

Job is text-only. Do not load vision (no mmproj, no `v.*` tower).

1. Get a Qwen 3.6/3.8 27B GGUF (`general.architecture = qwen35`, 64 layers,
   hidden 5120, FFN 17408, vocab 248320). Q4_K_M is the usual trace quant.
2. Build harness against a llama.cpp that has `LLM_ARCH_QWEN35` / `qwen35.cpp`
   (Gated DeltaNet + Gated Attention). Do not vendor the whole tree into this
   repo if a checkout + thin adapter works:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp
cmake -S harness -B harness/build \
  -DMICRO_LLM_LLAMA=ON \
  -DMICRO_LLM_LLAMA_DIR=/opt/llama.cpp \
  -DMICRO_LLM_CUDA=ON
cmake --build harness/build -j
```

3. Run the hour. Streamer pins 16 GA + KV + DeltaNet state + embed, streams
   FFNs with two scratches + prefetch n+1, `lm_head` only at logits:

```bash
./harness/build/micro-llm-trace \
  --model /path/to/qwen3.8-27b-q4_k_m.gguf \
  --prompt "You are a local coding assistant. Work in this repo." \
  --out prune_table.bin \
  --n-predict 20000 \
  --ctx 8192
```

Without weights the CLI exits non-zero:

```text
error: no weights at /path/to/missing.gguf
```

Without llama.cpp linked, a valid GGUF is still probed, then the backend
**refuses to fake the hour**:

```text
error: Qwen 27B hybrid GGUF recognized, but llama.cpp is not linked. Rebuild: ...
```

`--stub` is tests-only. It does not produce a real 27B MLPT.

4. After the hour, Export cuts (not this tree): unused-vocab strip + no vision
   + ~8 dead packs, not an FFN-only 25% slash. Pack remnant, bake
   `micro_llm.serve_ok=true` on a real Q4. C++ serve:

```bash
./harness/build/micro-llm-trace --check-serve remnant.gguf
```

`remnant_may_serve` is true only if `micro_llm.serve_ok` is present and true.
False means F16 host dump — refuse. Packed FFN keep width
(`micro_llm.keep_channels.n`, else `*.feed_forward_length`) must be a
multiple of 256 (Q4_K superblock). 13056 (25% cap) and 10496 (40% recover
floor) are valid; 10445 is not.

## Attach point

Verified: llama.cpp `llama_context_params.cb_eval`
(`ggml_backend_sched_eval_callback`) after each graph node. `qwen35.cpp`
names the sites we need (`ffn_gate-L`, `ffn_up-L`, `attn_residual-L`,
`result_output`). After each FFN: device hook on gate/up. After each
DeltaNet pack: relative residual. Vocab: prompt, sampled, top-k.
`after_logits` for the floor bitset.

If this llama.cpp build cannot execute the hybrid graph (missing Gated
DeltaNet kernels), the backend reports `llama_decode` failure and does
**not** write a successful-looking hour.

## Dump format (harness writer)

Header stays **80 bytes**. After the vocab bitset: `u64 layer_hooked`
(bit L set when layer L's FFN hook ran). `flags` bit1 means the trailer
is present. Old 17,997,328-byte files still load (no trailer → all bits
unset). New files are 17,997,336 bytes.

Unwired = `n_tokens>0 && bit unset`. Dead = hooked && all `n_fired==0`.

Pack spike: `r = ||out-in||_2 / (||in||_2 + 1e-12) > spike_eps` (default 0.02).
