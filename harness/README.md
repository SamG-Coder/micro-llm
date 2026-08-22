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
- Native hotspot window: no args or `--ui` opens the committed `ui/`
  sample map (no GGUF). `--model` starts the hour and opens the live
  window; `--ui --model` is the same. Llama decode runs on a worker so
  the window can pump. One HTR1 record per generated token.

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
  hook_ring.hpp      HTR1 encode (this-token fired bits + pack + vocab)
  trace_cli.hpp      --ui is not exclusive of --model; hour defaults
  hotspot_ui.hpp     locate committed UI; Windows WebView2 host
  micro_llm.hpp      umbrella
src/
  cli_dump.cpp       synthetic traffic → one prune table
  cli_trace.cpp      micro-llm-trace (sample window; --model starts the hour)
  cli_view.cpp       micro-llm-view (window only; WIN32 GUI on Windows)
  hook_ring.cpp      HTR1 record + overwrite ring
  trace_cli.cpp      CLI parse / hybrid pin patterns
  hotspot_ui.cpp     UI path lookup
  hotspot_ui_win32.cpp  WebView2 + live host messages (MSVC / Windows)
  llama_forward.cpp  llama.cpp attach (ifdef MICRO_LLM_HAS_LLAMA)
ui/                  committed static hotspot map (no npm at runtime)
tests/               serialize, pack id, floor, dead/spike, hooked, serve, live, ui, ring, cli
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

`--model PATH.gguf` starts the hour and opens the live window (default).
`--ui --model PATH.gguf` is the same: window AND decode. `--ui` is not
exclusive. When `--model` is set without `--n-predict`, n-predict is 20000
(the CLI/struct default stays 64 for `--stub` tests). The hour continues
after EOS. `--out` writes MLPT scores; every 2000 tokens the table is
written to `<out>.tmp` and atomically renamed to `<out>` (no mid-run pack).

Hybrid pin is tensor overrides, not ngl. `n_gpu_layers=0` (CPU default).
Not `ngl=16` (wrong 16 layers) and not `ngl=99` (parks the file).
Gated Attention on layers 3,7,...,63 plus KV stay CUDA. Park as many FFN
layers as fit under 14GB (GA+KV ~6.9 + 0.9 + parked + one stream slot).
Never all 64. Stream the rest one layer at a time. DeltaNet stays host.
MTP off. `block_count=65` / `nextn_predict_layers=1` is the extra MTP
block — hook ring stays 64. Flash-attn stays off. `op_offload` does not
move FFN compute (30328). `n_batch=512`, `n_ubatch=32`. Device fire tap
on live CUDA gate/up; do not dequant the layer to host just to score it.
Stderr proof: `ffn_cuda_park`, `ffn_cuda_bind`, `tokens/s=`. Hotspot
shows tok/s. Checkpoint is the ~18MB MLPT; do not pack it.

Windows (MSVC) hosts `ui/` in WebView2 (`WebView2Loader.dll` is copied next
to the exe). The page is a dark star field: quiet token streaks + dim glyphs,
spine as a dim point (never off, never a cube), packs as brief points/streaks
from spike bits, fired FFN bits as points/streaks in the layer volume. No bin
grid. No bloom pass.

**Evergreen WebView2 runtime** is required. Windows 11 and any machine with
current Microsoft Edge usually already have it. If the window fails to
create, install the Evergreen bootstrapper:

https://go.microsoft.com/fwlink/p/?LinkId=2124703

Docs: https://developer.microsoft.com/en-us/microsoft-edge/webview2/

```text
micro-llm-trace.exe
micro-llm-trace.exe --ui
micro-llm-trace.exe --ui --model C:\models\qwen.gguf --out hour.mlpt --n-predict 20000
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

3. Run the hour. Hybrid pin: ngl=0, 16 GA + KV on CUDA, as many FFN
   layers as fit parked on CUDA, unparked FFN streamed one at a time.
   Window opens on Windows. Decode is on a worker thread. Look for
   `ffn_cuda_park` / `ffn_cuda_bind` / `tokens/s=` on stderr.

```bash
./harness/build/micro-llm-trace \
  --ui --model /path/to/qwen3.8-27b-q4_k_m.gguf \
  --out hour.mlpt \
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
`result_output`). After each FFN: device hook on gate/up while that layer's workspace is
on the card, then evict. After each
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
