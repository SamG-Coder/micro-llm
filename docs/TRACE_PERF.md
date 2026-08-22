# Trace-hour decode performance

Target: Qwen 3.8 27B UD-Q4_K_M, **same GGUF, same prompt, ctx 4k/8k/16k (32k if it fits)**, several hundred generated tokens. Do not shrink the model. Do not drop to 2k context to fake tok/s.

This VM does **not** load the 17GB GGUF. Numbers below are (1) the 5080 host hour that already ran, and (2) a closed-form model fitted to that hour plus PCIe/GEMM first principles. The exe prints both the model and **measured** counters after a real run.

## Before (main after PRs 1–3, 6–11)

Host hour, RTX 5080 16GB, Qwen 27B UD-Q4_K_M:

| Knob | What happened |
| --- | --- |
| `ngl=99` + FFN/DeltaNet CPU overrides | `CUDA0 model buffer ≈ 2010 MiB`, `CPU_Mapped ≈ 14361 MiB` |
| Park FFN 0–29, then 41 layers | CUDA0 grew to **~6.7 GiB**, **tok/s stayed ~3.5** |
| Streamed FFN 30–63 | still **host ggml** |
| FA | off (FA + CPU FFN AV) |
| Hooks | D2H CUDA F32 + `cudaDeviceSynchronize` **per layer** (64/token) |
| UI | mutex queue + `PostWebMessageAsJson` in the token path |

Fitted wall (matches 3.5 tok/s):

```
23 host FFN * 5.3 ms     = 122 ms
48 host DeltaNet * 3.2 ms = 154 ms
16 GA CUDA * 0.5 ms       =   8 ms
64 layer sync * 0.08 ms   =   5 ms
                          -------
                            289 ms  →  3.46 tok/s
```

Park-41 without moving compute does not help: the 23 host FFN + 48 host DeltaNet still dominate.

**20 tok/s is impossible while DeltaNet stays on the host.** 48 * 3.2 ms = 154 ms floor (6.5 tok/s) even if every FFN is free.

## After (this PR)

| Change | Why |
| --- | --- |
| `ngl=0` + tensor overrides | not 16 (wrong 16), not 99 (parks the file) |
| DeltaNet weights on CUDA | proven required; host DeltaNet caps the hour |
| Park early FFN that fit under 14 GiB | persist hot tensors; never all 64 |
| Unparked FFN on **CUDA_Host** | GPU Q4_K GEMM, not host ggml. Prefetch N+1 |
| Embed CUDA_Host, `lm_head` at logits | do not spend VRAM on the 248k tables |
| MTP off, hook ring 64 | `block_count=65` is the extra block |
| FA on when compute is CUDA | `--no-flash-attn` if a build still AVs |
| Device fire tap, 1 sync/token | private stream; prune evidence kept |
| Lock-free HTR1 + 16 ms UI drain | WebView2 cannot stall decode |
| End-of-run telemetry | measured tok/s, H2D/D2H, VRAM, bottlenecks |

Planned wall (default Qwen 27B Q4_K_M catalog, ctx 8k, FP8-sized KV, PCIe 5.0 x16 ≈ 64 GB/s):

```
pcie_bytes/token = n_streamed * 150 MiB
H2D_ms           = pcie_bytes / 64e9
compute          = n_park * 0.35 ms + n_stream * 0.35 ms + 48*0.40 + 16*0.50
wall             ≈ n_park*0.35 + max(H2D, n_stream*0.35) + 0.3*attn
```

The planner parks as many early FFN as the 14 GiB soft card allows after GA+DeltaNet+KV+scratch+one stream slot. Typical default catalog: ~40 parked / ~24 streamed. The exe prints the actual plan from the GGUF **header** (tensor directory only — no 17GB weight load).

If the 5080 hour still prints `<20 tok/s`, the telemetry `pcie_B/token` and `token_ms` lines are the proof. Do not shrink the model.

## Benchmark recipe (5080 box, not this VM)

Same GGUF. Same coding-assistant prompt (the five held-out tasks stay out). Rebuild:

```text
cmake -S harness -B harness/build -DMICRO_LLM_LLAMA=ON -DMICRO_LLM_LLAMA_DIR=... -DMICRO_LLM_CUDA=ON
cmake --build harness/build -j
```

For each context, several hundred generated tokens:

```text
micro-llm-trace.exe --ui --model Qwen3.8-27B-UD-Q4_K_M.gguf --out hour.mlpt --n-predict 512 --ctx 4096
micro-llm-trace.exe --ui --model Qwen3.8-27B-UD-Q4_K_M.gguf --out hour.mlpt --n-predict 512 --ctx 8192
micro-llm-trace.exe --ui --model Qwen3.8-27B-UD-Q4_K_M.gguf --out hour.mlpt --n-predict 512 --ctx 16384
micro-llm-trace.exe --ui --model Qwen3.8-27B-UD-Q4_K_M.gguf --out hour.mlpt --n-predict 256 --ctx 32768
```

Copy the `==== micro-llm decode telemetry ====` block and the `tokens/s=` line. Compare to the before table above.

If FA AVs: add `--no-flash-attn`. If you need the old (wrong) host-DeltaNet pin to compare: `--host-deltanet` — expect ~3.5–6 tok/s and a telemetry line that 20 tok/s is impossible.

Checkpoint stays the ~18MB MLPT. Do not pack until ~20k tokens.

## What we did not invent

No custom GEMM. Reuse llama.cpp / ggml CUDA (Blackwell / sm_120 if the tree provides it). Unique code is residency, tracing, the lock-free ring, and the telemetry report.
