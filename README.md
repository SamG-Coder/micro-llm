# micro-llm

Smart pruning + harness so a large model (starting with Qwen 27B) actually runs on a 16GB card.

Give it a job prompt. Trace useful vs unused paths for about an hour. Cut what did not fire. Serve the packed remnant.

MIT licensed. Open source.

## Why

An RTX 5080 is 16GB GDDR7. Plan on **15.2GB usable** headless, **14.5GB** if a display is attached. CUDA plus decode scratch eats another 0.8-1.0GB before any weight loads.

Full Qwen 27B at Q4_K_M is about **17.1GB**. It does not fit. The remnant has to land at **12GB or under** for 8-16k context, **10GB or under** if we want 32k plus vision plus prefill headroom.

## How it works

1. **Prompt is the job.** You say what you want.
2. **The hour is the evidence.** We stream the full model on the host, score what fired, and do not shrink mid-session.
3. **The cut is after.** Dead first, then weak, down to the remnant ceiling.
4. **Serve is a packed GGUF.** Only surviving tensors, densely laid out. No offload on the serve path.

Two loaders, one file format. Details in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Layout

| Path | What |
| --- | --- |
| [harness/](harness/) | C++ trace streamer. Scores the hour. Writes one MLPT. Does not cut. |
| [export/](export/) | Packer. Reads MLPT + full GGUF, cuts, writes one packed remnant GGUF. |
| [docs/PRUNE_TABLE.md](docs/PRUNE_TABLE.md) | MLPT format |

## Status

Harness and export are on main. Streamer dumps scores. Packer cuts and writes the remnant.

First-job quality eval (local coding assistant): [docs/QUALITY.md](docs/QUALITY.md).

Live per-token hook-ring contract (C++ writer, Hotspot painter): [docs/HOOK_RING.md](docs/HOOK_RING.md).

## Build

cmake -S harness -B harness/build && cmake --build harness/build -j && ctest --test-dir harness/build --output-on-failure

cd export && pip install -r requirements.txt && pytest -q

## License

[MIT](LICENSE)
