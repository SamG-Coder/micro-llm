# Quality eval — local coding assistant

This is the first job. The company proof is that a 5080 remnant still writes real code, not that VRAM math works.

## Remnant under test

Do not eval FFN-only 25%. That file does not fit. The remnant under test is Memory's 5080 recipe:

- FFN width-cut at most 25% (keep >= 13056 of 17408 channels per layer). A layer with total `n_fired == 0` and no floor bits is a missing hook: keep all 17408.
- Unused vocab stripped (prompt + sampled + top-k + reserved core). Tokenizer IDs stay original.
- Vision tower dropped (text-only job).
- Up to 8 dead DeltaNet packs (`n_spike == 0`; relative residual `||out-in||/||in||` below `spike_eps` 0.02 all hour). Never drop packs on layers 0, 1, 62, 63.
- Never drop the 16 Gated Attention blocks (QKVO + 4 KV heads).
- Serve is packed Q4. `micro_llm.serve_ok` must be true. `--q4-k-to-f16` is a host dump and is not this eval.
- Must pass Memory's 15.2GB headless serve gate (file bytes + 0.9 CUDA + KV(ctx)). 12GB weights + 8k KV should pass. 14.5GB weights fail.

If the keep-mask is missing unused-vocab strip, vision drop, or ~8 dead packs, fail the eval as not-a-5080-file. Do not score quality on a file that cannot serve.

## Hour (in-distribution)

Job prompt: local coding assistant for Python and C++.

Stream the full model for about one hour (20k+ tokens). Mix of write, fix, explain, complete. No images. Do not include the held-out prompts below in this hour.

## Held-out (must not appear in the hour)

Use the five prompts in [eval/coding_heldout.md](eval/coding_heldout.md). Same prompts against:

- full Qwen 27B Q4 (baseline)
- the packed remnant (Q4, `serve_ok` true)

## Pass / fail

Hard fail (quality does not hold):

- collapse: empty, repetitive garbage, wrong language, or refuses to write code
- code-writing items are not parseable (Python `ast.parse` / `compile`; C++ is "looks like a real compilation unit", not a full compiler)
- serve OOM or `serve_ok` false

Soft fail (usable but not the company proof):

- parseable code that is noticeably worse than the full model on the same prompt (missing the ask, broken logic)

Pass:

- remnant serves under 15.2GB
- all five held-out items produce real code or a real technical answer
- no collapse vs the full model

Judge against the full model, not against perfect. An hour can overfit the session. Held-out is how we catch that.

## What this file is not

Not a benchmark suite. Not HumanEval. Not a 5080 driver. C++ owns the live forward. Export owns Q4_K requant. Memory owns the 15.2GB gate. This file owns the job, the held-out set, and the pass rule.
