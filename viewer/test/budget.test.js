import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  CUDA_SCRATCH_BYTES,
  GIB,
  PINNED_GA_WEIGHT_BYTES,
  PINNED_GA_KV_BYTES,
  Q4_FFN_WORKSPACE_BYTES,
  SAMPLE_BASE_CTX,
  SAMPLE_WEIGHT_BYTES,
  SERVE_USABLE_HEADLESS_BYTES,
  barState,
  liveCardStackAtToken,
  sampleBudgetAtToken,
  serveStackBytes,
} from "../src/budget.js";

describe("15.2 serve bar", () => {
  it("is weights + 0.9 CUDA + KV, green on the sample remnant", () => {
    const stack = serveStackBytes({
      weightBytes: SAMPLE_WEIGHT_BYTES,
      ctx: SAMPLE_BASE_CTX,
    });
    assert.equal(stack, SAMPLE_WEIGHT_BYTES + CUDA_SCRATCH_BYTES + SAMPLE_BASE_CTX * 65536);
    assert.ok(stack < SERVE_USABLE_HEADLESS_BYTES);
    const s = barState({ serveOk: true, vision: false });
    assert.equal(s.color, "green");
  });

  it("turns red if serve_ok is false or vision is on", () => {
    assert.equal(barState({ serveOk: false }).color, "red");
    assert.equal(barState({ vision: true }).color, "red");
  });

  it("turns yellow inside 0.5GB of the cap", () => {
    const near = SERVE_USABLE_HEADLESS_BYTES - Math.trunc(0.2 * GIB) - CUDA_SCRATCH_BYTES - SAMPLE_BASE_CTX * 65536;
    assert.equal(barState({ weightBytes: near, serveOk: true, vision: false }).color, "yellow");
  });

  it("barely moves across the sample ring", () => {
    const a = sampleBudgetAtToken(0);
    const b = sampleBudgetAtToken(47);
    assert.equal(a.color, "green");
    assert.equal(b.color, "green");
    assert.equal(b.kvBytes - a.kvBytes, 47 * 65536);
    assert.ok((b.stack - a.stack) / a.usable < 0.003);
    assert.ok(a.stack / GIB < 12.3);
    assert.ok(a.stack / GIB > 12.1);
  });

  it("live card stack is GA pin + 0.9 + one FFN workspace + KV, not 10.8", () => {
    const live = liveCardStackAtToken(0);
    const sample = sampleBudgetAtToken(0);
    assert.equal(live.weightBytes, PINNED_GA_WEIGHT_BYTES);
    assert.ok(live.ffnWorkspaceBytes > 0);
    assert.ok(live.ffnWorkspaceBytes < 0.2 * GIB);
    assert.ok(live.stack < sample.stack);
    assert.ok(Math.abs(live.weightBytes / GIB - 0.6) < 0.01);
    const parked = PINNED_GA_KV_BYTES + CUDA_SCRATCH_BYTES + 64 * Q4_FFN_WORKSPACE_BYTES;
    assert.ok(parked > SERVE_USABLE_HEADLESS_BYTES);
    const streamed = PINNED_GA_KV_BYTES + CUDA_SCRATCH_BYTES + Q4_FFN_WORKSPACE_BYTES;
    assert.ok(streamed < SERVE_USABLE_HEADLESS_BYTES);
  });
});
