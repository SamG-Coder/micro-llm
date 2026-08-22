import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  CUDA_SCRATCH_BYTES,
  GIB,
  SAMPLE_BASE_CTX,
  SAMPLE_WEIGHT_BYTES,
  SERVE_USABLE_HEADLESS_BYTES,
  barState,
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
});
