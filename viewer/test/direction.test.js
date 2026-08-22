import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { N_PACKS } from "../src/constants.js";
import { packLength, tokenDirection } from "../src/direction.js";
import { encodeRecord } from "../src/ring.js";

describe("direction paint, not a compass", () => {
  it("returns null when nothing fired", () => {
    const rec = {
      ffnFired: new Uint8Array(64 * 17408 / 8),
      packSpike: 0n,
      packRelResidual: new Float32Array(N_PACKS),
    };
    assert.equal(tokenDirection(rec, { ffnBitCount: 0, hottestLayer: null }), null);
  });

  it("heads up the tower toward the hottest layer; color is pack vs FFN", () => {
    const ffn = tokenDirection(
      { packSpike: 0n, packRelResidual: new Float32Array(N_PACKS) },
      { ffnBitCount: 12, hottestLayer: 17 },
    );
    assert.equal(ffn.kind, "ffn");
    assert.equal(ffn.heading, 17);
    assert.equal(ffn.length, 12);

    const packRel = new Float32Array(N_PACKS);
    packRel[3] = 0.4;
    const pack = tokenDirection(
      { packSpike: 1n << 3n, packRelResidual: packRel },
      { ffnBitCount: 0, hottestLayer: null },
    );
    assert.equal(pack.kind, "pack");
    assert.ok(pack.length > 0);
    assert.ok(Math.abs(packLength({ packSpike: 1n << 3n, packRelResidual: packRel }) - 0.4) < 1e-5);
  });

  it("does not mention NSEW or a compass", () => {
    const src = encodeRecord.toString();
    assert.equal(/\bNSEW\b/.test(src), false);
  });
});
