import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  FLAG_SPECIAL_OR_HIGH_LOSS,
  FFN_BITSET_BYTES,
  FIRE_EPS,
  HTR1_HEADER_BYTES,
  N_FFN,
  N_LAYERS,
  N_PACKS,
  OFF_FFN_FIRED,
  OFF_PACK_REL,
  OFF_PACK_SPIKE,
  RECORD_BYTES,
  RING_DEPTH,
  SPIKE_EPS,
  channelBitIndex,
} from "../src/constants.js";
import {
  bitTest,
  decodeRecord,
  encodeHTR1,
  encodeRecord,
  firedBinsFromBitset,
  generateSampleHTR1,
  generateSampleRecords,
  hottestLayerThisToken,
  packSpikeBit,
  parseHTR1,
} from "../src/ring.js";
import { RARE_ID, SPECIAL_TOKEN_INDEX } from "../src/vocab.js";

describe("HTR1 layout", () => {
  it("locks record size and field offsets", () => {
    assert.equal(FFN_BITSET_BYTES, 139264);
    assert.equal(OFF_FFN_FIRED, 280);
    assert.equal(OFF_PACK_REL, 139544);
    assert.equal(OFF_PACK_SPIKE, 139736);
    assert.equal(RECORD_BYTES, 139744);
    assert.equal(HTR1_HEADER_BYTES, 16);
    assert.equal(RING_DEPTH, 64);
  });

  it("round-trips a sparse record including bitset and pack spike", () => {
    const fires = [
      { layer: 0, channel: 0 },
      { layer: 7, channel: 17407 },
      { layer: 63, channel: 272 },
    ];
    const packRelResidual = new Float32Array(N_PACKS);
    packRelResidual[3] = 0.05;
    packRelResidual[47] = 0.021;
    packRelResidual[4] = 0.01;
    const buf = encodeRecord({
      tokenIndex: 9,
      sampledId: 21,
      flags: FLAG_SPECIAL_OR_HIGH_LOSS,
      topk: [21, 20, 0],
      fireEps: FIRE_EPS,
      spikeEps: SPIKE_EPS,
      fires,
      packRelResidual,
    });
    assert.equal(buf.byteLength, RECORD_BYTES);
    const rec = decodeRecord(buf);
    assert.equal(rec.tokenIndex, 9);
    assert.equal(rec.sampledId, 21);
    assert.equal(rec.flags & FLAG_SPECIAL_OR_HIGH_LOSS, 1);
    assert.equal(rec.nTopk, 3);
    assert.deepEqual(rec.topk, [21, 20, 0]);
    assert.ok(Math.abs(rec.fireEps - FIRE_EPS) < 1e-8);
    assert.ok(Math.abs(rec.spikeEps - SPIKE_EPS) < 1e-8);
    assert.equal(rec.ffnFired.length, FFN_BITSET_BYTES);
    assert.equal(bitTest(rec.ffnFired, channelBitIndex(0, 0)), true);
    assert.equal(bitTest(rec.ffnFired, channelBitIndex(7, 17407)), true);
    assert.equal(bitTest(rec.ffnFired, channelBitIndex(63, 272)), true);
    assert.equal(bitTest(rec.ffnFired, channelBitIndex(1, 0)), false);
    assert.equal(packSpikeBit(rec.packSpike, 3), true);
    assert.equal(packSpikeBit(rec.packSpike, 47), true);
    assert.equal(packSpikeBit(rec.packSpike, 4), false);
    assert.equal(packSpikeBit(rec.packSpike, 0), false);
  });

  it("parses an HTR1 file header and refuses a 20k archive", () => {
    const file = encodeHTR1([
      {
        tokenIndex: 0,
        sampledId: 10,
        flags: 0,
        topk: [10],
        fires: [{ layer: 3, channel: 8 }],
        packRelResidual: new Float32Array(N_PACKS),
      },
    ]);
    const parsed = parseHTR1(file);
    assert.equal(parsed.version, 1);
    assert.equal(parsed.nLayers, N_LAYERS);
    assert.equal(parsed.nFfn, N_FFN);
    assert.equal(parsed.records.length, 1);
    assert.equal(parsed.records[0].sampledId, 10);

    const tooMany = new Array(RING_DEPTH + 1).fill({
      tokenIndex: 0,
      sampledId: 1,
      flags: 0,
      topk: [1],
      fires: [],
      packRelResidual: new Float32Array(N_PACKS),
    });
    assert.throws(() => encodeHTR1(tooMany), /ring depth/);
  });

  it("does not flash a bin when every bit in that bin is 0", () => {
    const fires = [{ layer: 2, channel: 0 }];
    const rec = decodeRecord(
      encodeRecord({
        tokenIndex: 0,
        sampledId: 1,
        flags: 0,
        topk: [1],
        fires,
        packRelResidual: new Float32Array(N_PACKS),
      }),
    );
    const bins = firedBinsFromBitset(rec.ffnFired);
    assert.equal(bins[2 * 64 + 0], 1);
    assert.equal(bins[2 * 64 + 1], 0);
    assert.equal(bins[0], 0);
  });

  it("reports the layer with the most fired channels this token", () => {
    const fires = [
      { layer: 5, channel: 1 },
      { layer: 5, channel: 2 },
      { layer: 5, channel: 3 },
      { layer: 63, channel: 8 },
    ];
    const rec = decodeRecord(
      encodeRecord({
        tokenIndex: 0,
        sampledId: 1,
        flags: 0,
        topk: [1],
        fires,
        packRelResidual: new Float32Array(N_PACKS),
      }),
    );
    assert.equal(hottestLayerThisToken(rec.ffnFired), 5);
  });
});

describe("sample ring", () => {
  it("is deterministic, sparse, and includes rare + special tokens", () => {
    const a = generateSampleRecords(42, 48);
    const b = generateSampleRecords(42, 48);
    assert.equal(a.length, 48);
    assert.equal(a.length, b.length);
    assert.ok(a.length <= RING_DEPTH);
    for (let i = 0; i < a.length; i++) {
      assert.equal(a[i].tokenIndex, i);
      assert.equal(a[i].sampledId, b[i].sampledId);
      assert.equal(a[i].fires.length, b[i].fires.length);
      assert.ok(a[i].fires.length < N_LAYERS * 8);
    }
    assert.ok(a.some((r) => r.sampledId === RARE_ID));
    assert.equal(a[SPECIAL_TOKEN_INDEX].flags & FLAG_SPECIAL_OR_HIGH_LOSS, 1);
    assert.equal(
      a.filter((r) => r.flags & FLAG_SPECIAL_OR_HIGH_LOSS).length,
      1,
    );

    const file = generateSampleHTR1(42, 48);
    const parsed = parseHTR1(file);
    assert.equal(parsed.records.length, 48);
    const rec = parsed.records[0];
    let nFired = 0;
    for (let i = 0; i < rec.ffnFired.length * 8; i++) {
      if (bitTest(rec.ffnFired, i)) nFired++;
    }
    assert.ok(nFired > 0);
    assert.ok(nFired < N_LAYERS * N_FFN * 0.01);
    let nSpike = 0;
    for (let p = 0; p < N_PACKS; p++) {
      if (packSpikeBit(rec.packSpike, p)) nSpike++;
    }
    assert.ok(nSpike >= 3 && nSpike <= 6);
  });
});
