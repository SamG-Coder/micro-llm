import assert from "node:assert/strict";
import { describe, it } from "node:test";
import raw from "../sample/sample_keep_mask.json" with { type: "json" };
import { N_FFN, N_LAYERS, ffnBinIndex } from "../src/constants.js";
import {
  SAMPLE_DROPPED_PACKS,
  SAMPLE_KEEP_CHANNEL_RANGES,
  channelKept,
  expandRanges,
  loadKeepMask,
  packKept,
  presentBins,
  presentPacks,
} from "../src/keepmask.js";
import { decodeRecord, encodeRecord, firedBinsFromBitset } from "../src/ring.js";

describe("sample keep-mask", () => {
  it("is keep vs dropped only and matches the labeled example", () => {
    assert.equal("state" in raw, false);
    assert.equal(raw.sample, true);
    assert.match(raw.note, /Not a measured remnant/);
    assert.equal(raw.n_layer, 64);
    assert.equal(raw.n_ffn, 17408);
    assert.equal(raw.keep_vision, false);
    assert.equal(raw.keep_mtp, false);
    assert.equal(raw.serve_ok, true);

    const kept = expandRanges(SAMPLE_KEEP_CHANNEL_RANGES);
    assert.equal(kept.length, 192);
    assert.deepEqual(kept.slice(0, 3), [0, 1, 2]);
    assert.deepEqual(kept.slice(-3), [17405, 17406, 17407]);
    assert.ok(kept.includes(63));
    assert.ok(kept.includes(1000));
    assert.ok(kept.includes(1063));
    assert.ok(kept.includes(17344));
    assert.equal(kept.includes(64), false);
    assert.equal(kept.includes(999), false);

    const mask = loadKeepMask(raw);
    assert.equal(mask.keepChannels.length, N_LAYERS);
    for (let layer = 0; layer < N_LAYERS; layer++) {
      assert.equal(mask.keepChannels[layer].length, 192);
      assert.equal(channelKept(mask, layer, 0), true);
      assert.equal(channelKept(mask, layer, 64), false);
      assert.equal(channelKept(mask, layer, 1000), true);
      assert.equal(channelKept(mask, layer, N_FFN - 1), true);
    }
    assert.equal(packKept(mask, 0), true);
    assert.equal(packKept(mask, 1), true);
    assert.equal(packKept(mask, 47), true);
    for (const p of SAMPLE_DROPPED_PACKS) {
      assert.equal(packKept(mask, p), false);
    }
    assert.equal(mask.keepPacks.size, 45);
    assert.equal(mask.vocabRemap.get(0), 0);
    assert.equal(mask.vocabRemap.get(255), 255);
    assert.equal(mask.vocabRemap.get(1000), 256);
    assert.equal(mask.vocabRemap.get(200000), 262);
    assert.equal(mask.vocabRemap.get(248319), 263);
  });

  it("refuses a state field", () => {
    assert.throws(
      () => loadKeepMask({ ...raw, state: "fired" }),
      /state field/,
    );
  });

  it("does not flash a dropped channel; dropped bins are not present", () => {
    const mask = loadKeepMask(raw);
    const binsPresent = presentBins(mask);
    assert.equal(binsPresent[0 * 64 + ffnBinIndex(0)], 1);
    assert.equal(binsPresent[0 * 64 + ffnBinIndex(500)], 0);
    assert.equal(binsPresent[0 * 64 + ffnBinIndex(1000)], 1);
    assert.equal(binsPresent[0 * 64 + ffnBinIndex(17344)], 1);

    const packs = presentPacks(mask);
    assert.equal(packs[0], 1);
    assert.equal(packs[10], 0);
    assert.equal(packs[22], 0);
    assert.equal(packs[33], 0);
    assert.equal(packs[47], 1);

    const droppedOnly = decodeRecord(
      encodeRecord({
        tokenIndex: 0,
        sampledId: 1,
        flags: 0,
        topk: [1],
        fires: [{ layer: 0, channel: 200 }],
        packRelResidual: new Float32Array(48),
      }),
    );
    const keptFire = decodeRecord(
      encodeRecord({
        tokenIndex: 1,
        sampledId: 1,
        flags: 0,
        topk: [1],
        fires: [{ layer: 0, channel: 10 }],
        packRelResidual: new Float32Array(48),
      }),
    );
    assert.equal(firedBinsFromBitset(droppedOnly.ffnFired, mask)[ffnBinIndex(200)], 0);
    assert.equal(firedBinsFromBitset(keptFire.ffnFired, mask)[ffnBinIndex(10)], 1);
  });
});
