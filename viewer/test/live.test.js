import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { PINNED_GA_WEIGHT_BYTES, SAMPLE_WEIGHT_BYTES, liveCardStackAtToken } from "../src/budget.js";
import { FIRE_EPS, N_PACKS, RECORD_BYTES, SPIKE_EPS } from "../src/constants.js";
import { decodeHostPayload, installLivePush, stopSampleReplay } from "../src/live.js";
import { encodeRecord } from "../src/ring.js";

describe("live attach", () => {
  it("kills the sample timer when a live record arrives", () => {
    const g = { __sampleReplay: 1 };
    const ids = [];
    g.setInterval = () => 99;
    g.clearInterval = (id) => ids.push(id);
    g.__sampleReplay = 7;
    stopSampleReplay(g);
    assert.equal(g.__sampleReplay, null);
    assert.deepEqual(ids, [7]);
  });

  it("parses a raw HTR1 record and a live-attach host message", () => {
    const buf = encodeRecord({
      tokenIndex: 3,
      sampledId: 21,
      flags: 0,
      topk: [21],
      fireEps: FIRE_EPS,
      spikeEps: SPIKE_EPS,
      fires: [{ layer: 3, channel: 8 }],
      packRelResidual: new Float32Array(N_PACKS),
    });
    assert.equal(buf.byteLength, RECORD_BYTES);
    const rec = decodeHostPayload(buf);
    assert.equal(rec.kind, "record");
    assert.equal(rec.record.tokenIndex, 3);
    assert.equal(rec.record.sampledId, 21);

    const attach = decodeHostPayload({
      type: "live-attach",
      weightBytes: PINNED_GA_WEIGHT_BYTES,
      ctx: 8192,
    });
    assert.equal(attach.kind, "attach");
    assert.equal(attach.attach.weightBytes, PINNED_GA_WEIGHT_BYTES);

    const stats = decodeHostPayload({ type: "live-stats", tokensPerSec: 4.25, nTokens: 32 });
    assert.equal(stats.kind, "stats");
    assert.equal(stats.tokensPerSec, 4.25);
  });

  it("installs chrome.webview message and __htr1Push", () => {
    const seen = [];
    const listeners = {};
    const g = {
      chrome: {
        webview: {
          addEventListener(name, fn) {
            listeners[name] = fn;
          },
        },
      },
    };
    installLivePush((parsed) => seen.push(parsed.kind), g);
    assert.equal(typeof g.__htr1Push, "function");
    assert.equal(typeof listeners.message, "function");
    assert.equal(typeof listeners.sharedbufferreceived, "function");
    g.__htr1Push({ type: "live-attach" });
    assert.equal(seen[0], "attach");
  });
});

describe("live card stack", () => {
  it("uses pinned GA weights, not the 10.8 sample remnant", () => {
    const b = liveCardStackAtToken(0);
    assert.equal(b.weightBytes, PINNED_GA_WEIGHT_BYTES);
    assert.ok(b.weightBytes < SAMPLE_WEIGHT_BYTES);
    assert.ok(b.weightBytes / (1024 ** 3) < 1);
    assert.equal(b.color, "green");
  });
});
