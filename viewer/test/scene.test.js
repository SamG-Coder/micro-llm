import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";
import { SAMPLE_TOKEN_MS } from "../src/constants.js";

const src = join(dirname(fileURLToPath(import.meta.url)), "..", "src");
const scene = readFileSync(join(src, "scene.js"), "utf8");
const glyphs = readFileSync(join(src, "glyphs.js"), "utf8");

describe("space look: points and streaks, not a bin grid", () => {
  it("does not instance cubes for FFN bins, vocab bins, or spine", () => {
    assert.equal(scene.includes("BoxGeometry"), false);
    assert.equal(scene.includes("InstancedMesh"), false);
    assert.equal(glyphs.includes("BoxGeometry"), false);
  });

  it("paints fired bits, packs, and spine as points / thin lines", () => {
    assert.match(scene, /THREE\.Points/);
    assert.match(scene, /THREE\.LineSegments/);
    assert.match(glyphs, /Line2/);
    assert.match(glyphs, /LineGeometry/);
    assert.match(scene, /ffnVolumePos/);
  });

  it("keeps the sample ring at ~48ms", () => {
    assert.ok(SAMPLE_TOKEN_MS >= 40 && SAMPLE_TOKEN_MS <= 60);
    assert.match(scene, /HTR1|applyToken|packSpikeBit|presentBins|presentPacks/);
  });
});
