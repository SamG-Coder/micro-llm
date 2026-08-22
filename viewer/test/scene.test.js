import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";
import { N_LAYERS, RING_DEPTH, SAMPLE_TOKEN_MS } from "../src/constants.js";

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
    assert.match(glyphs, /pointSpriteTexture/);
    assert.match(scene, /ffnVolumePos/);
    assert.match(scene, /pointSpriteTexture/);
  });

  it("keeps the sample ring at ~48ms and the live HTR1 64-layer path", () => {
    assert.ok(SAMPLE_TOKEN_MS >= 40 && SAMPLE_TOKEN_MS <= 60);
    assert.equal(N_LAYERS, 64);
    assert.equal(RING_DEPTH, 64);
    assert.match(scene, /HTR1|applyToken|packSpikeBit|presentBins|presentPacks/);
  });
});

describe("flash lerp, no pop", () => {
  it("defines lerp / lerpRGB helpers", () => {
    assert.match(scene, /export function lerp\(/);
    assert.match(scene, /export function lerpRGB\(/);
    assert.match(scene, /a \+ \(b - a\) \* t/);
  });

  it("does not snap ffnFlash / packFlash / vocabFlash to 1 or 0", () => {
    assert.equal(/ffnFlash\[[^\]]+\]\s*=\s*1\b/.test(scene), false);
    assert.equal(/packFlash\[[^\]]+\]\s*=\s*1\b/.test(scene), false);
    assert.equal(/vocabFlash\[[^\]]+\]\s*=\s*1\b/.test(scene), false);
    assert.equal(/vocabFlash\.fill\(\s*0\s*\)/.test(scene), false);
    assert.match(scene, /ffnFlash\[i\] = lerp\(ffnFlash\[i\], 1,/);
    assert.match(scene, /packFlash\[p\] = lerp\(packFlash\[p\], 1,/);
    assert.match(scene, /vocabFlash\[b\] = lerp\(vocabFlash\[b\], 1,/);
    assert.match(scene, /lerp\((?:ffnFlash|packFlash|vocabFlash)\[[^\]]+\], 0,/);
    assert.match(scene, /FLASH_FALL = 0\.07[5-8]/);
  });

  it("lerps streak and point colors together instead of hard-cutting to black", () => {
    assert.equal(/f < 0\.03/.test(scene), false);
    assert.equal(/pulse >= 0\.04/.test(scene), false);
    assert.match(scene, /lerpRGB\(ffn\.col/);
    assert.match(scene, /lerpRGB\(ffnStreaks\.col/);
    assert.match(scene, /lerpRGB\(packs\.col/);
    assert.match(scene, /lerpRGB\(packStreaks\.col/);
    assert.match(scene, /lerpRGB\(\s*vocab\.col/);
  });

  it("fades token flyers with a longer smoothstep rise, not a 0.08 pop", () => {
    assert.equal(/t < 0\.08/.test(scene), false);
    assert.match(scene, /FLYER_RISE = 0\.22/);
    assert.match(scene, /smoothstep\(0, FLYER_RISE, t\)/);
  });
});
