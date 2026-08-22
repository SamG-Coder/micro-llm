// Dim streak glyphs for outgoing words/ids. Real sampled tokens only.
// Streaks are thin lines, not cubes. Glyphs stay readable in the volume.

import * as THREE from "three";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { rarityGlow } from "./constants.js";

const cache = new Map();

export function glyphLabel(word) {
  if (word === "\n") return "\\n";
  if (word === " ") return "␣";
  if (word === "") return "·";
  return word;
}

function paintGlyphCanvas(text, rare) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  const fontPx = 48;
  ctx.font = `500 ${fontPx}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
  const w = Math.ceil(ctx.measureText(text).width) + 28;
  canvas.width = Math.max(80, w);
  canvas.height = 64;
  ctx.font = `500 ${fontPx}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const a = 0.7 + rare * 0.18;
  ctx.fillStyle = `rgba(206, 216, 228, ${a})`;
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  return canvas;
}

export function glyphTexture(text, rare) {
  const key = `${text}|${rare.toFixed(2)}`;
  let tex = cache.get(key);
  if (!tex) {
    tex = new THREE.CanvasTexture(paintGlyphCanvas(text, rare));
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;
    cache.set(key, tex);
  }
  return tex;
}

export function glyphScale(text) {
  const width = Math.max(1.15, Math.min(2.45, text.length * 0.24));
  return { x: width, y: 0.52 };
}

export function makeGlyphSprite(word, id) {
  const text = glyphLabel(word);
  const rare = rarityGlow(id);
  const mat = new THREE.SpriteMaterial({
    map: glyphTexture(text, rare),
    transparent: true,
    depthWrite: false,
    blending: THREE.NormalBlending,
    opacity: 0,
  });
  const sprite = new THREE.Sprite(mat);
  const s = glyphScale(text);
  sprite.scale.set(s.x, s.y, 1);
  sprite.userData.rare = rare;
  sprite.userData.text = text;
  return sprite;
}

export function makeStreakLine(resolution) {
  const geo = new LineGeometry();
  geo.setPositions([0, 0, 0, 0, 0, 0.9]);
  const mat = new LineMaterial({
    color: 0xc8d4e2,
    transparent: true,
    opacity: 0,
    linewidth: 1.7,
    depthWrite: false,
    dashed: false,
    worldUnits: false,
  });
  if (resolution) mat.resolution.copy(resolution);
  const line = new Line2(geo, mat);
  line.computeLineDistances();
  return line;
}

export function makeHeadPoint() {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3), 3));
  const mat = new THREE.PointsMaterial({
    color: 0xd6e0ec,
    size: 0.15,
    transparent: true,
    opacity: 0,
    depthWrite: false,
    sizeAttenuation: true,
  });
  return new THREE.Points(geo, mat);
}
