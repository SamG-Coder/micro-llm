// Dim streak glyphs for outgoing words/ids. Real sampled tokens only.

import * as THREE from "three";
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
  const fontPx = 36;
  ctx.font = `500 ${fontPx}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
  const w = Math.ceil(ctx.measureText(text).width) + 24;
  canvas.width = Math.max(64, w);
  canvas.height = 48;
  ctx.font = `500 ${fontPx}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const a = 0.38 + rare * 0.22;
  ctx.fillStyle = `rgba(168, 184, 204, ${a})`;
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
  const width = Math.max(0.42, Math.min(1.15, text.length * 0.13));
  sprite.scale.set(width, 0.28, 1);
  sprite.userData.rare = rare;
  sprite.userData.text = text;
  return sprite;
}

export function makeStreakMesh() {
  const geo = new THREE.BoxGeometry(0.018, 0.018, 0.72);
  const mat = new THREE.MeshBasicMaterial({
    color: 0x7a8ea3,
    transparent: true,
    opacity: 0,
    depthWrite: false,
  });
  const mesh = new THREE.Mesh(geo, mat);
  return mesh;
}
