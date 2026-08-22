// Canvas sprites for outgoing words/ids. Real sampled tokens only.

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
  const fontPx = 72;
  ctx.font = `700 ${fontPx}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
  const w = Math.ceil(ctx.measureText(text).width) + 48;
  canvas.width = Math.max(128, w);
  canvas.height = 128;
  ctx.font = `700 ${fontPx}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const glow = 0.35 + rare * 0.65;
  ctx.shadowColor = `rgba(80, 255, 230, ${0.35 + glow * 0.45})`;
  ctx.shadowBlur = 18 + glow * 22;
  ctx.fillStyle = `rgba(190, 255, 248, ${0.72 + glow * 0.28})`;
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
    blending: THREE.AdditiveBlending,
    opacity: 0,
  });
  const sprite = new THREE.Sprite(mat);
  const width = Math.max(1.15, Math.min(3.6, text.length * 0.38));
  sprite.scale.set(width, 0.85, 1);
  sprite.userData.rare = rare;
  sprite.userData.text = text;
  return sprite;
}
