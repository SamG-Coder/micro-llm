import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  FFN_BINS_PER_LAYER,
  N_LAYERS,
  N_PACKS,
  N_SPINE,
  VOCAB_BINS,
  layerFromPackId,
  rarityGlow,
  vocabBinIndex,
} from "./constants.js";
import { tokenDirection } from "./direction.js";
import { glyphScale, makeGlyphSprite, makeHeadPoint, makeStreakLine, pointSpriteTexture } from "./glyphs.js";
import { packSpikeBit } from "./ring.js";

const FFN_COUNT = N_LAYERS * FFN_BINS_PER_LAYER;
const LAYER_H = 0.42;
const GROUP_GAP = 0.38;
const PACK_R = 2.35;
const VOCAB_R = 6.2;
const MAX_FLYERS = 20;
const GOLDEN = 0.6180339887498948;

function layerY(layer) {
  const group = Math.floor(layer / 4);
  return layer * LAYER_H + group * GROUP_GAP;
}

function packXZ(pack) {
  const slot = pack % 3;
  const group = Math.floor(pack / 3);
  const t = slot * ((Math.PI * 2) / 3) + group * 0.16;
  return { x: Math.sin(t) * PACK_R, z: Math.cos(t) * PACK_R };
}

function vocabXZ(bin) {
  const t = (bin / VOCAB_BINS) * Math.PI * 2;
  return { x: Math.sin(t) * VOCAB_R, z: Math.cos(t) * VOCAB_R };
}

// Fired FFN bins live in the layer volume. Not a ring, not a cube grid.
function ffnVolumePos(layer, bin) {
  const y = layerY(layer);
  const theta = ((bin * GOLDEN) % 1) * Math.PI * 2 + layer * 0.11;
  const u = ((bin * 17 + layer * 31) * GOLDEN) % 1;
  const r = 1.25 + u * 3.4;
  const yJ = (((bin * 13 + layer * 7) * GOLDEN) % 1) * 0.22 - 0.11;
  return { x: Math.sin(theta) * r, y: y + yJ, z: Math.cos(theta) * r };
}

function ffnStreakDir(layer, bin) {
  const theta = ((bin * GOLDEN + 0.17) % 1) * Math.PI * 2 + layer * 0.07;
  return {
    x: Math.cos(theta) * 0.38,
    y: 0.08,
    z: Math.sin(theta) * 0.38,
  };
}

function bezier3(a, b, c, t) {
  const u = 1 - t;
  return u * u * a + 2 * u * t * b + t * t * c;
}

function setRGB(col, i, r, g, b) {
  col[i * 3] = r;
  col[i * 3 + 1] = g;
  col[i * 3 + 2] = b;
}

export function lerp(a, b, t) {
  return a + (b - a) * t;
}

export function lerpRGB(col, i, r, g, b, t) {
  const o = i * 3;
  col[o] = lerp(col[o], r, t);
  col[o + 1] = lerp(col[o + 1], g, t);
  col[o + 2] = lerp(col[o + 2], b, t);
}

function smoothstep(edge0, edge1, x) {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

const FLASH_RISE = 0.62;
const FLASH_FALL = 0.075;
const COLOR_LERP = 0.32;
const FLYER_RISE = 0.22;

function addStarField(scene) {
  const n = 3200;
  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const r = 48 + Math.random() * 70;
    pos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    pos[i * 3 + 2] = r * Math.cos(phi);
    const w = 0.45 + Math.random() * 0.45;
    col[i * 3] = w * 0.86;
    col[i * 3 + 1] = w * 0.9;
    col[i * 3 + 2] = w;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  const stars = new THREE.Points(
    geo,
    new THREE.PointsMaterial({
      size: 0.055,
      map: pointSpriteTexture(),
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
      depthWrite: false,
    }),
  );
  scene.add(stars);
  return { geo, stars };
}

function makePointCloud(count, size) {
  const pos = new Float32Array(count * 3);
  const col = new Float32Array(count * 3);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  const pts = new THREE.Points(
    geo,
    new THREE.PointsMaterial({
      size,
      map: pointSpriteTexture(),
      vertexColors: true,
      transparent: true,
      opacity: 1,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    }),
  );
  return { geo, pos, col, pts };
}

function makeStreakCloud(count) {
  const pos = new Float32Array(count * 2 * 3);
  const col = new Float32Array(count * 2 * 3);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  const lines = new THREE.LineSegments(
    geo,
    new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }),
  );
  return { geo, pos, col, lines };
}

export function createMap(container, { presentBins = null, presentPacks = null } = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x010208, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const stars = addStarField(scene);

  const towerH = layerY(N_LAYERS - 1);
  const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 200);
  camera.position.set(7.6, 4.0, 10.4);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, towerH * 0.42, 0);
  controls.enableDamping = true;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.14;
  controls.maxDistance = 42;
  controls.minDistance = 5.5;
  controls.maxPolarAngle = Math.PI * 0.78;
  controls.minPolarAngle = Math.PI * 0.18;

  const streakRes = new THREE.Vector2(1, 1);

  const axisPos = new Float32Array([0, -1.6, 0, 0, towerH + 1.2, 0]);
  const axisGeo = new THREE.BufferGeometry();
  axisGeo.setAttribute("position", new THREE.BufferAttribute(axisPos, 3));
  const axis = new THREE.Line(
    axisGeo,
    new THREE.LineBasicMaterial({
      color: 0x1a2a30,
      transparent: true,
      opacity: 0.34,
      depthWrite: false,
    }),
  );
  scene.add(axis);

  const packs = makePointCloud(N_PACKS, 0.14);
  const packStreaks = makeStreakCloud(N_PACKS);
  const packFlash = new Float32Array(N_PACKS);
  for (let p = 0; p < N_PACKS; p++) {
    const layer = layerFromPackId(p);
    const xz = packXZ(p);
    const y = layerY(layer);
    packs.pos[p * 3] = xz.x;
    packs.pos[p * 3 + 1] = y;
    packs.pos[p * 3 + 2] = xz.z;
    packStreaks.pos[p * 6] = xz.x;
    packStreaks.pos[p * 6 + 1] = y;
    packStreaks.pos[p * 6 + 2] = xz.z;
    packStreaks.pos[p * 6 + 3] = xz.x * 0.72;
    packStreaks.pos[p * 6 + 4] = y + 0.12;
    packStreaks.pos[p * 6 + 5] = xz.z * 0.72;
  }
  packs.geo.attributes.position.needsUpdate = true;
  packStreaks.geo.attributes.position.needsUpdate = true;
  scene.add(packs.pts);
  scene.add(packStreaks.lines);

  const spine = makePointCloud(N_SPINE, 0.16);
  const spineSpark = new Float32Array(N_SPINE);
  const spineHeat = new Float32Array(N_SPINE);
  spineHeat.fill(0.42);
  for (let g = 0; g < N_SPINE; g++) {
    spine.pos[g * 3 + 1] = layerY(g * 4 + 3);
    setRGB(spine.col, g, 0.14, 0.26, 0.28);
  }
  spine.geo.attributes.position.needsUpdate = true;
  spine.geo.attributes.color.needsUpdate = true;
  scene.add(spine.pts);

  const ffn = makePointCloud(FFN_COUNT, 0.085);
  const ffnStreaks = makeStreakCloud(FFN_COUNT);
  const ffnFlash = new Float32Array(FFN_COUNT);
  const ffnHeat = new Float32Array(FFN_COUNT);
  for (let layer = 0; layer < N_LAYERS; layer++) {
    for (let b = 0; b < FFN_BINS_PER_LAYER; b++) {
      const i = layer * FFN_BINS_PER_LAYER + b;
      const p = ffnVolumePos(layer, b);
      const d = ffnStreakDir(layer, b);
      ffn.pos[i * 3] = p.x;
      ffn.pos[i * 3 + 1] = p.y;
      ffn.pos[i * 3 + 2] = p.z;
      ffnStreaks.pos[i * 6] = p.x - d.x;
      ffnStreaks.pos[i * 6 + 1] = p.y - d.y;
      ffnStreaks.pos[i * 6 + 2] = p.z - d.z;
      ffnStreaks.pos[i * 6 + 3] = p.x + d.x;
      ffnStreaks.pos[i * 6 + 4] = p.y + d.y;
      ffnStreaks.pos[i * 6 + 5] = p.z + d.z;
    }
  }
  ffn.geo.attributes.position.needsUpdate = true;
  ffnStreaks.geo.attributes.position.needsUpdate = true;
  scene.add(ffn.pts);
  scene.add(ffnStreaks.lines);

  const vocab = makePointCloud(VOCAB_BINS, 0.1);
  const vocabFlash = new Float32Array(VOCAB_BINS);
  const vocabRare = new Float32Array(VOCAB_BINS);
  for (let b = 0; b < VOCAB_BINS; b++) {
    const xz = vocabXZ(b);
    vocab.pos[b * 3] = xz.x;
    vocab.pos[b * 3 + 1] = -1.45;
    vocab.pos[b * 3 + 2] = xz.z;
  }
  vocab.geo.attributes.position.needsUpdate = true;
  scene.add(vocab.pts);

  const dirLine = makeStreakLine(streakRes);
  dirLine.material.opacity = 0;
  scene.add(dirLine);
  let dirFlash = 0;
  let dirKind = "ffn";

  const flyers = [];

  function disposeFlyer(fl) {
    scene.remove(fl.sprite);
    scene.remove(fl.streak);
    scene.remove(fl.head);
    fl.sprite.material.dispose();
    fl.streak.geometry.dispose();
    fl.streak.material.dispose();
    fl.head.geometry.dispose();
    fl.head.material.dispose();
  }

  function spawnToken(word, id) {
    const sprite = makeGlyphSprite(word, id);
    const streak = makeStreakLine(streakRes);
    const head = makeHeadPoint();
    const bin = vocabBinIndex(id);
    const start = vocabXZ(bin);
    const drift = ((id % 11) - 5) * 0.18;
    sprite.position.set(start.x, -1.15, start.z);
    head.position.copy(sprite.position);
    scene.add(sprite);
    scene.add(streak);
    scene.add(head);
    flyers.push({
      sprite,
      streak,
      head,
      age: 0,
      life: 1.12 + rarityGlow(id) * 0.28,
      sx: start.x,
      sy: -1.15,
      sz: start.z,
      mx: start.x * 0.48 + drift,
      my: towerH * (0.3 + (id % 9) * 0.035),
      mz: start.z * 0.48 - drift * 0.35,
      ex: drift * 0.3,
      ey: towerH + 0.9,
      ez: -drift * 0.22,
    });
    while (flyers.length > MAX_FLYERS) {
      disposeFlyer(flyers.shift());
    }
  }

  function applyToken(record, bins, extra = {}) {
    for (let i = 0; i < FFN_COUNT; i++) {
      if (bins[i] && (!presentBins || presentBins[i])) {
        ffnFlash[i] = lerp(ffnFlash[i], 1, FLASH_RISE);
        ffnHeat[i] = 0.2;
      }
    }
    for (let p = 0; p < N_PACKS; p++) {
      if (presentPacks && !presentPacks[p]) continue;
      if (packSpikeBit(record.packSpike, p)) {
        packFlash[p] = lerp(packFlash[p], 1, FLASH_RISE);
      }
    }
    for (let g = 0; g < N_SPINE; g++) {
      spineSpark[g] = 1;
      spineHeat[g] = Math.min(0.85, spineHeat[g] * 0.94 + 0.08);
    }
    const ids = [record.sampledId, ...record.topk];
    for (const id of ids) {
      const b = vocabBinIndex(id);
      vocabFlash[b] = lerp(vocabFlash[b], 1, FLASH_RISE);
      vocabRare[b] = Math.max(vocabRare[b], rarityGlow(id));
    }
    if (extra.word !== undefined) {
      spawnToken(extra.word, record.sampledId);
    }
    const dir = tokenDirection(record, {
      hottestLayer: extra.hottestLayer,
      ffnBitCount: extra.ffnBitCount,
    });
    if (dir && dir.length > 0) {
      dirFlash = lerp(dirFlash, 1, FLASH_RISE);
      dirKind = dir.kind;
      const y1 = layerY(dir.heading);
      const lenScale = Math.min(2.8, 0.35 + Math.log1p(dir.length) * 0.28);
      dirLine.geometry.setPositions([0, Math.max(-0.4, y1 - lenScale), 0, 0, y1, 0]);
      if (dirKind === "pack") {
        dirLine.material.color.setRGB(0.78, 0.44, 0.2);
      } else {
        dirLine.material.color.setRGB(0.28, 0.64, 0.8);
      }
    }
  }

  function paintClouds(dt) {
    for (let p = 0; p < N_PACKS; p++) {
      packFlash[p] = lerp(packFlash[p], 0, FLASH_FALL);
      const pulse = packFlash[p];
      const present = !(presentPacks && !presentPacks[p]);
      const r = present ? pulse * (0.42 + pulse * 0.28) : 0;
      const g = present ? pulse * (0.34 + pulse * 0.12) : 0;
      const b = present ? pulse * 0.2 : 0;
      lerpRGB(packs.col, p, r, g, b, COLOR_LERP);
      lerpRGB(packStreaks.col, p * 2, r, g, b, COLOR_LERP);
      lerpRGB(packStreaks.col, p * 2 + 1, r * 0.25, g * 0.25, b * 0.25, COLOR_LERP);
    }
    packs.geo.attributes.color.needsUpdate = true;
    packStreaks.geo.attributes.color.needsUpdate = true;

    for (let g = 0; g < N_SPINE; g++) {
      spineSpark[g] *= 0.9;
      const heat = Math.max(0.38, spineHeat[g]);
      const spark = spineSpark[g];
      setRGB(
        spine.col,
        g,
        0.12 + spark * 0.22 + heat * 0.08,
        0.24 + spark * 0.28 + heat * 0.14,
        0.26 + spark * 0.24 + heat * 0.14,
      );
    }
    spine.geo.attributes.color.needsUpdate = true;

    for (let i = 0; i < FFN_COUNT; i++) {
      ffnFlash[i] = lerp(ffnFlash[i], 0, FLASH_FALL);
      ffnHeat[i] *= 0.84;
      const f = ffnFlash[i];
      const h = ffnHeat[i];
      const glow = Math.min(1, f + h);
      const r = glow * (0.16 + f * 0.38 + h * 0.08);
      const g = glow * (0.22 + f * 0.32 + h * 0.06);
      const b = glow * (0.3 + f * 0.42 + h * 0.08);
      lerpRGB(ffn.col, i, r, g, b, COLOR_LERP);
      lerpRGB(ffnStreaks.col, i * 2, r, g, b, COLOR_LERP);
      lerpRGB(ffnStreaks.col, i * 2 + 1, r * 0.2, g * 0.2, b * 0.2, COLOR_LERP);
    }
    ffn.geo.attributes.color.needsUpdate = true;
    ffnStreaks.geo.attributes.color.needsUpdate = true;

    for (let b = 0; b < VOCAB_BINS; b++) {
      vocabFlash[b] = lerp(vocabFlash[b], 0, FLASH_FALL);
      const f = vocabFlash[b] * (vocabRare[b] || 1);
      lerpRGB(
        vocab.col,
        b,
        f * (0.28 + f * 0.22),
        f * (0.28 + f * 0.18),
        f * (0.22 + f * 0.12),
        COLOR_LERP,
      );
    }
    vocab.geo.attributes.color.needsUpdate = true;

    dirFlash = lerp(dirFlash, 0, FLASH_FALL);
    dirLine.material.opacity = dirFlash * 0.88;
    dirLine.material.resolution.copy(streakRes);

    for (let i = flyers.length - 1; i >= 0; i--) {
      const fl = flyers[i];
      fl.age += dt;
      const t = Math.min(1, fl.age / fl.life);
      const x = bezier3(fl.sx, fl.mx, fl.ex, t);
      const y = bezier3(fl.sy, fl.my, fl.ey, t);
      const z = bezier3(fl.sz, fl.mz, fl.ez, t);
      const back = Math.max(0, t - 0.09);
      const bx = bezier3(fl.sx, fl.mx, fl.ex, back);
      const by = bezier3(fl.sy, fl.my, fl.ey, back);
      const bz = bezier3(fl.sz, fl.mz, fl.ez, back);
      fl.sprite.position.set(x, y, z);
      fl.head.position.set(x, y, z);
      fl.streak.geometry.setPositions([bx, by, bz, x, y, z]);
      const fade =
        t < FLYER_RISE ? smoothstep(0, FLYER_RISE, t) : 1 - smoothstep(FLYER_RISE, 1, t);
      const a = Math.max(0, fade) * (0.62 + fl.sprite.userData.rare * 0.26);
      fl.sprite.material.opacity = a;
      fl.streak.material.opacity = Math.min(1, a * 1.15);
      fl.head.material.opacity = Math.min(1, a * 1.05);
      const s = glyphScale(fl.sprite.userData.text);
      fl.sprite.scale.set(s.x, s.y, 1);
      if (t >= 1) {
        disposeFlyer(fl);
        flyers.splice(i, 1);
      }
    }
  }

  function resize() {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / Math.max(1, h);
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
    streakRes.set(Math.max(1, w), Math.max(1, h));
    for (const fl of flyers) {
      fl.streak.material.resolution.copy(streakRes);
    }
  }

  window.addEventListener("resize", resize);
  resize();

  let raf = 0;
  let last = performance.now();
  function tick(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    paintClouds(dt);
    controls.update();
    renderer.render(scene, camera);
    raf = requestAnimationFrame(tick);
  }
  raf = requestAnimationFrame(tick);

  return {
    applyToken,
    tokenDirection,
    spawnToken,
    dispose() {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      controls.dispose();
      for (const fl of flyers) disposeFlyer(fl);
      flyers.length = 0;
      renderer.dispose();
      stars.geo.dispose();
      axisGeo.dispose();
      packs.geo.dispose();
      packStreaks.geo.dispose();
      spine.geo.dispose();
      ffn.geo.dispose();
      ffnStreaks.geo.dispose();
      vocab.geo.dispose();
      packs.pts.material.dispose();
      packStreaks.lines.material.dispose();
      spine.pts.material.dispose();
      ffn.pts.material.dispose();
      ffnStreaks.lines.material.dispose();
      vocab.pts.material.dispose();
      axis.material.dispose();
      dirLine.geometry.dispose();
      dirLine.material.dispose();
      if (renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
    },
  };
}
