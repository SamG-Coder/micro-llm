import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  FFN_BINS_PER_LAYER,
  N_GROUPS,
  N_LAYERS,
  N_PACKS,
  N_SPINE,
  VOCAB_BINS,
  layerFromPackId,
  rarityGlow,
  vocabBinIndex,
} from "./constants.js";
import { makeGlyphSprite, makeStreakMesh } from "./glyphs.js";
import { packSpikeBit } from "./ring.js";

const FFN_COUNT = N_LAYERS * FFN_BINS_PER_LAYER;
const LAYER_H = 0.42;
const GROUP_GAP = 0.38;
const FFN_R = 5.1;
const PACK_R = 2.35;
const VOCAB_R = 6.2;
const MAX_FLYERS = 28;

function layerY(layer) {
  const group = Math.floor(layer / 4);
  return layer * LAYER_H + group * GROUP_GAP;
}

function hex(n) {
  return new THREE.Color(n);
}

function ffnXZ(bin) {
  const t = (bin / FFN_BINS_PER_LAYER) * Math.PI * 2;
  return { x: Math.sin(t) * FFN_R, z: Math.cos(t) * FFN_R, t };
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

function bezier3(a, b, c, t) {
  const u = 1 - t;
  return u * u * a + 2 * u * t * b + t * t * c;
}

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
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
      depthWrite: false,
    }),
  );
  scene.add(stars);
  return { geo, stars };
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
  camera.position.set(9.2, 3.4, 12.6);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, towerH * 0.42, 0);
  controls.enableDamping = true;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.14;
  controls.maxDistance = 42;
  controls.minDistance = 6;
  controls.maxPolarAngle = Math.PI * 0.78;
  controls.minPolarAngle = Math.PI * 0.18;

  scene.add(new THREE.AmbientLight(0x6a7380, 0.16));
  const key = new THREE.DirectionalLight(0x9aa8b8, 0.12);
  key.position.set(8, 18, 10);
  scene.add(key);

  const dummy = new THREE.Object3D();
  const tmpColor = new THREE.Color();
  const look = new THREE.Vector3();

  const shaft = new THREE.Mesh(
    new THREE.CylinderGeometry(0.07, 0.07, towerH + 2.2, 12),
    new THREE.MeshBasicMaterial({
      color: 0x1a2a30,
      transparent: true,
      opacity: 0.28,
    }),
  );
  shaft.position.y = towerH * 0.5;
  scene.add(shaft);

  const groupRingMat = new THREE.MeshBasicMaterial({
    color: 0x121820,
    transparent: true,
    opacity: 0.22,
    side: THREE.DoubleSide,
  });
  for (let g = 0; g < N_GROUPS; g++) {
    const y = (layerY(g * 4) + layerY(g * 4 + 3)) / 2;
    const ring = new THREE.Mesh(new THREE.TorusGeometry(FFN_R + 0.12, 0.008, 6, 64), groupRingMat);
    ring.rotation.x = Math.PI / 2;
    ring.position.y = y;
    scene.add(ring);
  }

  const packGeo = new THREE.OctahedronGeometry(0.22, 0);
  const packMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const packs = new THREE.InstancedMesh(packGeo, packMat, N_PACKS);
  packs.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  const packFlash = new Float32Array(N_PACKS);
  for (let p = 0; p < N_PACKS; p++) {
    const layer = layerFromPackId(p);
    const xz = packXZ(p);
    dummy.position.set(xz.x, layerY(layer), xz.z);
    const shown = !presentPacks || presentPacks[p];
    dummy.scale.set(shown ? 1 : 0, shown ? 1 : 0, shown ? 1 : 0);
    dummy.rotation.set(0.2, 0.3, 0.08);
    dummy.updateMatrix();
    packs.setMatrixAt(p, dummy.matrix);
    packs.setColorAt(p, hex(0x141820));
  }
  scene.add(packs);

  const spineGeo = new THREE.IcosahedronGeometry(0.28, 1);
  const spineMat = new THREE.MeshStandardMaterial({
    color: 0x0a1818,
    emissive: 0x1a3a3a,
    emissiveIntensity: 0.22,
    roughness: 0.7,
    metalness: 0.05,
  });
  const spine = new THREE.InstancedMesh(spineGeo, spineMat, N_SPINE);
  const spineSpark = new Float32Array(N_SPINE);
  const spineHeat = new Float32Array(N_SPINE);
  spineHeat.fill(0.42);
  for (let g = 0; g < N_SPINE; g++) {
    dummy.position.set(0, layerY(g * 4 + 3), 0);
    dummy.scale.set(1, 1, 1);
    dummy.updateMatrix();
    spine.setMatrixAt(g, dummy.matrix);
    spine.setColorAt(g, hex(0x1c3334));
  }
  scene.add(spine);

  const ffnGeo = new THREE.BoxGeometry(0.14, 0.24, 0.14);
  const ffnMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const ffn = new THREE.InstancedMesh(ffnGeo, ffnMat, FFN_COUNT);
  const ffnFlash = new Float32Array(FFN_COUNT);
  const ffnHeat = new Float32Array(FFN_COUNT);
  for (let layer = 0; layer < N_LAYERS; layer++) {
    const y = layerY(layer);
    for (let b = 0; b < FFN_BINS_PER_LAYER; b++) {
      const i = layer * FFN_BINS_PER_LAYER + b;
      const xz = ffnXZ(b);
      dummy.position.set(xz.x, y, xz.z);
      look.set(0, y, 0);
      dummy.lookAt(look);
      const shown = !presentBins || presentBins[i];
      dummy.scale.set(shown ? 1 : 0, shown ? 1 : 0, shown ? 1 : 0);
      dummy.updateMatrix();
      ffn.setMatrixAt(i, dummy.matrix);
      ffn.setColorAt(i, hex(0x07090d));
    }
  }
  scene.add(ffn);

  const vocabGeo = new THREE.BoxGeometry(0.1, 0.16, 0.1);
  const vocabMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const vocab = new THREE.InstancedMesh(vocabGeo, vocabMat, VOCAB_BINS);
  const vocabFlash = new Float32Array(VOCAB_BINS);
  const vocabRare = new Float32Array(VOCAB_BINS);
  for (let b = 0; b < VOCAB_BINS; b++) {
    const xz = vocabXZ(b);
    dummy.position.set(xz.x, -1.45, xz.z);
    look.set(0, -1.45, 0);
    dummy.lookAt(look);
    dummy.scale.set(1, 1, 1);
    dummy.updateMatrix();
    vocab.setMatrixAt(b, dummy.matrix);
    vocab.setColorAt(b, hex(0x0c0d10));
  }
  scene.add(vocab);

  const flyers = [];

  function spawnToken(word, id) {
    const sprite = makeGlyphSprite(word, id);
    const streak = makeStreakMesh();
    const bin = vocabBinIndex(id);
    const start = vocabXZ(bin);
    const drift = ((id % 11) - 5) * 0.12;
    sprite.position.set(start.x, -1.15, start.z);
    streak.position.copy(sprite.position);
    scene.add(sprite);
    scene.add(streak);
    flyers.push({
      sprite,
      streak,
      age: 0,
      life: 0.72 + rarityGlow(id) * 0.18,
      sx: start.x,
      sy: -1.15,
      sz: start.z,
      mx: start.x * 0.22 + drift,
      my: towerH * 0.38,
      mz: start.z * 0.22 - drift * 0.3,
      ex: drift * 0.25,
      ey: towerH + 0.8,
      ez: -drift * 0.2,
    });
    while (flyers.length > MAX_FLYERS) {
      const old = flyers.shift();
      scene.remove(old.sprite);
      scene.remove(old.streak);
      old.sprite.material.dispose();
      old.streak.geometry.dispose();
      old.streak.material.dispose();
    }
  }

  function applyToken(record, bins, extra = {}) {
    for (let i = 0; i < FFN_COUNT; i++) {
      if (bins[i] && (!presentBins || presentBins[i])) {
        ffnFlash[i] = 1;
        ffnHeat[i] = 0.28;
      }
    }
    for (let p = 0; p < N_PACKS; p++) {
      if (presentPacks && !presentPacks[p]) continue;
      if (packSpikeBit(record.packSpike, p)) {
        packFlash[p] = 1;
      }
    }
    for (let g = 0; g < N_SPINE; g++) {
      spineSpark[g] = 1;
      spineHeat[g] = Math.min(0.85, spineHeat[g] * 0.94 + 0.08);
    }
    vocabFlash.fill(0);
    vocabRare.fill(0);
    const ids = [record.sampledId, ...record.topk];
    for (const id of ids) {
      const b = vocabBinIndex(id);
      vocabFlash[b] = Math.max(vocabFlash[b], 1);
      vocabRare[b] = Math.max(vocabRare[b], rarityGlow(id));
    }
    if (extra.word !== undefined) {
      spawnToken(extra.word, record.sampledId);
    }
  }

  function paintInstances(dt) {
    for (let p = 0; p < N_PACKS; p++) {
      packFlash[p] *= 0.88;
      const pulse = packFlash[p];
      if (pulse < 0.04) {
        tmpColor.setRGB(0.08, 0.1, 0.13);
      } else {
        tmpColor.setRGB(0.32 + pulse * 0.18, 0.26 + pulse * 0.08, 0.16);
      }
      packs.setColorAt(p, tmpColor);
      const layer = layerFromPackId(p);
      const xz = packXZ(p);
      dummy.position.set(xz.x, layerY(layer), xz.z);
      if (presentPacks && !presentPacks[p]) {
        dummy.scale.set(0, 0, 0);
      } else {
        const s = 1 + pulse * 0.1;
        dummy.scale.set(s, s, s);
      }
      dummy.rotation.set(0.2, 0.3 + pulse * 0.25, 0.08);
      dummy.updateMatrix();
      packs.setMatrixAt(p, dummy.matrix);
    }
    packs.instanceColor.needsUpdate = true;
    packs.instanceMatrix.needsUpdate = true;

    let maxSpark = 0;
    for (let g = 0; g < N_SPINE; g++) {
      spineSpark[g] *= 0.9;
      if (spineSpark[g] > maxSpark) maxSpark = spineSpark[g];
      const heat = Math.max(0.38, spineHeat[g]);
      const spark = spineSpark[g];
      tmpColor.setRGB(
        0.1 + spark * 0.12 + heat * 0.04,
        0.2 + spark * 0.16 + heat * 0.08,
        0.22 + spark * 0.14 + heat * 0.08,
      );
      spine.setColorAt(g, tmpColor);
      dummy.position.set(0, layerY(g * 4 + 3), 0);
      dummy.scale.set(1 + spark * 0.06, 1 + spark * 0.06, 1 + spark * 0.06);
      dummy.updateMatrix();
      spine.setMatrixAt(g, dummy.matrix);
    }
    spine.instanceColor.needsUpdate = true;
    spine.instanceMatrix.needsUpdate = true;
    spineMat.emissiveIntensity = 0.18 + maxSpark * 0.12;

    for (let i = 0; i < FFN_COUNT; i++) {
      ffnFlash[i] *= 0.82;
      ffnHeat[i] *= 0.9;
      const f = ffnFlash[i];
      const h = ffnHeat[i];
      if (f < 0.03 && h < 0.05) {
        tmpColor.setRGB(0.025, 0.03, 0.04);
      } else {
        tmpColor.setRGB(0.12 + f * 0.22 + h * 0.08, 0.16 + f * 0.2, 0.2 + f * 0.22 + h * 0.06);
      }
      ffn.setColorAt(i, tmpColor);
    }
    ffn.instanceColor.needsUpdate = true;

    for (let b = 0; b < VOCAB_BINS; b++) {
      vocabFlash[b] *= 0.82;
      const f = vocabFlash[b] * (vocabRare[b] || 1);
      if (f < 0.02) {
        tmpColor.setRGB(0.045, 0.048, 0.055);
      } else {
        tmpColor.setRGB(0.22 + f * 0.18, 0.22 + f * 0.16, 0.18 + f * 0.1);
      }
      vocab.setColorAt(b, tmpColor);
    }
    vocab.instanceColor.needsUpdate = true;

    for (let i = flyers.length - 1; i >= 0; i--) {
      const fl = flyers[i];
      fl.age += dt;
      const t = Math.min(1, fl.age / fl.life);
      const ease = t;
      const x = bezier3(fl.sx, fl.mx, fl.ex, ease);
      const y = bezier3(fl.sy, fl.my, fl.ey, ease);
      const z = bezier3(fl.sz, fl.mz, fl.ez, ease);
      const nx = bezier3(fl.sx, fl.mx, fl.ex, Math.min(1, ease + 0.04));
      const ny = bezier3(fl.sy, fl.my, fl.ey, Math.min(1, ease + 0.04));
      const nz = bezier3(fl.sz, fl.mz, fl.ez, Math.min(1, ease + 0.04));
      fl.sprite.position.set(x, y, z);
      fl.streak.position.set(x, y, z);
      fl.streak.lookAt(nx, ny, nz);
      const fade = t < 0.08 ? t / 0.08 : 1 - (t - 0.08) / 0.92;
      const a = Math.max(0, fade) * (0.28 + fl.sprite.userData.rare * 0.22);
      fl.sprite.material.opacity = a;
      fl.streak.material.opacity = a * 0.7;
      const w = Math.max(0.42, Math.min(1.15, fl.sprite.userData.text.length * 0.13));
      fl.sprite.scale.set(w, 0.28, 1);
      if (t >= 1) {
        scene.remove(fl.sprite);
        scene.remove(fl.streak);
        fl.sprite.material.dispose();
        fl.streak.geometry.dispose();
        fl.streak.material.dispose();
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
  }

  window.addEventListener("resize", resize);
  resize();

  let raf = 0;
  let last = performance.now();
  function tick(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    paintInstances(dt);
    controls.update();
    renderer.render(scene, camera);
    raf = requestAnimationFrame(tick);
  }
  raf = requestAnimationFrame(tick);

  return {
    applyToken,
    spawnToken,
    dispose() {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      controls.dispose();
      for (const fl of flyers) {
        scene.remove(fl.sprite);
        scene.remove(fl.streak);
        fl.sprite.material.dispose();
        fl.streak.geometry.dispose();
        fl.streak.material.dispose();
      }
      flyers.length = 0;
      renderer.dispose();
      packGeo.dispose();
      spineGeo.dispose();
      ffnGeo.dispose();
      vocabGeo.dispose();
      stars.geo.dispose();
      packMat.dispose();
      spineMat.dispose();
      ffnMat.dispose();
      vocabMat.dispose();
      if (renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
    },
  };
}
