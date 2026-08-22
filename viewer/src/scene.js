import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
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
import { makeGlyphSprite } from "./glyphs.js";
import { packSpikeBit } from "./ring.js";

const FFN_COUNT = N_LAYERS * FFN_BINS_PER_LAYER;
const LAYER_H = 0.42;
const GROUP_GAP = 0.38;
const FFN_R = 5.1;
const PACK_R = 2.35;
const VOCAB_R = 6.2;
const MAX_FLYERS = 22;

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

export function createMap(container, { presentBins = null, presentPacks = null } = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x04050a, 1);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.12;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x04050a, 0.028);

  const towerH = layerY(N_LAYERS - 1);
  const camera = new THREE.PerspectiveCamera(52, 1, 0.1, 120);
  camera.position.set(8.6, 1.8, 11.4);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, towerH * 0.42, 0);
  controls.enableDamping = true;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.42;
  controls.maxDistance = 36;
  controls.minDistance = 5;
  controls.maxPolarAngle = Math.PI * 0.78;
  controls.minPolarAngle = Math.PI * 0.18;

  scene.add(new THREE.AmbientLight(0x4a5366, 0.28));
  const key = new THREE.DirectionalLight(0xc8d6ea, 0.35);
  key.position.set(10, 22, 8);
  scene.add(key);

  const well = new THREE.PointLight(0x2ee6d6, 2.4, 28);
  well.position.set(0, towerH * 0.5, 0);
  scene.add(well);

  const dummy = new THREE.Object3D();
  const tmpColor = new THREE.Color();
  const look = new THREE.Vector3();

  const floor = new THREE.Mesh(
    new THREE.CylinderGeometry(7.4, 7.4, 0.06, 64),
    new THREE.MeshStandardMaterial({ color: 0x0a0d14, roughness: 0.92, metalness: 0.05 }),
  );
  floor.position.y = -1.85;
  scene.add(floor);

  const shaft = new THREE.Mesh(
    new THREE.CylinderGeometry(0.16, 0.16, towerH + 2.4, 20),
    new THREE.MeshBasicMaterial({
      color: 0x145e58,
      transparent: true,
      opacity: 0.55,
    }),
  );
  shaft.position.y = towerH * 0.5;
  scene.add(shaft);

  const shaftCore = new THREE.Mesh(
    new THREE.CylinderGeometry(0.06, 0.06, towerH + 2.6, 12),
    new THREE.MeshBasicMaterial({
      color: 0x7ff6ec,
      transparent: true,
      opacity: 0.7,
    }),
  );
  shaftCore.position.y = towerH * 0.5;
  scene.add(shaftCore);

  const groupRingMat = new THREE.MeshBasicMaterial({
    color: 0x1a3040,
    transparent: true,
    opacity: 0.35,
    side: THREE.DoubleSide,
  });
  for (let g = 0; g < N_GROUPS; g++) {
    const y = (layerY(g * 4) + layerY(g * 4 + 3)) / 2;
    const ring = new THREE.Mesh(new THREE.TorusGeometry(FFN_R + 0.15, 0.012, 8, 72), groupRingMat);
    ring.rotation.x = Math.PI / 2;
    ring.position.y = y;
    scene.add(ring);
  }

  const packGeo = new THREE.OctahedronGeometry(0.3, 0);
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
    dummy.rotation.set(0.25, 0.4, 0.1);
    dummy.updateMatrix();
    packs.setMatrixAt(p, dummy.matrix);
    packs.setColorAt(p, hex(0x1a2433));
  }
  scene.add(packs);

  const spineGeo = new THREE.IcosahedronGeometry(0.42, 1);
  const spineMat = new THREE.MeshStandardMaterial({
    color: 0x0d3d39,
    emissive: 0x2ee6d6,
    emissiveIntensity: 1.35,
    roughness: 0.18,
    metalness: 0.2,
  });
  const spine = new THREE.InstancedMesh(spineGeo, spineMat, N_SPINE);
  const spineSpark = new Float32Array(N_SPINE);
  const spineHeat = new Float32Array(N_SPINE);
  spineHeat.fill(0.45);
  const spineLights = [];
  for (let g = 0; g < N_SPINE; g++) {
    const y = layerY(g * 4 + 3);
    dummy.position.set(0, y, 0);
    dummy.scale.set(1, 1, 1);
    dummy.updateMatrix();
    spine.setMatrixAt(g, dummy.matrix);
    spine.setColorAt(g, hex(0x2ee6d6));
    const lamp = new THREE.PointLight(0x2ee6d6, 0.7, 8);
    lamp.position.set(0, y, 0);
    scene.add(lamp);
    spineLights.push(lamp);
  }
  scene.add(spine);

  const ffnGeo = new THREE.BoxGeometry(0.16, 0.28, 0.16);
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
      ffn.setColorAt(i, hex(0x0b0e14));
    }
  }
  scene.add(ffn);

  const vocabGeo = new THREE.BoxGeometry(0.12, 0.22, 0.12);
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
    vocab.setColorAt(b, hex(0x14110c));
  }
  scene.add(vocab);

  const moteGeo = new THREE.BufferGeometry();
  const moteN = 420;
  const motePos = new Float32Array(moteN * 3);
  for (let i = 0; i < moteN; i++) {
    const t = Math.random() * Math.PI * 2;
    const r = 1.2 + Math.random() * 7.2;
    motePos[i * 3] = Math.sin(t) * r;
    motePos[i * 3 + 1] = -1.6 + Math.random() * (towerH + 3);
    motePos[i * 3 + 2] = Math.cos(t) * r;
  }
  moteGeo.setAttribute("position", new THREE.BufferAttribute(motePos, 3));
  const motes = new THREE.Points(
    moteGeo,
    new THREE.PointsMaterial({
      color: 0x3a4a5c,
      size: 0.035,
      transparent: true,
      opacity: 0.35,
      depthWrite: false,
    }),
  );
  scene.add(motes);

  const flyers = [];

  function spawnToken(word, id) {
    const sprite = makeGlyphSprite(word, id);
    const bin = vocabBinIndex(id);
    const start = vocabXZ(bin);
    const drift = ((id % 11) - 5) * 0.18;
    const midY = 1.2 + (id % 17) * (towerH / 28);
    const endY = towerH * 0.72 + (id % 9) * 0.35;
    sprite.position.set(start.x, -1.15, start.z);
    scene.add(sprite);
    flyers.push({
      sprite,
      age: 0,
      life: 2.6 + rarityGlow(id) * 0.6,
      sx: start.x,
      sy: -1.15,
      sz: start.z,
      mx: start.x * 0.28 + drift,
      my: midY,
      mz: start.z * 0.28 - drift * 0.4,
      ex: drift * 0.4,
      ey: endY,
      ez: -drift * 0.3,
    });
    while (flyers.length > MAX_FLYERS) {
      const old = flyers.shift();
      scene.remove(old.sprite);
      old.sprite.material.dispose();
    }
  }

  function applyToken(record, bins, extra = {}) {
    for (let i = 0; i < FFN_COUNT; i++) {
      if (bins[i] && (!presentBins || presentBins[i])) {
        ffnFlash[i] = 1;
        ffnHeat[i] = 0.4;
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
      spineHeat[g] = Math.min(0.96, spineHeat[g] * 0.9 + 0.22);
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
      packFlash[p] *= 0.84;
      const pulse = packFlash[p];
      if (pulse < 0.04) {
        tmpColor.setRGB(0.12, 0.16, 0.22);
      } else {
        tmpColor.setRGB(1.0, 0.55 + pulse * 0.3, 0.08);
      }
      packs.setColorAt(p, tmpColor);
      const layer = layerFromPackId(p);
      const xz = packXZ(p);
      dummy.position.set(xz.x, layerY(layer), xz.z);
      if (presentPacks && !presentPacks[p]) {
        dummy.scale.set(0, 0, 0);
      } else {
        const s = 1 + pulse * 1.15;
        dummy.scale.set(s, s, s);
      }
      dummy.rotation.set(0.2, 0.4 + pulse * 1.6, 0.15);
      dummy.updateMatrix();
      packs.setMatrixAt(p, dummy.matrix);
    }
    packs.instanceColor.needsUpdate = true;
    packs.instanceMatrix.needsUpdate = true;

    let maxSpark = 0;
    for (let g = 0; g < N_SPINE; g++) {
      spineSpark[g] *= 0.86;
      if (spineSpark[g] > maxSpark) maxSpark = spineSpark[g];
      const heat = Math.max(0.4, spineHeat[g]);
      const spark = spineSpark[g];
      tmpColor.setRGB(
        0.2 + spark * 0.85,
        0.75 + spark * 0.25 + heat * 0.12,
        0.72 + spark * 0.28,
      );
      spine.setColorAt(g, tmpColor);
      dummy.position.set(0, layerY(g * 4 + 3), 0);
      const s = 1.1 + spark * 0.95 + heat * 0.12;
      dummy.scale.set(s, s, s);
      dummy.updateMatrix();
      spine.setMatrixAt(g, dummy.matrix);
      spineLights[g].intensity = 0.55 + spark * 3.4 + heat * 0.35;
    }
    spine.instanceColor.needsUpdate = true;
    spine.instanceMatrix.needsUpdate = true;
    spineMat.emissiveIntensity = 1.05 + maxSpark * 1.8;
    well.intensity = 1.6 + maxSpark * 3.2;
    shaftCore.material.opacity = 0.45 + maxSpark * 0.5;
    shaft.material.opacity = 0.4 + maxSpark * 0.25;

    for (let i = 0; i < FFN_COUNT; i++) {
      ffnFlash[i] *= 0.78;
      ffnHeat[i] *= 0.88;
      const f = ffnFlash[i];
      const h = ffnHeat[i];
      if (f < 0.03 && h < 0.05) {
        tmpColor.setRGB(0.035, 0.042, 0.055);
      } else {
        tmpColor.setRGB(0.25 + f * 0.85 + h * 0.2, 0.08 + f * 0.28, 0.16 + f * 0.55 + h * 0.12);
      }
      ffn.setColorAt(i, tmpColor);
    }
    ffn.instanceColor.needsUpdate = true;

    for (let b = 0; b < VOCAB_BINS; b++) {
      vocabFlash[b] *= 0.78;
      const f = vocabFlash[b] * (vocabRare[b] || 1);
      if (f < 0.02) {
        tmpColor.setRGB(0.07, 0.06, 0.045);
      } else {
        tmpColor.setRGB(0.5 + f * 0.5, 0.4 + f * 0.55, 0.12 + f * 0.25);
      }
      vocab.setColorAt(b, tmpColor);
    }
    vocab.instanceColor.needsUpdate = true;

    for (let i = flyers.length - 1; i >= 0; i--) {
      const fl = flyers[i];
      fl.age += dt;
      const t = Math.min(1, fl.age / fl.life);
      const ease = t * t * (3 - 2 * t);
      fl.sprite.position.set(
        bezier3(fl.sx, fl.mx, fl.ex, ease),
        bezier3(fl.sy, fl.my, fl.ey, ease),
        bezier3(fl.sz, fl.mz, fl.ez, ease),
      );
      const fade = t < 0.12 ? t / 0.12 : 1 - (t - 0.12) / 0.88;
      fl.sprite.material.opacity = Math.max(0, fade) * (0.55 + fl.sprite.userData.rare * 0.45);
      const s = 1 + (1 - t) * 0.35;
      fl.sprite.scale.set(
        Math.max(1.15, Math.min(3.6, fl.sprite.userData.text.length * 0.38)) * s,
        0.85 * s,
        1,
      );
      if (t >= 1) {
        scene.remove(fl.sprite);
        fl.sprite.material.dispose();
        flyers.splice(i, 1);
      }
    }

    motes.rotation.y += dt * 0.03;
  }

  const composer = new EffectComposer(renderer);
  const renderPass = new RenderPass(scene, camera);
  const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.72, 0.42, 0.32);
  composer.addPass(renderPass);
  composer.addPass(bloom);

  function resize() {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / Math.max(1, h);
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
    composer.setSize(w, h);
    bloom.resolution.set(w, h);
  }

  window.addEventListener("resize", resize);
  resize();

  let raf = 0;
  let last = performance.now();
  function tick(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    paintInstances(dt);
    const crane = Math.sin(now * 0.00012) * 1.15;
    camera.position.y = 1.8 + crane;
    controls.target.y = towerH * 0.42 + crane * 0.25;
    controls.update();
    composer.render();
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
        fl.sprite.material.dispose();
      }
      flyers.length = 0;
      renderer.dispose();
      composer.dispose();
      packGeo.dispose();
      spineGeo.dispose();
      ffnGeo.dispose();
      vocabGeo.dispose();
      moteGeo.dispose();
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
