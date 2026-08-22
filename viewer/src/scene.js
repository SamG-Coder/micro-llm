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
import { packSpikeBit } from "./ring.js";

const FFN_COUNT = N_LAYERS * FFN_BINS_PER_LAYER;
const LAYER_H = 0.3;
const GROUP_GAP = 0.18;

function layerY(layer) {
  const group = Math.floor(layer / 4);
  return layer * LAYER_H + group * GROUP_GAP;
}

function hex(n) {
  return new THREE.Color(n);
}

export function createMap(container, { presentBins = null, presentPacks = null } = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x07080c, 1);
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x07080c, 18, 48);

  const towerH = layerY(N_LAYERS - 1);
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 80);
  camera.position.set(11, towerH * 0.42, 16);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(3.2, towerH * 0.48, 0);
  controls.enableDamping = true;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.55;
  controls.maxDistance = 40;
  controls.minDistance = 6;

  scene.add(new THREE.AmbientLight(0x6b7384, 0.55));
  const key = new THREE.DirectionalLight(0xd8e2f0, 0.7);
  key.position.set(8, 20, 10);
  scene.add(key);
  const spineLight = new THREE.PointLight(0x2ee6d6, 1.4, 22);
  spineLight.position.set(-0.2, towerH * 0.5, 1.2);
  scene.add(spineLight);

  const dummy = new THREE.Object3D();
  const tmpColor = new THREE.Color();

  const packGeo = new THREE.OctahedronGeometry(0.28, 0);
  const packMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const packs = new THREE.InstancedMesh(packGeo, packMat, N_PACKS);
  packs.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  const packFlash = new Float32Array(N_PACKS);
  for (let p = 0; p < N_PACKS; p++) {
    const layer = layerFromPackId(p);
    dummy.position.set(-1.15, layerY(layer), 0);
    dummy.rotation.set(0.2, 0.4, 0);
    const shown = !presentPacks || presentPacks[p];
    dummy.scale.set(shown ? 1 : 0, shown ? 1 : 0, shown ? 1 : 0);
    dummy.updateMatrix();
    packs.setMatrixAt(p, dummy.matrix);
    packs.setColorAt(p, hex(0x243044));
  }
  scene.add(packs);

  const spineGeo = new THREE.IcosahedronGeometry(0.32, 1);
  const spineMat = new THREE.MeshStandardMaterial({
    color: 0x164e49,
    emissive: 0x2ee6d6,
    emissiveIntensity: 0.7,
    roughness: 0.25,
    metalness: 0.2,
  });
  const spine = new THREE.InstancedMesh(spineGeo, spineMat, N_SPINE);
  const spineSpark = new Float32Array(N_SPINE);
  const spineHeat = new Float32Array(N_SPINE);
  spineHeat.fill(0.22);
  for (let g = 0; g < N_SPINE; g++) {
    const layer = g * 4 + 3;
    dummy.position.set(-0.15, layerY(layer), 0.15);
    dummy.scale.set(1, 1, 1);
    dummy.updateMatrix();
    spine.setMatrixAt(g, dummy.matrix);
    spine.setColorAt(g, hex(0x2ee6d6));
  }
  scene.add(spine);

  const ffnGeo = new THREE.BoxGeometry(0.09, 0.2, 0.09);
  const ffnMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const ffn = new THREE.InstancedMesh(ffnGeo, ffnMat, FFN_COUNT);
  const ffnFlash = new Float32Array(FFN_COUNT);
  const ffnHeat = new Float32Array(FFN_COUNT);
  for (let layer = 0; layer < N_LAYERS; layer++) {
    for (let b = 0; b < FFN_BINS_PER_LAYER; b++) {
      const i = layer * FFN_BINS_PER_LAYER + b;
      dummy.position.set(0.7 + b * 0.12, layerY(layer), 0);
      const shown = !presentBins || presentBins[i];
      dummy.scale.set(shown ? 1 : 0, shown ? 1 : 0, shown ? 1 : 0);
      dummy.updateMatrix();
      ffn.setMatrixAt(i, dummy.matrix);
      ffn.setColorAt(i, hex(0x121722));
    }
  }
  scene.add(ffn);

  const vocabGeo = new THREE.BoxGeometry(0.085, 0.28, 0.12);
  const vocabMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const vocab = new THREE.InstancedMesh(vocabGeo, vocabMat, VOCAB_BINS);
  const vocabFlash = new Float32Array(VOCAB_BINS);
  const vocabRare = new Float32Array(VOCAB_BINS);
  for (let b = 0; b < VOCAB_BINS; b++) {
    dummy.position.set(0.7 + b * (0.12 * FFN_BINS_PER_LAYER) / VOCAB_BINS, -1.15, 0.35);
    dummy.scale.set(1, 1, 1);
    dummy.updateMatrix();
    vocab.setMatrixAt(b, dummy.matrix);
    vocab.setColorAt(b, hex(0x1a160f));
  }
  scene.add(vocab);

  const rail = new THREE.Mesh(
    new THREE.BoxGeometry(0.06, towerH + 1.2, 0.06),
    new THREE.MeshStandardMaterial({ color: 0x1a2433, roughness: 0.8 }),
  );
  rail.position.set(-0.15, towerH * 0.5, -0.35);
  scene.add(rail);

  const groupLineMat = new THREE.MeshBasicMaterial({ color: 0x1a2230 });
  for (let g = 1; g < N_GROUPS; g++) {
    const y = (layerY(g * 4 - 1) + layerY(g * 4)) / 2;
    const sep = new THREE.Mesh(new THREE.BoxGeometry(8.4, 0.02, 0.02), groupLineMat);
    sep.position.set(3.4, y, -0.4);
    scene.add(sep);
  }

  function applyToken(record, bins) {
    for (let i = 0; i < FFN_COUNT; i++) {
      if (bins[i] && (!presentBins || presentBins[i])) {
        ffnFlash[i] = 1;
        ffnHeat[i] = 0.35;
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
      spineHeat[g] = Math.min(0.85, spineHeat[g] + 0.08);
    }
    vocabFlash.fill(0);
    vocabRare.fill(0);
    const ids = [record.sampledId, ...record.topk];
    for (const id of ids) {
      const b = vocabBinIndex(id);
      vocabFlash[b] = Math.max(vocabFlash[b], 1);
      vocabRare[b] = Math.max(vocabRare[b], rarityGlow(id));
    }
  }

  function paintInstances() {
    for (let p = 0; p < N_PACKS; p++) {
      packFlash[p] *= 0.84;
      const pulse = packFlash[p];
      if (pulse < 0.04) {
        tmpColor.setRGB(0.18, 0.22, 0.3);
      } else {
        tmpColor.setRGB(1.0, 0.62 + pulse * 0.2, 0.08);
      }
      packs.setColorAt(p, tmpColor);
      const layer = layerFromPackId(p);
      dummy.position.set(-1.15, layerY(layer), 0);
      if (presentPacks && !presentPacks[p]) {
        dummy.scale.set(0, 0, 0);
      } else {
        const s = 1 + pulse * 0.9;
        dummy.scale.set(s, s, s);
      }
      dummy.rotation.set(0.2, 0.4 + pulse * 1.2, 0);
      dummy.updateMatrix();
      packs.setMatrixAt(p, dummy.matrix);
    }
    packs.instanceColor.needsUpdate = true;
    packs.instanceMatrix.needsUpdate = true;

    for (let g = 0; g < N_SPINE; g++) {
      spineSpark[g] *= 0.88;
      const glow = 0.28 + spineHeat[g] * 0.45 + spineSpark[g] * 0.7;
      tmpColor.setRGB(0.1 + glow * 0.15, 0.55 + glow * 0.4, 0.52 + glow * 0.45);
      spine.setColorAt(g, tmpColor);
      const layer = g * 4 + 3;
      dummy.position.set(-0.15, layerY(layer), 0.15);
      const s = 1 + spineSpark[g] * 0.35;
      dummy.scale.set(s, s, s);
      dummy.updateMatrix();
      spine.setMatrixAt(g, dummy.matrix);
    }
    spine.instanceColor.needsUpdate = true;
    spine.instanceMatrix.needsUpdate = true;
    spineMat.emissiveIntensity = 0.65 + 0.7 * Math.max(...spineSpark);

    for (let i = 0; i < FFN_COUNT; i++) {
      ffnFlash[i] *= 0.78;
      ffnHeat[i] *= 0.88;
      const f = ffnFlash[i];
      const h = ffnHeat[i];
      if (f < 0.03 && h < 0.05) {
        tmpColor.setRGB(0.055, 0.065, 0.085);
      } else {
        tmpColor.setRGB(0.2 + f * 0.8 + h * 0.2, 0.05 + f * 0.25, 0.12 + f * 0.45 + h * 0.1);
      }
      ffn.setColorAt(i, tmpColor);
    }
    ffn.instanceColor.needsUpdate = true;

    for (let b = 0; b < VOCAB_BINS; b++) {
      vocabFlash[b] *= 0.8;
      const f = vocabFlash[b] * (vocabRare[b] || 1);
      if (f < 0.02) {
        tmpColor.setRGB(0.09, 0.08, 0.06);
      } else {
        tmpColor.setRGB(0.3 + f * 0.7, 0.28 + f * 0.6, 0.12 + f * 0.25);
      }
      vocab.setColorAt(b, tmpColor);
    }
    vocab.instanceColor.needsUpdate = true;
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
  function tick() {
    paintInstances();
    controls.update();
    renderer.render(scene, camera);
    raf = requestAnimationFrame(tick);
  }
  tick();

  return {
    applyToken,
    dispose() {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      controls.dispose();
      renderer.dispose();
      packGeo.dispose();
      spineGeo.dispose();
      ffnGeo.dispose();
      vocabGeo.dispose();
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
