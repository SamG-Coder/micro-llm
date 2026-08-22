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

export function createMap(container) {
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
  const packMat = new THREE.MeshStandardMaterial({
    color: 0x1c2433,
    emissive: 0x000000,
    roughness: 0.45,
    metalness: 0.35,
  });
  const packs = new THREE.InstancedMesh(packGeo, packMat, N_PACKS);
  packs.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  const packFlash = new Float32Array(N_PACKS);
  for (let p = 0; p < N_PACKS; p++) {
    const layer = layerFromPackId(p);
    dummy.position.set(-1.15, layerY(layer), 0);
    dummy.rotation.set(0.2, 0.4, 0);
    dummy.scale.set(1, 1, 1);
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

  const ffnGeo = new THREE.BoxGeometry(0.1, 0.22, 0.1);
  const ffnMat = new THREE.MeshStandardMaterial({
    color: 0x141821,
    emissive: 0xff4d8d,
    emissiveIntensity: 0.15,
    roughness: 0.55,
    metalness: 0.1,
  });
  const ffn = new THREE.InstancedMesh(ffnGeo, ffnMat, FFN_COUNT);
  const ffnFlash = new Float32Array(FFN_COUNT);
  const ffnHeat = new Float32Array(FFN_COUNT);
  for (let layer = 0; layer < N_LAYERS; layer++) {
    for (let b = 0; b < FFN_BINS_PER_LAYER; b++) {
      const i = layer * FFN_BINS_PER_LAYER + b;
      dummy.position.set(0.7 + b * 0.12, layerY(layer), 0);
      dummy.scale.set(1, 1, 1);
      dummy.updateMatrix();
      ffn.setMatrixAt(i, dummy.matrix);
      ffn.setColorAt(i, hex(0x121722));
    }
  }
  scene.add(ffn);

  const vocabGeo = new THREE.BoxGeometry(0.085, 0.28, 0.12);
  const vocabMat = new THREE.MeshStandardMaterial({
    color: 0x16120c,
    emissive: 0xf5e6a3,
    emissiveIntensity: 0.2,
    roughness: 0.4,
  });
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

  for (let g = 0; g < N_GROUPS; g++) {
    const y0 = layerY(g * 4) - 0.12;
    const y1 = layerY(g * 4 + 3) + 0.12;
    const h = y1 - y0;
    const frame = new THREE.Mesh(
      new THREE.BoxGeometry(9.2, h, 0.02),
      new THREE.MeshBasicMaterial({ color: 0x121826, transparent: true, opacity: 0.35 }),
    );
    frame.position.set(3.6, (y0 + y1) / 2, -0.55);
    scene.add(frame);
  }

  function applyToken(record, bins) {
    for (let i = 0; i < FFN_COUNT; i++) {
      if (bins[i]) {
        ffnFlash[i] = 1;
        ffnHeat[i] = 0.35;
      }
    }
    for (let p = 0; p < N_PACKS; p++) {
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
      packFlash[p] *= 0.82;
      const pulse = packFlash[p];
      tmpColor.setRGB(0.14 + pulse * 0.86, 0.13 + pulse * 0.55, 0.16 * (1 - pulse));
      packs.setColorAt(p, tmpColor);
      const layer = layerFromPackId(p);
      dummy.position.set(-1.15, layerY(layer), 0);
      const s = 1 + pulse * 0.9;
      dummy.scale.set(s, s, s);
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
    spineMat.emissiveIntensity = 0.55 + 0.55 * Math.max(...spineSpark);

    for (let i = 0; i < FFN_COUNT; i++) {
      ffnFlash[i] *= 0.8;
      ffnHeat[i] *= 0.9;
      const f = ffnFlash[i];
      const h = ffnHeat[i];
      if (f < 0.02 && h < 0.04) {
        tmpColor.setRGB(0.07, 0.09, 0.12);
      } else {
        tmpColor.setRGB(0.14 + f * 0.86 + h * 0.25, 0.06 + f * 0.18 + h * 0.06, 0.16 + f * 0.35 + h * 0.12);
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
