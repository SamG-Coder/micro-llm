// HTR1 hook ring. C++ writes this later. This file only consumes / samples it.
// Little-endian. Layout is docs/HOOK_RING.md. Not an MLPT.

import {
  FFN_BITSET_BYTES,
  FFN_BINS_PER_LAYER,
  FLAG_SPECIAL_OR_HIGH_LOSS,
  FIRE_EPS,
  HTR1_HEADER_BYTES,
  HTR1_MAGIC,
  HTR1_VERSION,
  N_FFN,
  N_LAYERS,
  N_PACKS,
  OFF_FFN_FIRED,
  OFF_FIRE_EPS,
  OFF_FLAGS,
  OFF_N_TOPK,
  OFF_PACK_REL,
  OFF_PACK_SPIKE,
  OFF_SAMPLED_ID,
  OFF_SPIKE_EPS,
  OFF_TOKEN_INDEX,
  OFF_TOPK,
  RECORD_BYTES,
  RING_DEPTH,
  SPIKE_EPS,
  TOPK_MAX,
  channelBitIndex,
  ffnBinIndex,
} from "./constants.js";
import { RARE_ID, SAMPLE_IDS, SPECIAL_TOKEN_INDEX } from "./vocab.js";

export { RECORD_BYTES };

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function writeMagic(view, offset, magic) {
  for (let i = 0; i < 4; i++) {
    view.setUint8(offset + i, magic.charCodeAt(i));
  }
}

function readMagic(view, offset) {
  return String.fromCharCode(
    view.getUint8(offset),
    view.getUint8(offset + 1),
    view.getUint8(offset + 2),
    view.getUint8(offset + 3),
  );
}

export function bitSet(bytes, i) {
  bytes[i >> 3] = bytes[i >> 3] | (1 << (i & 7));
}

export function bitTest(bytes, i) {
  return (bytes[i >> 3] & (1 << (i & 7))) !== 0;
}

export function encodeRecord(record) {
  const buf = new ArrayBuffer(RECORD_BYTES);
  const view = new DataView(buf);
  const u8 = new Uint8Array(buf);

  view.setUint32(OFF_TOKEN_INDEX, record.tokenIndex, true);
  view.setUint32(OFF_SAMPLED_ID, record.sampledId, true);
  view.setUint32(OFF_FLAGS, record.flags, true);
  const nTopk = Math.min(TOPK_MAX, record.topk.length);
  view.setUint32(OFF_N_TOPK, nTopk, true);
  for (let i = 0; i < TOPK_MAX; i++) {
    view.setUint32(OFF_TOPK + i * 4, i < nTopk ? record.topk[i] : 0, true);
  }
  view.setFloat32(OFF_FIRE_EPS, record.fireEps ?? FIRE_EPS, true);
  view.setFloat32(OFF_SPIKE_EPS, record.spikeEps ?? SPIKE_EPS, true);

  const fired = u8.subarray(OFF_FFN_FIRED, OFF_FFN_FIRED + FFN_BITSET_BYTES);
  if (record.ffnFired instanceof Uint8Array && record.ffnFired.length === FFN_BITSET_BYTES) {
    fired.set(record.ffnFired);
  } else {
    for (const { layer, channel } of record.fires ?? []) {
      bitSet(fired, channelBitIndex(layer, channel));
    }
  }

  const residuals = record.packRelResidual;
  let spike = 0n;
  const spikeEps = record.spikeEps ?? SPIKE_EPS;
  for (let p = 0; p < N_PACKS; p++) {
    const r = residuals ? residuals[p] : 0;
    view.setFloat32(OFF_PACK_REL + p * 4, r, true);
    if (r > spikeEps) {
      spike |= 1n << BigInt(p);
    }
  }
  if (record.packSpike !== undefined) {
    spike = BigInt(record.packSpike);
  }
  view.setBigUint64(OFF_PACK_SPIKE, spike, true);
  return buf;
}

export function decodeRecord(buf, offset = 0) {
  const view = new DataView(buf, offset, RECORD_BYTES);
  const u8 = new Uint8Array(buf, offset, RECORD_BYTES);
  const nTopk = view.getUint32(OFF_N_TOPK, true);
  const topk = [];
  for (let i = 0; i < Math.min(TOPK_MAX, nTopk); i++) {
    topk.push(view.getUint32(OFF_TOPK + i * 4, true));
  }
  const packRelResidual = new Float32Array(N_PACKS);
  for (let p = 0; p < N_PACKS; p++) {
    packRelResidual[p] = view.getFloat32(OFF_PACK_REL + p * 4, true);
  }
  return {
    tokenIndex: view.getUint32(OFF_TOKEN_INDEX, true),
    sampledId: view.getUint32(OFF_SAMPLED_ID, true),
    flags: view.getUint32(OFF_FLAGS, true),
    nTopk,
    topk,
    fireEps: view.getFloat32(OFF_FIRE_EPS, true),
    spikeEps: view.getFloat32(OFF_SPIKE_EPS, true),
    ffnFired: u8.slice(OFF_FFN_FIRED, OFF_FFN_FIRED + FFN_BITSET_BYTES),
    packRelResidual,
    packSpike: view.getBigUint64(OFF_PACK_SPIKE, true),
  };
}

export function encodeHTR1(records) {
  if (records.length > RING_DEPTH) {
    throw new Error(`ring depth is ${RING_DEPTH}; got ${records.length}`);
  }
  const buf = new ArrayBuffer(HTR1_HEADER_BYTES + records.length * RECORD_BYTES);
  const view = new DataView(buf);
  writeMagic(view, 0, HTR1_MAGIC);
  view.setUint32(4, HTR1_VERSION, true);
  view.setUint32(8, N_LAYERS, true);
  view.setUint32(12, N_FFN, true);
  const out = new Uint8Array(buf);
  for (let i = 0; i < records.length; i++) {
    const rec = records[i] instanceof ArrayBuffer ? records[i] : encodeRecord(records[i]);
    out.set(new Uint8Array(rec), HTR1_HEADER_BYTES + i * RECORD_BYTES);
  }
  return buf;
}

export function parseHTR1(buf) {
  if (buf.byteLength < HTR1_HEADER_BYTES) {
    throw new Error("HTR1 truncated header");
  }
  const view = new DataView(buf);
  const magic = readMagic(view, 0);
  if (magic !== HTR1_MAGIC) {
    throw new Error(`bad magic ${magic}`);
  }
  const version = view.getUint32(4, true);
  const nLayers = view.getUint32(8, true);
  const nFfn = view.getUint32(12, true);
  if (version !== HTR1_VERSION || nLayers !== N_LAYERS || nFfn !== N_FFN) {
    throw new Error(`unsupported HTR1 v${version} layers=${nLayers} ffn=${nFfn}`);
  }
  const body = buf.byteLength - HTR1_HEADER_BYTES;
  if (body % RECORD_BYTES !== 0) {
    throw new Error(`HTR1 body ${body} is not a multiple of ${RECORD_BYTES}`);
  }
  const n = body / RECORD_BYTES;
  if (n > RING_DEPTH) {
    throw new Error(`ring depth is ${RING_DEPTH}; file has ${n}`);
  }
  const records = [];
  for (let i = 0; i < n; i++) {
    records.push(decodeRecord(buf, HTR1_HEADER_BYTES + i * RECORD_BYTES));
  }
  return { version, nLayers, nFfn, records };
}

export function packSpikeBit(packSpike, pack) {
  return (packSpike & (1n << BigInt(pack))) !== 0n;
}

export function firedBinsFromBitset(ffnFired) {
  const bins = new Uint8Array(N_LAYERS * FFN_BINS_PER_LAYER);
  for (let layer = 0; layer < N_LAYERS; layer++) {
    const base = layer * N_FFN;
    for (let ch = 0; ch < N_FFN; ch++) {
      if (bitTest(ffnFired, base + ch)) {
        bins[layer * FFN_BINS_PER_LAYER + ffnBinIndex(ch)] = 1;
      }
    }
  }
  return bins;
}

export function layersFiredThisToken(ffnFired) {
  const layers = [];
  for (let layer = 0; layer < N_LAYERS; layer++) {
    const base = layer * N_FFN;
    let any = false;
    const end = base + N_FFN;
    for (let i = base; i < end; i++) {
      if (bitTest(ffnFired, i)) {
        any = true;
        break;
      }
    }
    if (any) layers.push(layer);
  }
  return layers;
}

export function lastFiredLayer(ffnFired) {
  const layers = layersFiredThisToken(ffnFired);
  return layers.length ? layers[layers.length - 1] : null;
}

// Sparse, deterministic sample. A few FFN channels per layer, a few pack spikes.
// Quiet layers are not-this-token. Does not invent unwired/dead/weak/floor.
export function generateSampleRecords(seed = 42, nTokens = 48) {
  const n = Math.min(RING_DEPTH, nTokens);
  const rng = mulberry32(seed);
  const records = [];

  for (let t = 0; t < n; t++) {
    const sampledId = SAMPLE_IDS[t % SAMPLE_IDS.length];
    const flags = t === SPECIAL_TOKEN_INDEX ? FLAG_SPECIAL_OR_HIGH_LOSS : 0;

    const fires = [];
    for (let layer = 0; layer < N_LAYERS; layer++) {
      // A few quiet layers this token. Not unwired.
      if (rng() < 0.07) continue;
      const nFire = 2 + Math.floor(rng() * 4); // 2..5
      for (let i = 0; i < nFire; i++) {
        fires.push({ layer, channel: Math.floor(rng() * N_FFN) });
      }
    }

    const packRelResidual = new Float32Array(N_PACKS);
    const nSpike = 3 + Math.floor(rng() * 4); // 3..6
    const spiked = new Set();
    while (spiked.size < nSpike) {
      spiked.add(Math.floor(rng() * N_PACKS));
    }
    for (let p = 0; p < N_PACKS; p++) {
      packRelResidual[p] = spiked.has(p) ? SPIKE_EPS + 0.01 + rng() * 0.12 : rng() * (SPIKE_EPS * 0.7);
    }

    const topk = [sampledId];
    const extra = 7;
    for (let i = 0; i < extra; i++) {
      if (sampledId === RARE_ID && i === 0) {
        topk.push(RARE_ID);
        continue;
      }
      topk.push(10 + Math.floor(rng() * 26));
    }
    if (sampledId === RARE_ID && !topk.includes(RARE_ID)) {
      topk[1] = RARE_ID;
    }

    records.push({
      tokenIndex: t,
      sampledId,
      flags,
      topk,
      fireEps: FIRE_EPS,
      spikeEps: SPIKE_EPS,
      fires,
      packRelResidual,
    });
  }
  return records;
}

export function generateSampleHTR1(seed = 42, nTokens = 48) {
  return encodeHTR1(generateSampleRecords(seed, nTokens));
}
