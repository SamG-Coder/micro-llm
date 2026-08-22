// Sample keep-mask. Keep vs dropped only. No state field.
// Unwired / dead / fired / weak / floor are MLPT + ring. We have no MLPT.

import {
  FFN_BINS_PER_LAYER,
  N_FFN,
  N_LAYERS,
  N_PACKS,
  ffnBinIndex,
} from "./constants.js";

export const SAMPLE_KEEP_MASK_PATH = "viewer/sample/sample_keep_mask.json";

export const SAMPLE_KEEP_CHANNEL_RANGES = Object.freeze([
  [0, 63],
  [1000, 1063],
  [17344, 17407],
]);

export const SAMPLE_DROPPED_PACKS = Object.freeze([10, 22, 33]);

export function expandRanges(ranges) {
  const out = [];
  for (const [lo, hi] of ranges) {
    for (let i = lo; i <= hi; i++) out.push(i);
  }
  return out;
}

export function loadKeepMask(raw) {
  if (!raw || typeof raw !== "object") {
    throw new Error("keep-mask missing");
  }
  if ("state" in raw) {
    throw new Error("keep-mask has a state field; keep vs dropped only");
  }
  const nLayer = raw.n_layer;
  const nFfn = raw.n_ffn;
  if (nLayer !== N_LAYERS || nFfn !== N_FFN) {
    throw new Error(`keep-mask n_layer=${nLayer} n_ffn=${nFfn}`);
  }

  const keepChannels = [];
  const src = raw.keep_channels;
  for (let layer = 0; layer < nLayer; layer++) {
    const row = Array.isArray(src) ? src[layer] : src[String(layer)];
    if (!row) {
      throw new Error(`keep-mask missing layer ${layer}`);
    }
    keepChannels.push(Uint16Array.from(row));
  }

  const keepPacks = new Set(raw.keep_packs.map((p) => Number(p)));
  const vocabRemap = new Map();
  for (const [k, v] of Object.entries(raw.vocab_remap ?? {})) {
    vocabRemap.set(Number(k), Number(v));
  }

  return {
    sample: raw.sample === true,
    note: raw.note ?? "",
    nLayer,
    nFfn,
    keepChannels,
    keepPacks,
    vocabRemap,
    keepVision: raw.keep_vision === true,
    keepMtp: raw.keep_mtp === true,
    serveOk: raw.serve_ok === true,
  };
}

export function channelKept(mask, layer, channel) {
  const row = mask.keepChannels[layer];
  let lo = 0;
  let hi = row.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const v = row[mid];
    if (v === channel) return true;
    if (v < channel) lo = mid + 1;
    else hi = mid - 1;
  }
  return false;
}

export function packKept(mask, pack) {
  return mask.keepPacks.has(pack);
}

// Bins with no kept channel are not present. Not a five-state paint.
export function presentBins(mask) {
  const bins = new Uint8Array(N_LAYERS * FFN_BINS_PER_LAYER);
  for (let layer = 0; layer < N_LAYERS; layer++) {
    for (const ch of mask.keepChannels[layer]) {
      bins[layer * FFN_BINS_PER_LAYER + ffnBinIndex(ch)] = 1;
    }
  }
  return bins;
}

export function presentPacks(mask) {
  const out = new Uint8Array(N_PACKS);
  for (const p of mask.keepPacks) {
    if (p >= 0 && p < N_PACKS) out[p] = 1;
  }
  return out;
}
