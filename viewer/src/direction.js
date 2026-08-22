// Direction paint, not a compass. Heading is up the 64-layer tower.
// Color is pack vs FFN. Length is bits fired (packs use rel_residual).
// Zero bits → no streak. No NSEW.

import { N_PACKS, layerFromPackId } from "./constants.js";
import { packSpikeBit } from "./ring.js";

export function countFiredBits(ffnFired) {
  if (!ffnFired) return 0;
  let n = 0;
  for (let i = 0; i < ffnFired.length; i++) {
    let v = ffnFired[i];
    while (v) {
      n += v & 1;
      v >>= 1;
    }
  }
  return n;
}

export function packLength(record) {
  if (!record) return 0;
  const rel = record.packRelResidual;
  let len = 0;
  for (let p = 0; p < N_PACKS; p++) {
    if (packSpikeBit(record.packSpike, p)) {
      const r = rel && rel[p] != null ? Number(rel[p]) : 0;
      len += r > 0 ? r : 1;
    }
  }
  return len;
}

export function hottestPackLayer(record) {
  if (!record) return null;
  const rel = record.packRelResidual;
  let best = null;
  let bestR = -1;
  for (let p = 0; p < N_PACKS; p++) {
    if (!packSpikeBit(record.packSpike, p)) continue;
    const r = rel && rel[p] != null ? Number(rel[p]) : 0;
    if (r >= bestR) {
      bestR = r;
      best = layerFromPackId(p);
    }
  }
  return best;
}

// heading = hottest layer this token (up the 64-layer tower).
// kind = pack | ffn. Zero energy → null (no streak).
export function tokenDirection(record, extra = {}) {
  const ffnBits =
    extra.ffnBitCount != null ? extra.ffnBitCount : countFiredBits(record?.ffnFired);
  const packLen = packLength(record);
  if (ffnBits <= 0 && packLen <= 0) {
    return null;
  }
  const kind = packLen > ffnBits ? "pack" : "ffn";
  const length = kind === "pack" ? packLen : ffnBits;
  let heading = extra.hottestLayer;
  if (heading == null) {
    heading = kind === "pack" ? hottestPackLayer(record) : 0;
  }
  return { heading: heading ?? 0, kind, length };
}
