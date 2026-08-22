// Locked facts. Same numbers as docs/ARCHITECTURE.md and docs/HOOK_RING.md.

export const N_LAYERS = 64;
export const N_FFN = 17408;
export const N_GROUPS = 16;
export const GROUP_SIZE = 4;
export const N_PACKS = 48;
export const N_SPINE = 16;
export const VOCAB_SIZE = 248320;
export const FIRE_EPS = 1e-6;
export const SPIKE_EPS = 0.02;
export const RING_DEPTH = 64;
export const TOPK_MAX = 64;
// Sample replay step. Fast enough to read as a stream, not a nightclub strobe.
export const SAMPLE_TOKEN_MS = 48;

export const HTR1_MAGIC = "HTR1";
export const HTR1_VERSION = 1;
export const HTR1_HEADER_BYTES = 16;

// Per-token record. Little-endian. See docs/HOOK_RING.md.
export const OFF_TOKEN_INDEX = 0;
export const OFF_SAMPLED_ID = 4;
export const OFF_FLAGS = 8;
export const OFF_N_TOPK = 12;
export const OFF_TOPK = 16;
export const OFF_FIRE_EPS = 272;
export const OFF_SPIKE_EPS = 276;
export const OFF_FFN_FIRED = 280;
export const FFN_BITSET_BYTES = (N_LAYERS * N_FFN) / 8; // 139264
export const OFF_PACK_REL = OFF_FFN_FIRED + FFN_BITSET_BYTES; // 139544
export const OFF_PACK_SPIKE = OFF_PACK_REL + N_PACKS * 4; // 139736
export const RECORD_BYTES = OFF_PACK_SPIKE + 8; // 139744

export const FLAG_SPECIAL_OR_HIGH_LOSS = 1;

// FFN heat-strip: 64 bins/layer. 17408 / 64 = 272.
export const FFN_BINS_PER_LAYER = 64;
export const FFN_CHANNELS_PER_BIN = N_FFN / FFN_BINS_PER_LAYER;

export const VOCAB_BINS = 256;

// Floor layers: do not drop. Packs on 0/1/62; 63 is GA. Scaffold pulses from spike only.
export const FLOOR_LAYERS = Object.freeze([0, 1, 62, 63]);

export function isGatedAttentionLayer(layer) {
  return layer % GROUP_SIZE === GROUP_SIZE - 1;
}

export function packIdFromLayer(layer) {
  const group = Math.floor(layer / GROUP_SIZE);
  const slot = layer % GROUP_SIZE;
  return 3 * group + slot;
}

export function layerFromPackId(pack) {
  const group = Math.floor(pack / 3);
  const slot = pack % 3;
  return GROUP_SIZE * group + slot;
}

export function channelBitIndex(layer, channel) {
  return layer * N_FFN + channel;
}

export function ffnBinIndex(channel) {
  return Math.min(FFN_BINS_PER_LAYER - 1, Math.floor(channel / FFN_CHANNELS_PER_BIN));
}

export function vocabBinIndex(id) {
  if (id >= VOCAB_SIZE) return VOCAB_BINS - 1;
  return Math.min(VOCAB_BINS - 1, Math.floor((id / VOCAB_SIZE) * VOCAB_BINS));
}

// Higher tokenizer ids are treated as rarer. Brighter spark.
export function rarityGlow(id) {
  const t = Math.log1p(Math.min(VOCAB_SIZE - 1, Math.max(0, id))) / Math.log1p(VOCAB_SIZE - 1);
  return 0.28 + 0.72 * t;
}
