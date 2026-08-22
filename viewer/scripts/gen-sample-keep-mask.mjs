// Writes viewer/sample/sample_keep_mask.json. Keep vs dropped only. No state field.

import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const N_LAYER = 64;
const N_FFN = 17408;
const RANGES = [
  [0, 63],
  [1000, 1063],
  [17344, 17407],
];
const DROPPED_PACKS = new Set([10, 22, 33]);

const channels = [];
for (const [lo, hi] of RANGES) {
  for (let i = lo; i <= hi; i++) channels.push(i);
}
if (channels.length !== 192) {
  throw new Error(`expected 192 keep channels, got ${channels.length}`);
}

const keep_channels = {};
for (let layer = 0; layer < N_LAYER; layer++) {
  keep_channels[String(layer)] = channels;
}

const keep_packs = [];
for (let p = 0; p < 48; p++) {
  if (!DROPPED_PACKS.has(p)) keep_packs.push(p);
}

const vocab_remap = {};
for (let i = 0; i <= 255; i++) vocab_remap[String(i)] = i;
vocab_remap["1000"] = 256;
vocab_remap["2000"] = 257;
vocab_remap["5000"] = 258;
vocab_remap["10000"] = 259;
vocab_remap["50000"] = 260;
vocab_remap["100000"] = 261;
vocab_remap["200000"] = 262;
vocab_remap["248319"] = 263;

const doc = {
  sample: true,
  note: "Labeled example only. Not a measured remnant. Keep vs dropped only — no state field.",
  n_layer: N_LAYER,
  n_ffn: N_FFN,
  keep_channels,
  keep_packs,
  vocab_remap,
  keep_vision: false,
  keep_mtp: false,
  serve_ok: true,
};

const out = join(dirname(fileURLToPath(import.meta.url)), "..", "sample", "sample_keep_mask.json");
writeFileSync(out, `${JSON.stringify(doc, null, 2)}\n`);
console.log(out, "layers", N_LAYER, "keep/layer", channels.length, "packs", keep_packs.length);
