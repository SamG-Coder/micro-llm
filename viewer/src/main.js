import "./style.css";
import sampleKeepMask from "../sample/sample_keep_mask.json";
import { FLAG_SPECIAL_OR_HIGH_LOSS } from "./constants.js";
import { formatGiB, sampleBudgetAtToken } from "./budget.js";
import { loadKeepMask, presentBins, presentPacks } from "./keepmask.js";
import {
  firedBinsFromBitset,
  generateSampleHTR1,
  hottestLayerThisToken,
  parseHTR1,
} from "./ring.js";
import { createMap } from "./scene.js";
import { decodeId } from "./vocab.js";

const TOKEN_MS = 220;

const hud = {
  tokens: document.getElementById("tokens"),
  layer: document.getElementById("layer"),
  out: document.getElementById("out"),
  flags: document.getElementById("flags"),
};
const fill = document.getElementById("budget-fill");
const budgetNums = document.getElementById("budget-nums");
const budgetParts = document.getElementById("budget-parts");

const keepMask = loadKeepMask(sampleKeepMask);
const sample = parseHTR1(generateSampleHTR1(42, 48));
const map = createMap(document.getElementById("stage"), {
  presentBins: presentBins(keepMask),
  presentPacks: presentPacks(keepMask),
});

const spoken = [];
let cursor = 0;

function paintBudget(tokenIndex) {
  const b = sampleBudgetAtToken(tokenIndex);
  fill.style.width = `${(b.fill * 100).toFixed(2)}%`;
  fill.className = `fill ${b.color}`;
  budgetNums.textContent = `${formatGiB(b.stack)} / ${formatGiB(b.usable)} GiB`;
  budgetParts.textContent =
    `w ${formatGiB(b.weightBytes)}  cuda ${formatGiB(b.cudaBytes)}  ` +
    `kv ${formatGiB(b.kvBytes)}  ctx ${b.ctx}  serve_ok  no vision` +
    `  · labeled example`;
}

function step() {
  const rec = sample.records[cursor];
  const bins = firedBinsFromBitset(rec.ffnFired, keepMask);
  map.applyToken(rec, bins);

  const word = decodeId(rec.sampledId);
  spoken.push(word);
  if (spoken.length > 96) spoken.shift();
  hud.out.textContent = spoken.join("");
  hud.tokens.textContent = String(rec.tokenIndex + 1);
  const layer = hottestLayerThisToken(rec.ffnFired, keepMask);
  hud.layer.textContent = layer === null ? "—" : `L${layer}`;
  const special = (rec.flags & FLAG_SPECIAL_OR_HIGH_LOSS) !== 0;
  hud.flags.textContent = special ? "special/high-loss" : "";
  paintBudget(rec.tokenIndex);

  cursor += 1;
  if (cursor >= sample.records.length) {
    cursor = 0;
    spoken.push("\n");
  }
}

paintBudget(0);
step();
setInterval(step, TOKEN_MS);
