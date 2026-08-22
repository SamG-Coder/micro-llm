import "./style.css";
import sampleKeepMask from "../sample/sample_keep_mask.json";
import { FLAG_SPECIAL_OR_HIGH_LOSS, SAMPLE_TOKEN_MS } from "./constants.js";
import { formatGiB, liveCardStackAtToken, sampleBudgetAtToken } from "./budget.js";
import { loadKeepMask, presentBins, presentPacks } from "./keepmask.js";
import { decodeHostPayload, installLivePush, stopSampleReplay } from "./live.js";
import {
  firedBinsFromBitset,
  generateSampleHTR1,
  hottestLayerThisToken,
  parseHTR1,
} from "./ring.js";
import { createMap } from "./scene.js";
import { decodeId } from "./vocab.js";

const TOKEN_MS = SAMPLE_TOKEN_MS;

const hud = {
  tokens: document.getElementById("tokens"),
  layer: document.getElementById("layer"),
  flags: document.getElementById("flags"),
};
const fill = document.getElementById("budget-fill");
const budgetNums = document.getElementById("budget-nums");
const budgetParts = document.getElementById("budget-parts");
const badge = document.querySelector(".badge");
const modeLine = document.querySelector("#hud-tl .k");
const capLabel = document.querySelector("#budget .cap span");

const keepMask = loadKeepMask(sampleKeepMask);
const sample = parseHTR1(generateSampleHTR1(42, 48));
let map = createMap(document.getElementById("stage"), {
  presentBins: presentBins(keepMask),
  presentPacks: presentPacks(keepMask),
});

let cursor = 0;
let played = 0;
let live = false;
let liveAttach = null;

function paintBudget(tokenIndex) {
  if (live) {
    const b = liveCardStackAtToken(tokenIndex, {
      weightBytes: liveAttach?.weightBytes,
      baseCtx: liveAttach?.ctx,
    });
    fill.style.width = `${(b.fill * 100).toFixed(2)}%`;
    fill.className = `fill ${b.color}`;
    budgetNums.textContent = `${formatGiB(b.stack)} / ${formatGiB(b.usable)} GiB`;
    budgetParts.textContent =
      `GA pin ${formatGiB(b.weightBytes)}  cuda ${formatGiB(b.cudaBytes)}  ` +
      `kv ${formatGiB(b.kvBytes)}  ctx ${b.ctx}  card stack  no host GGUF`;
    if (capLabel) capLabel.textContent = "15.2 card stack";
    return;
  }
  const b = sampleBudgetAtToken(tokenIndex);
  fill.style.width = `${(b.fill * 100).toFixed(2)}%`;
  fill.className = `fill ${b.color}`;
  budgetNums.textContent = `${formatGiB(b.stack)} / ${formatGiB(b.usable)} GiB`;
  budgetParts.textContent =
    `w ${formatGiB(b.weightBytes)}  cuda ${formatGiB(b.cudaBytes)}  ` +
    `kv ${formatGiB(b.kvBytes)}  ctx ${b.ctx}  serve_ok  no vision` +
    `  · labeled example 10.8 + 0.9 + KV`;
}

function applyRecord(rec, mask) {
  const bins = firedBinsFromBitset(rec.ffnFired, mask);
  const word = decodeId(rec.sampledId);
  map.applyToken(rec, bins, { word });
  played += 1;
  hud.tokens.textContent = String(played);
  const layer = hottestLayerThisToken(rec.ffnFired, mask);
  hud.layer.textContent = layer === null ? "—" : `L${layer}`;
  const special = (rec.flags & FLAG_SPECIAL_OR_HIGH_LOSS) !== 0;
  hud.flags.textContent = special ? "special/high-loss" : "";
  paintBudget(rec.tokenIndex);
}

function step() {
  if (live) return;
  const rec = sample.records[cursor];
  applyRecord(rec, keepMask);
  cursor += 1;
  if (cursor >= sample.records.length) {
    cursor = 0;
  }
}

function enterLive(attach) {
  stopSampleReplay();
  live = true;
  liveAttach = attach || liveAttach;
  if (badge) badge.textContent = "LIVE";
  if (modeLine) modeLine.textContent = "HTR1 · 64-layer 3+1 · live hour";
  document.title = "micro-llm hotspot — live hour";
  if (map && map.dispose) {
    map.dispose();
  }
  map = createMap(document.getElementById("stage"), {});
  played = 0;
  paintBudget(0);
}

function onHost(parsed) {
  if (parsed.kind === "attach") {
    enterLive(parsed.attach);
    return;
  }
  if (parsed.kind === "record") {
    if (!live) enterLive(null);
    applyRecord(parsed.record, null);
  }
}

paintBudget(0);
step();
if (window.__sampleReplay) clearInterval(window.__sampleReplay);
window.__sampleReplay = setInterval(step, TOKEN_MS);

installLivePush((parsed) => onHost(parsed));
// Named for hosts that call the function directly.
window.__htr1Push = (data) => onHost(decodeHostPayload(data));
