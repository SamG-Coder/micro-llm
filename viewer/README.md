# viewer

Hotspot map. Sample ring when the exe opens with no args / `--ui`. Live
hour when the host posts HTR1 records (`chrome.webview` message /
`window.__htr1Push`). No npm at runtime. No PID scrape.

Paints [docs/HOOK_RING.md](../docs/HOOK_RING.md). 64-layer 3+1 hybrid (16 groups of 3 DeltaNet packs + 1 Gated Attention). Packs are global 0..47, `layer = 4*group+slot`. Not 17408 cubes.

The map is a **dark 3D volume** — star field, not a nightclub, not a bin grid. Outgoing words/ids are quiet streaks + dim glyphs through the volume. Spine is a dim point (spark then heat, never off, never a cube). A kept pack that spiked is a brief point or streak, not a cube tick. Fired FFN bits are points or streaks in the layer volume. Same HTR1 bins, no standing cube grid. A zero bit stays dark. No bloom pass.

Replay is a generated HTR1 ring (depth ≤ 64) until a live record arrives.
The sample `setInterval` is cleared on the first host frame. Tokens move
only when a real record is pushed. C++ encodes this layout from TraceHooks.

Labeled keep-mask: [sample/sample_keep_mask.json](sample/sample_keep_mask.json). Keep vs dropped only. No state field. Not a measured remnant. Dropped channels/packs are not present. Live flash still comes only from the sample hook ring.

## Run (no npm)

The committed snapshot is [harness/ui/](../harness/ui/). On Windows,
`micro-llm-trace.exe` (no args or `--ui`) opens the sample map.
`--model` / `--ui --model` opens the live hour window. Do not run `npm`
to see the map.

## Rebuild the snapshot (maintainers)

```bash
cd viewer
npm install
npm test
npm run build
# or: scripts/sync-harness-ui.sh
```

`npm start` is still a Vite dev server. SamG does not need it.

## What it paints

From each hook-ring record, and only from those bits:

- FFN heat-strip bins that have a fired bit this token on a kept channel (point/streak in the layer volume, then fade). A zero bit does not flash and is not drawn as a cube. A dropped channel is not present. A quiet layer this token is not-this-token, not unwired.
- Pack points whose `pack_spike` bit is set, if the pack is kept. Dropped packs (sample: 10, 22, 33) are not present. Brief point or streak from the spike mask only. No cube ticks.
- Vocab sparks on `sampled_id` and top-k. Rarer ids a little brighter. Toy vocab decodes a few ids; everything else prints the id. Those words/ids fly as quiet streaks + dim glyphs, not HUD pop words or cubes.
- Spine (16 Gated Attention blocks) always a dim point. Each token sparks it, then the spark fades into heat. Spine never goes off and is never a cube. Spine lighting is not from the hook ring.

Unwired / dead / fired / weak / floor are not painted. Those five states need MLPT scores + the hook ring. There is no MLPT in this tree. The keep-mask has no state field.

## 15.2 bar

Sample (no live attach): labeled example 10.8 + 0.9 + KV. Not a remnant.

Live: **card stack** = pinned GA weights (~0.6 GiB) + 0.9 CUDA + KV.
Not the 10.8 sample, not the host GGUF file size.

## What it does not

- Attach to a process or PID (the exe posts records; no PID scrape)
- Drive tokens from a 16/48/160ms timer while live
- Invent glow for a zero bit
- Keep more than 64 records
- Put `n_fired` / sumsq / keep-mask / gigabytes in the ring
- Ship outgoing tokens as a 2D ticker strip
