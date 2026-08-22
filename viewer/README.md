# viewer

Sample-only hotspot map. No live attach. No PID. No hour.mlpt. No `C:\models`.

Paints [docs/HOOK_RING.md](../docs/HOOK_RING.md). 64-layer 3+1 hybrid (16 groups of 3 DeltaNet packs + 1 Gated Attention). Packs are global 0..47, `layer = 4*group+slot`. Not 17408 cubes.

The map is a **dark 3D volume** — star field, not a nightclub. Outgoing words/ids are quiet streaks through the tower. Spine stays dim (spark then heat, never off). Packs tick from spike bits only. FFN bins flash from this-token fired bits on kept channels. A zero bit stays dark. No bloom pass.

Replay is a generated HTR1 ring (depth ≤ 64). C++ will write this layout later. This package only consumes it.

Labeled keep-mask: [sample/sample_keep_mask.json](sample/sample_keep_mask.json). Keep vs dropped only. No state field. Not a measured remnant. Dropped channels/packs are not present. Live flash still comes only from the sample hook ring.

## Run (no npm)

The committed snapshot is [harness/ui/](../harness/ui/). On Windows, `micro-llm-trace.exe` (no args or `--ui`) or `micro-llm-view.exe` opens it in WebView2. Do not run `npm` to see the map.

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

- FFN heat-strip bins that have a fired bit this token on a kept channel (flash, then fade). A zero bit does not flash. A dropped channel is not present. A quiet layer this token is not-this-token, not unwired.
- Pack nodes whose `pack_spike` bit is set, if the pack is kept. Dropped packs (sample: 10, 22, 33) are not present. Pulse from the spike mask only.
- Vocab ring on `sampled_id` and top-k. Rarer ids a little brighter. Toy vocab decodes a few ids; everything else prints the id. Those words/ids fly as dim streaks, not HUD pop words.
- Spine (16 Gated Attention blocks) always lit. Each token sparks it, then the spark fades into heat. Spine never goes off. Spine lighting is not from the hook ring.

Unwired / dead / fired / weak / floor are not painted. Those five states need MLPT scores + the hook ring. There is no MLPT in this tree. The keep-mask has no state field.

## 15.2 bar

Corner chrome under the volume. Labeled example only — not a measured remnant, not live VRAM, not a serve budget: 10.8 + 0.9 + KV under 15.2. `serve_ok`, no vision, ctx 8192 + token. Green. The bar barely moves on purpose. Do not treat 10.8 as a real remnant.

## What it does not

- Attach to a process or PID
- Open a WebSocket
- Invent glow for a zero bit
- Keep more than 64 records
- Put `n_fired` / sumsq / keep-mask / gigabytes in the ring
- Ship outgoing tokens as a 2D ticker strip
