# viewer

Sample-only hotspot map. No live attach. No PID. No hour.mlpt. No `C:\models`.

Paints [docs/HOOK_RING.md](../docs/HOOK_RING.md). 64-layer 3+1 hybrid (16 groups of 3 DeltaNet packs + 1 Gated Attention). Packs are global 0..47, `layer = 4*group+slot`. Not 17408 cubes.

Replay is a generated HTR1 ring (depth ≤ 64). C++ will write this layout later. This package only consumes it.

## Run

```bash
cd viewer
npm install
npm start
```

Opens a page that replays the sample ring: FFN flashes, fade, spine stays lit, packs pulse, vocab sparks, token count climbs, words/ids come out. The 15.2 bar sits under the map and barely moves.

```bash
npm test
```

## What it paints

From each hook-ring record, and only from those bits:

- FFN heat-strip bins that have a fired bit this token (flash, then fade). A zero bit does not flash. A quiet layer this token is not-this-token, not unwired.
- Pack nodes whose `pack_spike` bit is set. Pulse from the spike mask only. Locked/live/no-op come later.
- Vocab strip on `sampled_id` and top-k. Rarer ids brighter. Toy vocab decodes a few ids; everything else prints the id.
- Spine (16 Gated Attention blocks) always lit. Each token sparks it, then the spark fades into heat. Spine never goes off. Spine lighting is not from the hook ring.

Unwired / dead / weak / floor are not decided here. Unwired is MLPT total `n_fired==0`. There is no MLPT in this tree.

## 15.2 bar

Under the map. The only memory truth: remnant weights + 0.9 CUDA + KV as ctx grows. Sample is a stable serve-like stack (11.5 GiB weights, `serve_ok`, no vision, ctx 8192 + token). Green under 15.2. Yellow inside 0.5GB of the cap. Red if `serve_ok` is false or vision is on. Heat is session energy, not gigabytes. The bar barely moves on purpose.

## What it does not

- Attach to a process or PID
- Open a WebSocket
- Invent glow for a zero bit
- Keep more than 64 records
- Put `n_fired` / sumsq / keep-mask / gigabytes in the ring
