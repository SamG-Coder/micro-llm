// Memory's 15.2 bar. The only memory truth. Heat is not gigabytes.
// Stack = weights + 0.9 CUDA + KV(ctx). Labeled example, not a measured remnant.

export const GIB = 1024 ** 3;
export const CUDA_SCRATCH_BYTES = Math.trunc(0.9 * GIB);
export const SERVE_USABLE_HEADLESS_BYTES = Math.trunc(15.2 * GIB);
export const YELLOW_BAND_BYTES = Math.trunc(0.5 * GIB);
export const KV_BYTES_PER_TOKEN_FP16 = 65536;
// Labeled example only. Not a measured remnant.
export const SAMPLE_WEIGHT_BYTES = Math.trunc(10.8 * GIB);
export const SAMPLE_BASE_CTX = 8192;
// Pinned 16 GA blocks at Q4. Live card stack, not the host GGUF file.
export const PINNED_GA_WEIGHT_BYTES = Math.trunc(0.6 * GIB);
export const PINNED_GA_KV_BYTES = Math.trunc(6.8 * GIB);
export const Q4_FFN_WORKSPACE_BYTES = 150 * 1024 * 1024;

export function kvBytes(ctx, perToken = KV_BYTES_PER_TOKEN_FP16) {
  return ctx * perToken;
}

export function serveStackBytes({
  weightBytes = SAMPLE_WEIGHT_BYTES,
  ctx = SAMPLE_BASE_CTX,
  cudaBytes = CUDA_SCRATCH_BYTES,
  kvPerToken = KV_BYTES_PER_TOKEN_FP16,
  ffnWorkspaceBytes = 0,
} = {}) {
  return weightBytes + cudaBytes + kvBytes(ctx, kvPerToken) + ffnWorkspaceBytes;
}

// Green under 15.2. Yellow inside 0.5GB of the cap. Red if serve_ok is false or vision is on.
export function barState({
  weightBytes = SAMPLE_WEIGHT_BYTES,
  ctx = SAMPLE_BASE_CTX,
  serveOk = true,
  vision = false,
  ffnWorkspaceBytes = 0,
} = {}) {
  const stack = serveStackBytes({ weightBytes, ctx, ffnWorkspaceBytes });
  const usable = SERVE_USABLE_HEADLESS_BYTES;
  let color = "green";
  if (!serveOk || vision || stack > usable) {
    color = "red";
  } else if (stack >= usable - YELLOW_BAND_BYTES) {
    color = "yellow";
  }
  return {
    stack,
    usable,
    weightBytes,
    cudaBytes: CUDA_SCRATCH_BYTES,
    kvBytes: kvBytes(ctx),
    ffnWorkspaceBytes,
    ctx,
    serveOk,
    vision,
    color,
    fill: Math.min(1, stack / usable),
  };
}

export function formatGiB(bytes) {
  return (bytes / GIB).toFixed(2);
}

// Sample replay: ctx = 8192 + token_index. KV grows 64KB/token. Bar barely moves.
export function sampleBudgetAtToken(tokenIndex) {
  return barState({
    weightBytes: SAMPLE_WEIGHT_BYTES,
    ctx: SAMPLE_BASE_CTX + tokenIndex,
    serveOk: true,
    vision: false,
  });
}

// Live card stack: pinned GA + ~0.9 CUDA + one FFN workspace + KV.
// Not 10.8, not host GGUF, not 64 parked FFNs.
export function liveCardStackAtToken(tokenIndex, extra = {}) {
  return barState({
    weightBytes: extra.weightBytes ?? PINNED_GA_WEIGHT_BYTES,
    ctx: (extra.baseCtx ?? SAMPLE_BASE_CTX) + tokenIndex,
    serveOk: true,
    vision: false,
    ffnWorkspaceBytes: extra.ffnWorkspaceBytes ?? Q4_FFN_WORKSPACE_BYTES,
  });
}
