"""RTX 5080 serve-stack budget. Pure functions; no nvidia-smi.

12GB is the remnant *weight* ceiling (cut.py). This gate is the serve stack:
on-disk weights + CUDA/decode scratch + KV(ctx). Headless usable is 15.2 GiB;
display is 14.5 GiB. Coding remnant must not include vision.
"""

from __future__ import annotations

GIB = 1024 ** 3

# Locked facts. Integer bytes (GiB truncated toward zero).
CUDA_SCRATCH_BYTES = int(0.9 * GIB)
SERVE_USABLE_HEADLESS_BYTES = int(15.2 * GIB)
SERVE_USABLE_DISPLAY_BYTES = int(14.5 * GIB)

# 16 GQA layers, 4 KV heads, d=256.
KV_BYTES_PER_TOKEN_FP16 = 65536
KV_BYTES_PER_TOKEN_FP8 = 32768

DEFAULT_CTX = 8192


def kv_bytes(ctx: int = DEFAULT_CTX, dtype: str = "fp16") -> int:
    """KV cache bytes for ctx tokens. dtype is 'fp16' (64KB/tok) or 'fp8' (32KB/tok)."""
    if dtype == "fp16":
        return int(ctx) * KV_BYTES_PER_TOKEN_FP16
    if dtype == "fp8":
        return int(ctx) * KV_BYTES_PER_TOKEN_FP8
    raise ValueError(f"dtype must be 'fp16' or 'fp8', got {dtype!r}")


def serve_stack_bytes(
    weight_bytes: int,
    ctx: int = DEFAULT_CTX,
    *,
    kv_dtype: str = "fp16",
    cuda_bytes: int = CUDA_SCRATCH_BYTES,
    display: bool = False,
) -> int:
    """Weights + CUDA scratch + KV(ctx). display does not change the stack, only the ceiling."""
    del display
    return int(weight_bytes) + int(cuda_bytes) + kv_bytes(ctx, kv_dtype)


def serve_reason(
    weight_bytes: int,
    ctx: int = DEFAULT_CTX,
    *,
    serve_ok: bool = True,
    keep_vision: bool = False,
    kv_dtype: str = "fp16",
    cuda_bytes: int = CUDA_SCRATCH_BYTES,
    display: bool = False,
) -> str | None:
    """Refusal string, or None if the 5080 coding serve gate allows."""
    if not serve_ok:
        return "serve_ok is false"
    if keep_vision:
        return "keep_vision is true"
    stack = serve_stack_bytes(
        weight_bytes,
        ctx,
        kv_dtype=kv_dtype,
        cuda_bytes=cuda_bytes,
        display=display,
    )
    usable = SERVE_USABLE_DISPLAY_BYTES if display else SERVE_USABLE_HEADLESS_BYTES
    if stack > usable:
        return f"serve stack {stack} exceeds usable {usable}"
    return None


def serve_allowed(
    weight_bytes: int,
    ctx: int = DEFAULT_CTX,
    *,
    serve_ok: bool = True,
    keep_vision: bool = False,
    kv_dtype: str = "fp16",
    cuda_bytes: int = CUDA_SCRATCH_BYTES,
    display: bool = False,
) -> bool:
    return (
        serve_reason(
            weight_bytes,
            ctx,
            serve_ok=serve_ok,
            keep_vision=keep_vision,
            kv_dtype=kv_dtype,
            cuda_bytes=cuda_bytes,
            display=display,
        )
        is None
    )
