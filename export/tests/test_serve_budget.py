"""RTX 5080 serve stack: 12GB is weights; 15.2GB is weights + 0.9 CUDA + KV."""

from __future__ import annotations

from gguf import GGUFReader

from export.names import (
    KV_CUDA_SCRATCH_BYTES,
    KV_PER_TOKEN_FP16,
    KV_PER_TOKEN_FP8,
    KV_SERVE_USABLE_BYTES,
    KV_WEIGHT_BYTES,
)
from export.serve_budget import (
    CUDA_SCRATCH_BYTES,
    GIB,
    KV_BYTES_PER_TOKEN_FP16,
    KV_BYTES_PER_TOKEN_FP8,
    SERVE_USABLE_HEADLESS_BYTES,
    kv_bytes,
    serve_allowed,
    serve_reason,
    serve_stack_bytes,
)


W12 = 12 * GIB
W145 = int(14.5 * GIB)


def test_12gb_8k_fp16_allows():
    # 12GB weights + 0.9 CUDA + 8k FP16 KV (0.5GB) = 13.4 < 15.2
    assert kv_bytes(8192, "fp16") == 8192 * KV_BYTES_PER_TOKEN_FP16
    stack = serve_stack_bytes(W12, 8192)
    assert stack == W12 + CUDA_SCRATCH_BYTES + kv_bytes(8192, "fp16")
    assert stack < SERVE_USABLE_HEADLESS_BYTES
    assert serve_allowed(W12, 8192, serve_ok=True, keep_vision=False) is True
    assert serve_reason(W12, 8192, serve_ok=True, keep_vision=False) is None


def test_145gb_weights_8k_refuses():
    # 14.5GB weights + 0.9 + 0.5 = 15.9 → refuse
    assert serve_allowed(W145, 8192, serve_ok=True, keep_vision=False) is False
    reason = serve_reason(W145, 8192, serve_ok=True, keep_vision=False)
    assert reason is not None
    assert "exceeds usable" in reason


def test_serve_ok_false_refuses():
    assert serve_allowed(W12, 8192, serve_ok=False, keep_vision=False) is False
    assert serve_reason(W12, 8192, serve_ok=False, keep_vision=False) == "serve_ok is false"


def test_keep_vision_true_refuses():
    assert serve_allowed(W12, 8192, serve_ok=True, keep_vision=True) is False
    assert serve_reason(W12, 8192, serve_ok=True, keep_vision=True) == "keep_vision is true"


def test_12gb_32k_fp16_headless_allows():
    # 12GB + 32k FP16 KV (2GB) + 0.9 = 14.9 < 15.2
    assert serve_allowed(W12, 32768, serve_ok=True, keep_vision=False) is True
    assert serve_reason(W12, 32768, serve_ok=True, keep_vision=False) is None


def test_12gb_32k_fp16_display_refuses():
    # 12GB + 32k FP16 + 0.9 on display 14.5 → refuse
    assert serve_allowed(W12, 32768, serve_ok=True, keep_vision=False, display=True) is False
    reason = serve_reason(W12, 32768, serve_ok=True, keep_vision=False, display=True)
    assert reason is not None
    assert "exceeds usable" in reason


def test_baked_serve_stack_keys(remnant_gguf):
    reader = GGUFReader(str(remnant_gguf))
    assert reader.get_field(KV_CUDA_SCRATCH_BYTES).contents() == CUDA_SCRATCH_BYTES
    assert reader.get_field(KV_PER_TOKEN_FP16).contents() == KV_BYTES_PER_TOKEN_FP16
    assert reader.get_field(KV_PER_TOKEN_FP8).contents() == KV_BYTES_PER_TOKEN_FP8
    assert reader.get_field(KV_SERVE_USABLE_BYTES).contents() == SERVE_USABLE_HEADLESS_BYTES
    # Serve prefers on-disk file size; writer patches this after stream-write.
    assert reader.get_field(KV_WEIGHT_BYTES).contents() == remnant_gguf.stat().st_size
