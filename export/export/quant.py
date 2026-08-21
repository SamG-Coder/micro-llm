"""Quantization helpers.

v1 packs F16 / F32 tensors fully (dequant is a no-op / cast, gather, write).

Q4_K superblocks do not line up with arbitrary FFN channels or vocab rows.
The only correct path is dequant -> gather -> requant. Do NOT slice Q4_K
bytes in place.

gguf-py implements dequantize(Q4_K) but not quantize_blocks(Q4_K).
`requantize_q4_k` is the hook for a real encoder. Tests skip the requant
path rather than faking a Q4_K write.
"""

from __future__ import annotations

import numpy as np
from gguf import GGMLQuantizationType, dequantize

FLOAT_TYPES = {
    GGMLQuantizationType.F32,
    GGMLQuantizationType.F16,
    GGMLQuantizationType.BF16,
    GGMLQuantizationType.F64,
}

COPYABLE_RAW_TYPES = FLOAT_TYPES | {
    GGMLQuantizationType.I8,
    GGMLQuantizationType.I16,
    GGMLQuantizationType.I32,
    GGMLQuantizationType.I64,
}


class Q4KRequantNotImplemented(NotImplementedError):
    """Raised when a gathered tensor would need a real Q4_K requant.

    Implement `requantize_q4_k` with a correct Q4_K encoder (gguf-py does
    not ship `Q4_K.quantize_blocks`). Until then, either convert the source
    model to F16 or pass `--q4-k-to-f16` to emit F16 after dequant+gather.
    """


def is_float_type(qtype: GGMLQuantizationType) -> bool:
    return qtype in FLOAT_TYPES


def is_q4_k(qtype: GGMLQuantizationType) -> bool:
    return qtype == GGMLQuantizationType.Q4_K


def to_f32(data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
    if qtype == GGMLQuantizationType.F32:
        return np.asarray(data, dtype=np.float32)
    if qtype == GGMLQuantizationType.F16:
        return np.asarray(data, dtype=np.float32)
    if qtype == GGMLQuantizationType.BF16:
        # gguf dequantize handles BF16
        return np.asarray(dequantize(np.asarray(data), qtype), dtype=np.float32)
    if qtype == GGMLQuantizationType.F64:
        return np.asarray(data, dtype=np.float32)
    if qtype == GGMLQuantizationType.Q4_K:
        return np.asarray(dequantize(np.asarray(data), qtype), dtype=np.float32)
    raise TypeError(f"no dequant path for {qtype.name}")


def requantize_q4_k(array_f32: np.ndarray) -> np.ndarray:
    """HOOK: real Q4_K encoder. Not implemented in v1.

    Must produce ggml Q4_K superblocks for `array_f32` (last dim multiple of
    256). Do not invent a slice of the original Q4_K bytes ? channels will
    not land on superblock boundaries.
    """
    raise Q4KRequantNotImplemented(
        "Q4_K requant is not implemented in v1 (gguf-py has no Q4_K.quantize_blocks). "
        "Dequant -> gather is correct; a real requant encoder must be wired here. "
        "Use an F16/F32 source, or --q4-k-to-f16 to emit F16 after gather."
    )


def from_f32(
    array_f32: np.ndarray,
    qtype: GGMLQuantizationType,
    *,
    q4_k_to_f16: bool = False,
) -> tuple[np.ndarray, GGMLQuantizationType]:
    """Requantize gathered F32 back to `qtype`.

    Returns (array, output_qtype). Q4_K hits the hook unless q4_k_to_f16.
    """
    array_f32 = np.ascontiguousarray(array_f32, dtype=np.float32)
    if qtype == GGMLQuantizationType.F32:
        return array_f32, qtype
    if qtype == GGMLQuantizationType.F16:
        return np.ascontiguousarray(array_f32.astype(np.float16)), qtype
    if qtype == GGMLQuantizationType.F64:
        return np.ascontiguousarray(array_f32.astype(np.float64)), qtype
    if qtype == GGMLQuantizationType.Q4_K:
        if q4_k_to_f16:
            return np.ascontiguousarray(array_f32.astype(np.float16)), GGMLQuantizationType.F16
        requantize_q4_k(array_f32)
    raise TypeError(f"no requant path for {qtype.name}")
