"""Quantization helpers.

v1 packs F16 / F32 tensors fully (dequant is a no-op / cast, gather, write).

Q4_K superblocks do not line up with arbitrary FFN channels or vocab rows.
The only correct path is dequant -> gather -> requant. Do NOT slice Q4_K
bytes in place.

gguf-py implements dequantize(Q4_K) but not quantize_blocks(Q4_K).
`requantize_q4_k` ports llama.cpp `quantize_row_q4_K_ref` (the real Q4_K
block encoder used when there is no importance matrix). That is the Q4_K
format inside Q4_K_M remnants. --q4-k-to-f16 stays host-debug only.
"""

from __future__ import annotations

import numpy as np
from gguf import GGMLQuantizationType, dequantize
from gguf.constants import GGML_QUANT_SIZES
from gguf.quants import quant_shape_to_byte_shape

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

QK_K = 256
Q4_K_TYPE_SIZE = GGML_QUANT_SIZES[GGMLQuantizationType.Q4_K][1]  # 144
assert Q4_K_TYPE_SIZE == 144


class Q4KRequantNotImplemented(NotImplementedError):
    """Kept for import compat. requantize_q4_k is a real encoder now."""


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
        return np.asarray(dequantize(np.asarray(data), qtype), dtype=np.float32)
    if qtype == GGMLQuantizationType.F64:
        return np.asarray(data, dtype=np.float32)
    if qtype == GGMLQuantizationType.Q4_K:
        return np.asarray(dequantize(np.asarray(data), qtype), dtype=np.float32)
    raise TypeError(f"no dequant path for {qtype.name}")


def _nearest_int(x: np.ndarray) -> np.ndarray:
    """llama.cpp nearest_int: magic-float round (ggml-quants.c)."""
    x32 = np.asarray(x, dtype=np.float32)
    magic = np.float32(12582912.0)  # 1.5 * 2^23
    bits = np.add(x32, magic, dtype=np.float32).view(np.int32)
    return ((bits & np.int32(0x007FFFFF)) - np.int32(0x00400000)).astype(np.int32)


def _make_qkx2_quants(
    x: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of llama.cpp make_qkx2_quants for n=32, nmax=15, Q4_K_ref args.

    rmin=-1, rdelta=0.1, nstep=20, use_mad=false.
    x, weights: (N, 32). Returns L (N, 32) uint8, scale (N,), the_min (N,).
    """
    nmax = 15
    x = np.asarray(x, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    n_rows = x.shape[0]

    xmin = np.minimum(x.min(axis=1), np.float32(0.0))
    xmax = x.max(axis=1)
    sum_w = weights.sum(axis=1)
    sum_x = (weights * x).sum(axis=1)

    L = np.zeros((n_rows, 32), dtype=np.uint8)
    scale = np.zeros(n_rows, dtype=np.float32)
    minv = xmin.copy()

    flat = xmax == xmin
    ok = ~flat
    if not np.any(ok):
        return L, scale, -minv

    span0 = xmax - xmin
    iscale0 = np.zeros(n_rows, dtype=np.float32)
    iscale0[ok] = np.float32(nmax) / span0[ok]
    scale[ok] = np.float32(1.0) / iscale0[ok]

    l0 = np.clip(_nearest_int(iscale0[:, None] * (x - xmin[:, None])), 0, nmax)
    L[ok] = l0[ok].astype(np.uint8)
    diff0 = scale[:, None] * l0.astype(np.float32) + xmin[:, None] - x
    best_error = (weights * diff0 * diff0).sum(axis=1)

    # Sequential search: a better fit updates min for later steps (llama.cpp).
    for step in range(21):
        span = xmax - minv
        valid = ok & (span > 0)
        iscale = np.zeros(n_rows, dtype=np.float32)
        iscale[valid] = (np.float32(-1.0 + 0.1 * step + nmax)) / span[valid]
        l_aux = np.clip(_nearest_int(iscale[:, None] * (x - minv[:, None])), 0, nmax).astype(
            np.float32
        )
        w_l = weights * l_aux
        sum_l = w_l.sum(axis=1)
        sum_l2 = (w_l * l_aux).sum(axis=1)
        sum_xl = (w_l * x).sum(axis=1)
        D = sum_w * sum_l2 - sum_l * sum_l
        good = valid & (D > 0)
        this_scale = np.zeros(n_rows, dtype=np.float32)
        this_min = np.zeros(n_rows, dtype=np.float32)
        this_scale[good] = (sum_w[good] * sum_xl[good] - sum_x[good] * sum_l[good]) / D[good]
        this_min[good] = (sum_l2[good] * sum_x[good] - sum_l[good] * sum_xl[good]) / D[good]
        pos_min = good & (this_min > 0)
        this_min[pos_min] = 0.0
        sl2 = sum_l2[pos_min]
        safe = sl2 != 0
        idx = np.flatnonzero(pos_min)
        this_scale[idx[safe]] = sum_xl[pos_min][safe] / sl2[safe]
        this_scale[idx[~safe]] = 0.0

        diff = this_scale[:, None] * l_aux + this_min[:, None] - x
        cur_error = (weights * diff * diff).sum(axis=1)
        better = good & (cur_error < best_error)
        if np.any(better):
            L[better] = l_aux[better].astype(np.uint8)
            best_error[better] = cur_error[better]
            scale[better] = this_scale[better]
            minv[better] = this_min[better]

    return L, scale, -minv


def _quantize_q4_k_llama_ref(array_f32: np.ndarray) -> np.ndarray:
    """llama.cpp quantize_row_q4_K_ref. array last dim is a multiple of 256."""
    logical = tuple(int(d) for d in array_f32.shape)
    rows = np.ascontiguousarray(array_f32, dtype=np.float32).reshape(-1, QK_K)
    n_blocks = rows.shape[0]
    n_sub = QK_K // 32  # 8

    x_sub = rows.reshape(n_blocks, n_sub, 32)
    # weights[l] = av_x + |x| with av_x = sqrt(mean(x^2)) per 32-wide sub-block
    sum_x2 = np.square(x_sub, dtype=np.float32).sum(axis=-1, keepdims=True)
    av_x = np.sqrt(sum_x2 / np.float32(32.0))
    weights = av_x + np.abs(x_sub)

    L_sub, scales, mins = _make_qkx2_quants(
        x_sub.reshape(n_blocks * n_sub, 32),
        weights.reshape(n_blocks * n_sub, 32),
    )
    L = L_sub.reshape(n_blocks, QK_K)
    scales = scales.reshape(n_blocks, n_sub)
    mins = mins.reshape(n_blocks, n_sub)

    max_scale = scales.max(axis=1)
    max_min = mins.max(axis=1)
    inv_scale = np.divide(np.float32(63.0), max_scale, out=np.zeros_like(max_scale), where=max_scale > 0)
    inv_min = np.divide(np.float32(63.0), max_min, out=np.zeros_like(max_min), where=max_min > 0)
    ls = np.minimum(_nearest_int(inv_scale[:, None] * scales), 63).astype(np.uint8)
    lm = np.minimum(_nearest_int(inv_min[:, None] * mins), 63).astype(np.uint8)

    packed = np.zeros((n_blocks, 12), dtype=np.uint8)
    packed[:, 0:4] = ls[:, 0:4]
    packed[:, 4:8] = lm[:, 0:4]
    packed[:, 8:12] = (ls[:, 4:8] & np.uint8(0x0F)) | ((lm[:, 4:8] & np.uint8(0x0F)) << 4)
    packed[:, 0:4] = packed[:, 0:4] | ((ls[:, 4:8] >> 4) << 6)
    packed[:, 4:8] = packed[:, 4:8] | ((lm[:, 4:8] >> 4) << 6)

    d = (max_scale / np.float32(63.0)).astype(np.float16)
    dmin = (max_min / np.float32(63.0)).astype(np.float16)
    d_f = d.astype(np.float32)
    dmin_f = dmin.astype(np.float32)

    # Unpack 6-bit scales the same way gguf-py / llama.cpp get_scale_min_k4 does.
    sc = np.concatenate(
        [
            packed[:, 0:4] & np.uint8(0x3F),
            (packed[:, 8:12] & np.uint8(0x0F)) | ((packed[:, 0:4] >> 2) & np.uint8(0x30)),
        ],
        axis=1,
    ).astype(np.float32)
    mn = np.concatenate(
        [
            packed[:, 4:8] & np.uint8(0x3F),
            (packed[:, 8:12] >> 4) | ((packed[:, 4:8] >> 2) & np.uint8(0x30)),
        ],
        axis=1,
    ).astype(np.float32)

    d_sub = (d_f[:, None] * sc).reshape(n_blocks, n_sub, 1)
    dm_sub = (dmin_f[:, None] * mn).reshape(n_blocks, n_sub, 1)
    nonzero = d_sub[:, :, 0] != 0
    l_refit = _nearest_int((x_sub + dm_sub) / np.where(d_sub == 0, 1.0, d_sub))
    l_refit = np.clip(l_refit, 0, 15).astype(np.uint8)
    L = L.reshape(n_blocks, n_sub, 32)
    L[nonzero] = l_refit[nonzero]
    L = L.reshape(n_blocks, QK_K)

    groups = L.reshape(n_blocks, 4, 64)
    qs = (groups[:, :, :32] | (groups[:, :, 32:] << 4)).astype(np.uint8).reshape(n_blocks, QK_K // 2)

    d_bytes = d.view(np.uint8).reshape(n_blocks, 2)
    dmin_bytes = dmin.view(np.uint8).reshape(n_blocks, 2)
    blocks = np.concatenate([d_bytes, dmin_bytes, packed, qs], axis=1)
    assert blocks.shape == (n_blocks, Q4_K_TYPE_SIZE)
    byte_shape = quant_shape_to_byte_shape(logical, GGMLQuantizationType.Q4_K)
    return np.ascontiguousarray(blocks.reshape(byte_shape))


def requantize_q4_k(array_f32: np.ndarray) -> np.ndarray:
    """Real Q4_K encoder. Produces ggml superblocks; last dim must be a multiple of 256.

    Tries gguf-py `quantize` first. gguf-py 0.19 has dequant only, so we fall
    through to the llama.cpp `quantize_row_q4_K_ref` port. Do not slice the
    original Q4_K bytes — channels will not land on superblock boundaries.
    """
    array_f32 = np.ascontiguousarray(array_f32, dtype=np.float32)
    if array_f32.ndim < 1:
        raise ValueError("Q4_K requant needs at least a 1-D array")
    if array_f32.shape[-1] % QK_K != 0:
        raise ValueError(
            f"Q4_K requant needs last dim a multiple of {QK_K}, got shape {array_f32.shape}. "
            "FFN-down keep counts must be a multiple of 256 (weak cap 13056 is). "
            "Or pass --q4-k-to-f16 for a host-only F16 dump (serve_ok=false)."
        )
    try:
        from gguf.quants import quantize as gguf_quantize

        out = gguf_quantize(array_f32, GGMLQuantizationType.Q4_K)
        return np.ascontiguousarray(out)
    except NotImplementedError:
        return _quantize_q4_k_llama_ref(array_f32)


def from_f32(
    array_f32: np.ndarray,
    qtype: GGMLQuantizationType,
    *,
    q4_k_to_f16: bool = False,
) -> tuple[np.ndarray, GGMLQuantizationType]:
    """Requantize gathered F32 back to `qtype`.

    Returns (array, output_qtype). Q4_K uses the real encoder unless q4_k_to_f16.
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
        return requantize_q4_k(array_f32), GGMLQuantizationType.Q4_K
    raise TypeError(f"no requant path for {qtype.name}")
