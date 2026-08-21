"""Gather FFN channels and vocab rows. Same index for gate / up / down."""

from __future__ import annotations

import numpy as np

from export.names import FFN_DOWN, FFN_GATE, FFN_UP, parse_blk


def axis_matching(shape: tuple[int, ...], size: int, hint: str) -> int:
    matches = [i for i, d in enumerate(shape) if d == size]
    if not matches:
        raise ValueError(f"{hint}: no axis of size {size} in shape {shape}")
    if len(matches) > 1:
        # Prefer the conventional axis: 0 for (n_ff, n_embd) / (n_vocab, n_embd).
        if matches[0] == 0 or size != shape[-1]:
            return matches[0]
        return matches[0]
    return matches[0]


def gather_axis(arr: np.ndarray, indices: list[int] | np.ndarray, axis: int) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError("gather indices must be 1-D")
    if np.any(idx < 0) or np.any(idx >= arr.shape[axis]):
        raise IndexError(
            f"gather index out of range for axis {axis} size {arr.shape[axis]}: {idx.tolist()}"
        )
    return np.ascontiguousarray(np.take(arr, idx, axis=axis))


def ffn_role(name: str) -> str:
    parsed = parse_blk(name)
    if parsed is None:
        raise ValueError(f"not a block tensor: {name}")
    suffix = parsed[1]
    if suffix.startswith("ffn_gate"):
        return "gate"
    if suffix.startswith("ffn_up"):
        return "up"
    if suffix.startswith("ffn_down"):
        return "down"
    raise ValueError(f"not an FFN tensor: {name}")


def gather_ffn_channels(
    arr: np.ndarray,
    keep_channels: list[int],
    n_ff: int,
    name: str = "",
) -> np.ndarray:
    """Gather the same channel index from gate, up, or down.

    llama.cpp QWEN35 stores (numpy view):
      ffn_gate / ffn_up : (n_ff, n_embd)  -> gather axis 0
      ffn_down          : (n_embd, n_ff)  -> gather axis 1
    We locate the n_ff axis rather than hard-coding, so either layout works.
    """
    axis = axis_matching(arr.shape, n_ff, name or "ffn")
    return gather_axis(arr, keep_channels, axis)


def gather_vocab_rows(
    arr: np.ndarray,
    old_ids_in_row_order: list[int],
    n_vocab: int,
    name: str = "",
) -> np.ndarray:
    """Gather embed / lm_head rows. Output row i is original tokenizer id old_ids[i]."""
    axis = axis_matching(arr.shape, n_vocab, name or "vocab")
    return gather_axis(arr, old_ids_in_row_order, axis)


def expected_ffn_axis(name: str, shape: tuple[int, ...], n_ff: int) -> int:
    """Document the preferred axis; used by tests to assert alignment."""
    role = ffn_role(name)
    axis = axis_matching(shape, n_ff, name)
    if role in ("gate", "up") and len(shape) == 2:
        # Conventional numpy layout after GGUFReader: (n_ff, n_embd)
        if shape[0] == n_ff:
            assert axis == 0
    if role == "down" and len(shape) == 2:
        if shape[1] == n_ff:
            assert axis == 1
    return axis


# Re-export suffixes so tests can name tensors without importing names twice.
FFN_SUFFIXES = (FFN_GATE, FFN_UP, FFN_DOWN)
