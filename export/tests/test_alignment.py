"""Packed tensor data is 256-byte aligned."""

from __future__ import annotations

from export.names import TENSOR_ALIGN
from export.writer import tensor_offsets_aligned


def test_every_tensor_offset_256(remnant_gguf):
    rows = tensor_offsets_aligned(remnant_gguf, TENSOR_ALIGN)
    assert rows, "expected tensors in remnant"
    bad = [(name, off) for name, off, ok in rows if not ok]
    assert bad == [], f"misaligned tensors: {bad}"
    for _name, off, ok in rows:
        assert off % 256 == 0
        assert ok
