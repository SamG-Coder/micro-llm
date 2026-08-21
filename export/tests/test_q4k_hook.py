"""Q4_K requant is a real hook, not a fake in-place slice. v1 skips requant."""

from __future__ import annotations

import numpy as np
import pytest

from export.quant import Q4KRequantNotImplemented, requantize_q4_k
from gguf import GGMLQuantizationType


def test_requantize_q4_k_hook_raises():
    arr = np.zeros((2, 256), dtype=np.float32)
    with pytest.raises(Q4KRequantNotImplemented):
        requantize_q4_k(arr)


@pytest.mark.skip(reason="Q4_K dequant-gather-requant not implemented in v1 (gguf-py has no Q4_K.quantize_blocks)")
def test_q4k_dequant_gather_requant():
    """When requantize_q4_k is wired, gather FFN channels on a Q4_K tensor
    via dequant -> gather -> requant (never slice superblocks in place).
    """
    _ = GGMLQuantizationType.Q4_K
    raise AssertionError("unskip only after a real Q4_K encoder is hooked")
