"""Attention tensors copy through with original shapes."""

from __future__ import annotations

import numpy as np

from export.names import (
    ATTN_K,
    ATTN_K_NORM,
    ATTN_OUT,
    ATTN_Q,
    ATTN_Q_NORM,
    ATTN_V,
    blk,
    is_gated_attention_layer,
)
from export.writer import list_tensors
from tests.conftest import N_LAYER


ATTN_SUFFIXES = (ATTN_Q, ATTN_K, ATTN_V, ATTN_OUT, ATTN_Q_NORM, ATTN_K_NORM)


def test_attention_shapes_and_values(full_gguf, remnant_gguf):
    src = list_tensors(full_gguf)
    dst = list_tensors(remnant_gguf)
    ga_layers = [i for i in range(N_LAYER) if is_gated_attention_layer(i)]
    assert ga_layers == [3, 7]

    for i in ga_layers:
        for suffix in ATTN_SUFFIXES:
            name = blk(i, suffix)
            assert name in dst, f"missing attention tensor {name}"
            assert dst[name]["shape"] == src[name]["shape"]
            assert np.array_equal(dst[name]["data"], src[name]["data"])


def test_vision_and_mtp_omitted(remnant_gguf):
    names = set(list_tensors(remnant_gguf))
    assert "v.patch_embd.weight" not in names
    assert "blk.8.nextn.eh_proj.weight" not in names
    assert "blk.8.ffn_gate.weight" not in names
