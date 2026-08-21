"""Dropped DeltaNet packs are absent; kept packs are present."""

from __future__ import annotations

from export.names import (
    ATTN_GATE,
    ATTN_QKV,
    SSM_A,
    SSM_CONV1D,
    SSM_OUT,
    blk,
    layer_to_pack,
    pack_to_layer,
    protected_packs,
)
from export.writer import list_tensors
from tests.conftest import N_LAYER


def test_pack_layer_mapping_global():
    # 8 layers -> packs 0..5. Not 0..2 inside a group.
    assert pack_to_layer(0) == 0
    assert pack_to_layer(1) == 1
    assert pack_to_layer(2) == 2
    assert pack_to_layer(3) == 4
    assert pack_to_layer(4) == 5
    assert pack_to_layer(5) == 6
    assert layer_to_pack(3) is None  # Gated Attention
    assert layer_to_pack(7) is None
    assert layer_to_pack(4) == 3


def test_dropped_packs_absent_kept_present(remnant_gguf, prune_table):
    tensors = list_tensors(remnant_gguf)
    names = set(tensors)

    # Input asked for packs [0, 3]; protection adds first two / last two.
    assert set(prune_table.keep_packs) == set([0, 3]) | set(protected_packs(N_LAYER))
    # protected: layers 0,1,6 -> packs 0,1,5. Plus requested 3.
    assert set(prune_table.keep_packs) == {0, 1, 3, 5}

    dropped = {2, 4}  # layers 2 and 5
    kept = set(prune_table.keep_packs)

    for p in dropped:
        layer = pack_to_layer(p)
        for suffix in (ATTN_QKV, ATTN_GATE, SSM_CONV1D, SSM_A, SSM_OUT):
            assert blk(layer, suffix) not in names, f"dropped pack {p} still has {suffix}"

    for p in kept:
        layer = pack_to_layer(p)
        assert blk(layer, ATTN_QKV) in names
        assert blk(layer, SSM_CONV1D) in names
        # mixer data is a real copy, not a zero-fill placeholder of a new name
        assert tensors[blk(layer, ATTN_QKV)]["data"].size > 0


def test_dropped_pack_not_zero_filled(remnant_gguf):
    names = set(list_tensors(remnant_gguf))
    assert blk(2, ATTN_QKV) not in names
    assert blk(5, ATTN_QKV) not in names
    # FFN of those layers still exists (width-cut), layer was not deleted
    assert blk(2, "ffn_gate.weight") in names
    assert blk(5, "ffn_gate.weight") in names
