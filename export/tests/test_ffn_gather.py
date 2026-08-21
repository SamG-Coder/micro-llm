"""gate/up/down gather uses the same channel index and stays aligned."""

from __future__ import annotations

import numpy as np

from export.gather import ffn_role, gather_ffn_channels
from export.names import FFN_DOWN, FFN_GATE, FFN_UP, blk
from export.writer import list_tensors
from tests.conftest import N_EMBD, N_FF, default_prune_dict


def test_same_channel_index_numpy():
    n_ff, n_embd = 16, 8
    keep = [2, 5, 7]
    gate = np.zeros((n_ff, n_embd), dtype=np.float32)
    up = np.zeros((n_ff, n_embd), dtype=np.float32)
    down = np.zeros((n_embd, n_ff), dtype=np.float32)
    for c in range(n_ff):
        gate[c, :] = c
        up[c, :] = c + 100
        down[:, c] = c + 200
    g = gather_ffn_channels(gate, keep, n_ff, name="blk.0.ffn_gate.weight")
    u = gather_ffn_channels(up, keep, n_ff, name="blk.0.ffn_up.weight")
    d = gather_ffn_channels(down, keep, n_ff, name="blk.0.ffn_down.weight")
    assert g.shape == (3, n_embd)
    assert u.shape == (3, n_embd)
    assert d.shape == (n_embd, 3)
    for i, c in enumerate(keep):
        assert np.all(g[i] == c)
        assert np.all(u[i] == c + 100)
        assert np.all(d[:, i] == c + 200)


def test_ffn_roles():
    assert ffn_role("blk.3.ffn_gate.weight") == "gate"
    assert ffn_role("blk.3.ffn_up.weight") == "up"
    assert ffn_role("blk.3.ffn_down.weight") == "down"


def test_packed_ffn_same_index_and_aligned(remnant_gguf):
    tensors = list_tensors(remnant_gguf)
    raw = default_prune_dict()["keep_channels"]
    for layer, keep in ((int(k), v) for k, v in raw.items()):
        gate = tensors[blk(layer, FFN_GATE)]["data"]
        up = tensors[blk(layer, FFN_UP)]["data"]
        down = tensors[blk(layer, FFN_DOWN)]["data"]
        assert gate.shape == (len(keep), N_EMBD)
        assert up.shape == (len(keep), N_EMBD)
        assert down.shape == (N_EMBD, len(keep))
        for i, c in enumerate(keep):
            assert np.allclose(gate[i].astype(np.float32), np.float32(layer * 64 + c))
            assert np.allclose(up[i].astype(np.float32), np.float32(layer * 64 + 256 + c))
            assert np.allclose(down[:, i].astype(np.float32), np.float32(layer * 64 + 512 + c))
        # channel axes stay aligned: row i is the same original id on all three
        assert gate.shape[0] == up.shape[0] == down.shape[1]


def test_original_n_ff_not_in_packed_ffn(remnant_gguf):
    tensors = list_tensors(remnant_gguf)
    for name, info in tensors.items():
        if name.endswith(".ffn_gate.weight"):
            assert N_FF not in info["shape"]
