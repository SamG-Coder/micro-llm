"""Real Q4_K requant: llama.cpp ref encoder, dequant→gather→requant, serve_ok."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter, dequantize

from export.gather import gather_ffn_channels
from export.names import (
    FFN_DOWN,
    FFN_GATE,
    FFN_UP,
    KV_SERVE_OK,
    TENSOR_ALIGN,
    TOKEN_EMBD,
    blk,
)
from export.prune_table import parse_prune_table
from export.quant import QK_K, from_f32, requantize_q4_k, to_f32
from export.writer import pack_gguf, remnant_serve_ok, tensor_offsets_aligned

# Q4_K (4-bit + per-32 min/scale) typically lands ~5-10% relative RMSE on
# Gaussian weights. A broken encoder (zeros, identity slice, random bytes)
# misses this window by a lot.
Q4K_REL_RMSE_MAX = 0.15
Q4K_REL_RMSE_MIN = 0.01


def test_requantize_q4_k_roundtrip_error_bounds():
    rng = np.random.default_rng(1)
    src = rng.normal(0.0, 0.05, size=(6, 256)).astype(np.float32)
    q = requantize_q4_k(src)
    assert q.dtype == np.uint8
    assert q.shape == (6, 144)
    back = np.asarray(dequantize(q, GGMLQuantizationType.Q4_K), dtype=np.float32)
    assert back.shape == src.shape
    err = back - src
    rel = float(np.sqrt(np.mean(err**2)) / np.sqrt(np.mean(src**2)))
    assert Q4K_REL_RMSE_MIN < rel < Q4K_REL_RMSE_MAX
    assert float(np.max(np.abs(err))) < 0.05


def test_requantize_q4_k_zeros_stay_zero():
    src = np.zeros((2, 512), dtype=np.float32)
    q = requantize_q4_k(src)
    back = np.asarray(dequantize(q, GGMLQuantizationType.Q4_K), dtype=np.float32)
    assert back.shape == src.shape
    assert np.max(np.abs(back)) == 0.0


def test_requantize_rejects_unaligned_last_dim():
    with pytest.raises(ValueError, match="multiple of 256"):
        requantize_q4_k(np.zeros((2, 255), dtype=np.float32))


def test_from_f32_q4k_default_stays_q4k():
    src = np.linspace(-0.2, 0.2, 256, dtype=np.float32).reshape(1, 256)
    out, qtype = from_f32(src, GGMLQuantizationType.Q4_K, q4_k_to_f16=False)
    assert qtype == GGMLQuantizationType.Q4_K
    assert out.dtype == np.uint8
    assert out.shape[-1] == 144


def test_from_f32_q4k_to_f16_is_host_debug():
    src = np.linspace(-0.2, 0.2, 256, dtype=np.float32).reshape(1, 256)
    out, qtype = from_f32(src, GGMLQuantizationType.Q4_K, q4_k_to_f16=True)
    assert qtype == GGMLQuantizationType.F16
    assert out.dtype == np.float16
    assert out.shape == src.shape


def test_q4k_dequant_gather_requant():
    """Gather FFN channels on a Q4_K tensor via dequant -> gather -> requant."""
    rng = np.random.default_rng(2)
    n_ff, n_embd = 512, 256
    keep = list(range(0, 512, 2))  # 256 rows; last dim of gate stays n_embd=256
    gate = rng.normal(0, 0.04, size=(n_ff, n_embd)).astype(np.float32)
    q_gate = requantize_q4_k(gate)
    f32 = to_f32(q_gate, GGMLQuantizationType.Q4_K)
    gathered = gather_ffn_channels(f32, keep, n_ff, name="blk.0.ffn_gate.weight")
    requant, qtype = from_f32(gathered, GGMLQuantizationType.Q4_K)
    assert qtype == GGMLQuantizationType.Q4_K
    back = np.asarray(dequantize(requant, GGMLQuantizationType.Q4_K), dtype=np.float32)
    assert back.shape == (256, n_embd)
    # Must match dequant of the kept rows, within a second Q4_K pass.
    target = f32[keep]
    err = back - target
    rel = float(np.sqrt(np.mean(err**2)) / max(float(np.sqrt(np.mean(target**2))), 1e-8))
    assert rel < Q4K_REL_RMSE_MAX
    # Not an in-place superblock slice of the original bytes (even rows ≠ prefix).
    assert requant.tobytes() != q_gate[:256].tobytes()


def _write_q4k_ffn_gguf(path: Path) -> tuple[Path, list[int]]:
    """Tiny Q4_K hybrid: 4 layers, n_embd=256, n_ff=512. Keep 256 channels."""
    n_layer, n_embd, n_ff, n_vocab = 4, 256, 512, 256
    keep = list(range(256))
    rng = np.random.default_rng(3)

    w = GGUFWriter(str(path), "qwen35")
    w.add_name("q4k-hybrid")
    w.add_block_count(n_layer)
    w.add_embedding_length(n_embd)
    w.add_feed_forward_length(n_ff)
    w.add_vocab_size(n_vocab)
    w.add_custom_alignment(TENSOR_ALIGN)
    w.add_token_list([f"t{i:03d}" for i in range(n_vocab)])

    emb = rng.normal(0, 0.03, size=(n_vocab, n_embd)).astype(np.float32)
    # Byte-shaped uint8 + raw_dtype: GGUFWriter converts to logical dims.
    w.add_tensor(TOKEN_EMBD, requantize_q4_k(emb), raw_dtype=GGMLQuantizationType.Q4_K)

    for i in range(n_layer):
        gate = rng.normal(0, 0.03, size=(n_ff, n_embd)).astype(np.float32)
        up = rng.normal(0, 0.03, size=(n_ff, n_embd)).astype(np.float32)
        down = rng.normal(0, 0.03, size=(n_embd, n_ff)).astype(np.float32)
        w.add_tensor(blk(i, FFN_GATE), requantize_q4_k(gate), raw_dtype=GGMLQuantizationType.Q4_K)
        w.add_tensor(blk(i, FFN_UP), requantize_q4_k(up), raw_dtype=GGMLQuantizationType.Q4_K)
        w.add_tensor(blk(i, FFN_DOWN), requantize_q4_k(down), raw_dtype=GGMLQuantizationType.Q4_K)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path, keep


def _q4k_prune(n_layer: int = 4, keep=None):
    keep = keep or list(range(256))
    return parse_prune_table(
        {
            "keep_channels": {str(i): keep for i in range(n_layer)},
            "keep_packs": [0, 1, 2],
            "vocab_remap": {str(i): i for i in range(256)},
            "keep_vision": False,
            "keep_mtp": False,
        },
        n_layer=n_layer,
    )


def test_packed_q4k_remnant_is_q4k_aligned_serve_ok(tmp_path: Path):
    src, keep = _write_q4k_ffn_gguf(tmp_path / "full_q4.gguf")
    table = _q4k_prune(keep=keep)
    out = tmp_path / "remnant_q4.gguf"
    pack_gguf(src, table, out)

    reader = GGUFReader(str(out))
    assert reader.get_field(KV_SERVE_OK).contents() is True
    assert remnant_serve_ok(q4_k_to_f16=False, source_has_q4_k=True) is True

    gathered = 0
    for t in reader.tensors:
        assert t.tensor_type == GGMLQuantizationType.Q4_K
        if t.name.endswith(".ffn_gate.weight"):
            gathered += 1
            # logical last dim is n_embd=256; numpy data is byte-shaped
            logical = tuple(int(x) for x in reversed(list(t.shape)))
            assert logical[0] == 256
            assert logical[-1] % QK_K == 0
    assert gathered == 4

    rows = tensor_offsets_aligned(out, TENSOR_ALIGN)
    assert rows and all(ok for _n, _off, ok in rows)


def test_q4k_to_f16_pack_sets_serve_ok_false(tmp_path: Path):
    src, keep = _write_q4k_ffn_gguf(tmp_path / "full_q4.gguf")
    table = _q4k_prune(keep=keep)
    out = tmp_path / "debug_f16.gguf"
    pack_gguf(src, table, out, q4_k_to_f16=True)
    reader = GGUFReader(str(out))
    assert reader.get_field(KV_SERVE_OK).contents() is False
    # Gathered FFN/vocab became F16; do not load this on the 5080.
    types = {t.name: t.tensor_type for t in reader.tensors}
    assert types[blk(0, FFN_GATE)] == GGMLQuantizationType.F16
    assert types[TOKEN_EMBD] == GGMLQuantizationType.F16
