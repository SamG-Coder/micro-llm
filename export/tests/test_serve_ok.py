"""micro_llm.serve_ok: false on --q4-k-to-f16 / F16 remnant; true on a real Q4 remnant."""

from __future__ import annotations

from gguf import GGUFReader, GGMLQuantizationType

from export.names import KV_SERVE_OK
from export.writer import pack_gguf, remnant_serve_ok, source_has_q4_k


def test_serve_ok_false_on_q4k_to_f16():
    assert remnant_serve_ok(q4_k_to_f16=True, source_has_q4_k=True) is False
    assert remnant_serve_ok(q4_k_to_f16=True, source_has_q4_k=False) is False


def test_serve_ok_true_on_real_q4_remnant():
    assert remnant_serve_ok(q4_k_to_f16=False, source_has_q4_k=True) is True


def test_serve_ok_false_on_f16_source():
    assert remnant_serve_ok(q4_k_to_f16=False, source_has_q4_k=False) is False


def test_baked_serve_ok_false_on_synthetic_f16(remnant_gguf):
    reader = GGUFReader(str(remnant_gguf))
    assert reader.get_field(KV_SERVE_OK) is not None
    assert reader.get_field(KV_SERVE_OK).contents() is False
    assert source_has_q4_k(reader) is False


def test_baked_serve_ok_true_when_forced(full_gguf, prune_table, tmp_path):
    out = tmp_path / "q4_ok.gguf"
    pack_gguf(full_gguf, prune_table, out, serve_ok=True)
    reader = GGUFReader(str(out))
    assert reader.get_field(KV_SERVE_OK).contents() is True


def test_q4k_to_f16_forces_serve_ok_false(full_gguf, prune_table, tmp_path):
    out = tmp_path / "debug.gguf"
    pack_gguf(full_gguf, prune_table, out, q4_k_to_f16=True)
    reader = GGUFReader(str(out))
    assert reader.get_field(KV_SERVE_OK).contents() is False
    # Host debug path does not invent Q4_K blocks.
    for t in reader.tensors:
        assert t.tensor_type != GGMLQuantizationType.Q4_K
