"""Pack stream-writes one tensor at a time (no add_tensor buffer of the remnant)."""

from __future__ import annotations

from gguf import GGUFWriter

from export.writer import pack_gguf


def test_pack_uses_tensor_info_not_add_tensor(full_gguf, prune_table, tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError(
            "add_tensor buffers the remnant; stream via add_tensor_info + write_tensor_data"
        )

    monkeypatch.setattr(GGUFWriter, "add_tensor", boom)
    out = tmp_path / "stream.gguf"
    names = pack_gguf(full_gguf, prune_table, out)
    assert names
    assert out.is_file()


def test_write_tensor_data_once_per_tensor(full_gguf, prune_table, tmp_path, monkeypatch):
    calls = []
    orig = GGUFWriter.write_tensor_data

    def wrapped(self, tensor, *a, **k):
        calls.append(int(tensor.nbytes))
        return orig(self, tensor, *a, **k)

    monkeypatch.setattr(GGUFWriter, "write_tensor_data", wrapped)
    out = tmp_path / "stream2.gguf"
    names = pack_gguf(full_gguf, prune_table, out)
    assert len(calls) == len(names)
    assert all(n > 0 for n in calls)
