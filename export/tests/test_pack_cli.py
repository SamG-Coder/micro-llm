"""CLI: JSON keep-mask still packs; .mlpt is detected and cut."""

from __future__ import annotations

from pathlib import Path

from export.mlpt import write_mlpt
from export.names import protected_packs
from export.pack import main, resolve_prune_table
from tests.conftest import N_LAYER


def test_cli_json_keep_mask(full_gguf, prune_path, tmp_path):
    out = tmp_path / "from_json.gguf"
    rc = main(["--model", str(full_gguf), "--prune-table", str(prune_path), "--out", str(out)])
    assert rc == 0
    assert out.is_file()


def test_resolve_json_vs_mlpt(tmp_path, prune_path):
    json_table = resolve_prune_table(prune_path, n_layer=N_LAYER)
    assert json_table.n_layer == N_LAYER

    mlpt_path = tmp_path / "dump.mlpt"
    write_mlpt(
        mlpt_path,
        channels={(4, 9): (2, 1.0, 0.5)},
        packs={6: (2, 0.3)},
        vocab=[3, 7],
        floor=[(4, 1)],
    )
    # magic MLPT even if we also check suffix
    table = resolve_prune_table(mlpt_path, n_layer=64, ceiling_gb=12)
    assert table.n_layer == 64
    assert 9 in table.keep_channels[4]
    assert 1 in table.keep_channels[4]  # floor
    assert 6 in table.keep_packs
    for p in protected_packs(64):
        assert p in table.keep_packs
    assert table.vocab_remap == {3: 0, 7: 1}

    # no suffix, detect by magic
    bare = tmp_path / "dump.bin"
    bare.write_bytes(mlpt_path.read_bytes())
    by_magic = resolve_prune_table(bare, n_layer=64, ceiling_gb=12)
    assert by_magic.vocab_remap == table.vocab_remap
