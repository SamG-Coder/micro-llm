"""MLPT parse + cut: floor keep, dead drop, packs 0..47, vocab remap."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from export.cut import cut_mlpt
from export.mlpt import (
    MAGIC,
    MLPT_SIZE,
    N_FFN,
    N_LAYERS,
    N_PACKS,
    VOCAB,
    make_mlpt,
    read_mlpt,
    write_mlpt,
)
from export.names import pack_to_layer


def test_mlpt_roundtrip(tmp_path: Path) -> None:
    dump = make_mlpt(
        n_tokens=1234,
        channels={(3, 17): (9, 2.5, 1.25)},
        packs={7: (4, 0.5)},
        floor=[(3, 17)],
        vocab=[151643],
    )
    path = tmp_path / "dump.mlpt"
    write_mlpt(path, dump)
    assert path.stat().st_size == MLPT_SIZE
    got = read_mlpt(path)
    assert got.header.n_tokens == 1234
    rec = got.channel(3, 17)
    assert rec.n_fired == 9
    assert rec.sumsq == np.float32(2.5)
    assert int(got.packs[47]["pack"]) == 47
    assert int(got.packs[47]["layer"]) == pack_to_layer(47)
    assert got.pack(7).n_spike == 4
    assert got.floor_keep(3, 17)
    assert got.vocab_seen(151643)
    assert not got.vocab_seen(0)


def test_smoke_bin_if_present() -> None:
    smoke = Path("/workspace/micro-llm-harness/build/smoke_prune.bin")
    if not smoke.is_file():
        return
    got = read_mlpt(smoke)
    assert got.header.magic == MAGIC
    assert got.header.n_layers == N_LAYERS == 64
    assert got.header.n_ffn_channels == N_FFN == 17408
    assert got.header.n_packs == N_PACKS == 48
    assert got.header.vocab_size == VOCAB == 248320
    assert got.header.version == 1
    assert got.header.tensor_align == 256
    # known channel row (layer 0, channel 0)
    rec = got.channel(0, 0)
    assert rec.n_fired == 1
    assert rec.sumsq == np.float32(6.982231140136719)
    assert rec.maxabs == np.float32(2.6423912048339844)
    # pack 0..47: pack id equals index, layer = 4*(p//3)+(p%3)
    assert int(got.packs[0]["pack"]) == 0
    assert int(got.packs[0]["layer"]) == 0
    assert got.pack(0).n_spike == 2
    assert int(got.packs[3]["pack"]) == 3
    assert int(got.packs[3]["layer"]) == 4
    assert int(got.packs[47]["pack"]) == 47
    assert int(got.packs[47]["layer"]) == 62
    # floor + vocab bits
    assert got.floor_keep(0, 0)
    assert got.vocab_seen(0)
    assert not got.vocab_seen(16)
    assert got.vocab_seen(17)


def test_floor_keeps_zero_energy_dead_is_cut() -> None:
    dump = make_mlpt(
        channels={(5, 12): (3, 8.0, 1.0)},
        floor=[(5, 10)],
        packs={p: (1, 0.0) for p in range(N_PACKS)},
        vocab=[1],
    )
    table = cut_mlpt(dump, ceiling_gb=1000.0)
    assert 10 in table.keep_channels[5]
    assert 11 not in table.keep_channels[5]
    assert 12 in table.keep_channels[5]


def test_dead_pack_dropped_spiked_kept() -> None:
    dump = make_mlpt(
        channels={(layer, 0): (1, 1.0, 1.0) for layer in range(N_LAYERS)},
        packs={5: (2, 0.1)},
        vocab=[0],
    )
    table = cut_mlpt(dump, ceiling_gb=1000.0)
    assert 5 in table.keep_packs
    assert 6 not in table.keep_packs
    assert 0 in table.keep_packs
    assert 1 in table.keep_packs
    assert 47 in table.keep_packs


def test_vocab_remap_dense() -> None:
    dump = make_mlpt(
        channels={(layer, 0): (1, 1.0, 1.0) for layer in range(N_LAYERS)},
        packs={p: (1, 0.0) for p in range(N_PACKS)},
        vocab=[10, 3, 99],
    )
    table = cut_mlpt(dump, ceiling_gb=1000.0)
    assert table.vocab_remap == {3: 0, 10: 1, 99: 2}
