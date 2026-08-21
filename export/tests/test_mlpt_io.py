"""MLPT write/read roundtrip and smoke-header parse."""

from __future__ import annotations

from pathlib import Path

import pytest

from export.mlpt import (
    FLAG_HAS_FLOOR,
    HEADER_SIZE,
    MAGIC,
    N_FFN_CHANNELS,
    N_LAYERS,
    N_PACKS,
    TOTAL_BYTES,
    VOCAB_SIZE,
    make_mlpt,
    read_mlpt,
    read_mlpt_header,
    write_mlpt,
)
from export.names import pack_to_layer

SMOKE = Path("/workspace/micro-llm-harness/build/smoke_prune.bin")


def test_write_read_roundtrip(tmp_path: Path):
    dump = make_mlpt(
        fire_eps=2.5e-5,
        spike_eps=3.5e-5,
        n_tokens=123,
        flags=FLAG_HAS_FLOOR,
        channels={
            (0, 0): (9, 4.25, 1.5),
            (63, 17407): (2, 0.25, 0.5),
            (16, 100): (1, 8.0, 2.0),
        },
        packs={
            0: (4, 12.5),
            47: (1, 0.01),
        },
        floor=[(3, 7), (63, 17407)],
        vocab=[0, 17, 248319],
    )
    path = tmp_path / "fixture.mlpt"
    write_mlpt(path, dump)
    assert path.stat().st_size == TOTAL_BYTES

    back = read_mlpt(path)
    assert back.header.magic == MAGIC
    assert back.header.version == 1
    assert back.header.n_layers == N_LAYERS
    assert back.header.n_ffn_channels == N_FFN_CHANNELS
    assert back.header.n_packs == N_PACKS
    assert back.header.vocab_size == VOCAB_SIZE
    assert back.header.tensor_align == 256
    assert back.header.header_size == HEADER_SIZE
    assert back.header.n_tokens == 123
    assert back.header.flags & FLAG_HAS_FLOOR
    assert back.header.fire_eps == pytest.approx(2.5e-5, rel=1e-6, abs=1e-8)
    assert back.header.spike_eps == pytest.approx(3.5e-5, rel=1e-6, abs=1e-8)

    assert back.channel(0, 0).n_fired == 9
    assert back.channel(0, 0).sumsq == pytest.approx(4.25)
    assert back.channel(0, 0).maxabs == pytest.approx(1.5)
    assert back.channel(63, 17407).n_fired == 2
    assert back.channel(63, 17407).maxabs == pytest.approx(0.5)
    assert back.channel(16, 100).sumsq == pytest.approx(8.0)

    assert back.pack(0).n_spike == 4
    assert back.pack(0).sumsq_residual == pytest.approx(12.5)
    assert back.pack(47).pack == 47
    assert back.pack(47).layer == pack_to_layer(47) == 62
    assert back.pack(47).n_spike == 1

    for p in range(N_PACKS):
        rec = back.pack(p)
        assert rec.pack == p
        assert rec.layer == pack_to_layer(p)

    assert back.floor_keep(3, 7)
    assert back.floor_keep(63, 17407)
    assert not back.floor_keep(0, 1)
    assert back.vocab_seen(0)
    assert back.vocab_seen(17)
    assert back.vocab_seen(248319)
    assert not back.vocab_seen(18)


def test_write_mlpt_kwargs_fixture(tmp_path: Path):
    path = tmp_path / "kw.mlpt"
    write_mlpt(path, n_tokens=7, channels={(1, 2): (3, 1.0, 0.5)}, vocab=[5])
    back = read_mlpt(path)
    assert back.header.n_tokens == 7
    assert back.channel(1, 2).n_fired == 3
    assert back.vocab_seen(5)


def test_bad_magic_rejected(tmp_path: Path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"XXXX" + b"\x00" * 76)
    with pytest.raises(ValueError, match="magic"):
        read_mlpt_header(path)


@pytest.mark.skipif(not SMOKE.is_file(), reason="smoke_prune.bin not present")
def test_smoke_prune_header_magic_and_dims():
    header = read_mlpt_header(SMOKE)
    assert header.magic == MAGIC
    assert header.version == 1
    assert header.n_layers == 64
    assert header.n_ffn_channels == 17408
    assert header.n_packs == 48
    assert header.vocab_size == 248320
    assert header.tensor_align == 256
    assert header.header_size == 80
    assert SMOKE.stat().st_size == TOTAL_BYTES
