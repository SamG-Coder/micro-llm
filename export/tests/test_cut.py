"""MLPT scores -> keep-mask: floor, dead FFN, packs, vocab remap, ceiling."""

from __future__ import annotations

import pytest

from export.cut import (
    DEFAULT_CEILING_GB,
    cut_mlpt,
    estimate_weight_gb,
)
from export.mlpt import N_LAYERS, N_PACKS, make_mlpt
from export.names import (
    is_gated_attention_layer,
    pack_to_layer,
    protected_layers,
    protected_packs,
)


def test_floor_keeps_zero_energy_dead_nonfloor_cut():
    dump = make_mlpt(
        channels={
            (5, 10): (0, 0.0, 0.0),  # dead + floor -> keep
            (5, 11): (0, 0.0, 0.0),  # dead, no floor -> cut
            (5, 12): (5, 4.0, 1.0),  # live -> keep
        },
        floor=[(5, 10)],
        packs={0: (1, 1.0)},
        vocab=[1],
    )
    table = cut_mlpt(dump, ceiling_gb=12)
    assert 10 in table.keep_channels[5]
    assert 11 not in table.keep_channels[5]
    assert 12 in table.keep_channels[5]


def test_empty_layer_keeps_at_least_one_channel():
    dump = make_mlpt(vocab=[0])
    table = cut_mlpt(dump, ceiling_gb=12)
    for layer in range(N_LAYERS):
        assert len(table.keep_channels[layer]) >= 1, f"layer {layer} emptied"


def test_protected_layers_still_width_cut_not_dropped():
    dump = make_mlpt(
        channels={(0, 3): (0, 0.0, 0.0), (0, 4): (2, 1.0, 1.0)},
        vocab=[0],
    )
    table = cut_mlpt(dump, ceiling_gb=12)
    assert set(protected_layers(N_LAYERS)) == {0, 1, 62, 63}
    for layer in (0, 1, 62, 63):
        assert table.keep_channels[layer], f"protected layer {layer} missing"
    assert 3 not in table.keep_channels[0]
    assert 4 in table.keep_channels[0]


def test_gated_attention_layers_still_have_ffn_rows():
    dump = make_mlpt(channels={(3, 1): (4, 2.0, 1.0)}, vocab=[0])
    table = cut_mlpt(dump, ceiling_gb=12)
    ga = [i for i in range(N_LAYERS) if is_gated_attention_layer(i)]
    assert ga[0] == 3 and ga[-1] == 63
    assert len(table.keep_channels) == 64
    assert 1 in table.keep_channels[3]


def test_packs_global_dead_dropped_spiked_kept():
    dump = make_mlpt(
        packs={
            5: (0, 0.0),   # dead, not protected -> drop
            6: (3, 1.25),  # spiked -> keep
            0: (0, 0.0),   # dead but protected (layer 0)
            1: (0, 0.0),   # dead but protected (layer 1)
            47: (0, 0.0),  # dead but protected (layer 62)
        },
        vocab=[0],
    )
    table = cut_mlpt(dump, ceiling_gb=12)
    assert pack_to_layer(0) == 0
    assert pack_to_layer(47) == 62
    assert set(range(N_PACKS)) == set(range(48))
    assert 5 not in table.keep_packs
    assert 6 in table.keep_packs
    assert set(protected_packs(N_LAYERS)) == {0, 1, 47}
    for p in protected_packs(N_LAYERS):
        assert p in table.keep_packs
    assert 47 in table.keep_packs


def test_vocab_remap_dense_increasing_old_id():
    dump = make_mlpt(vocab=[100, 3, 7, 248319], packs={2: (1, 0.1)})
    table = cut_mlpt(dump, ceiling_gb=12)
    assert table.vocab_remap == {3: 0, 7: 1, 100: 2, 248319: 3}
    assert table.old_ids_in_row_order() == [3, 7, 100, 248319]
    assert table.rows_in_row_order() == [0, 1, 2, 3]


def test_ceiling_does_not_cut_floor_or_empty_layer():
    dump = make_mlpt(
        channels={
            (8, 1): (0, 0.0, 0.0),  # floor zero-energy
            (8, 2): (3, 0.01, 0.1),  # weak live
            (8, 3): (3, 9.0, 2.0),  # strong live
        },
        floor=[(8, 1)],
        packs={2: (1, 1.0)},
        vocab=[0, 1, 2],
    )
    table = cut_mlpt(dump, ceiling_gb=0.0)
    assert 1 in table.keep_channels[8]
    assert 2 not in table.keep_channels[8]
    assert 3 not in table.keep_channels[8]
    for layer in range(N_LAYERS):
        assert table.keep_channels[layer], f"layer {layer} emptied under 0 GiB ceiling"


def test_weak_cut_drops_lowest_sumsq_first():
    dump = make_mlpt(
        channels={
            (9, 0): (1, 0.001, 0.1),
            (9, 1): (1, 0.002, 0.1),
            (9, 2): (1, 5.0, 1.0),
        },
        floor=[(9, 2)],
        packs={2: (1, 1.0)},
        vocab=[0],
    )
    # With a huge default ceiling nothing extra is cut.
    wide = cut_mlpt(dump, ceiling_gb=DEFAULT_CEILING_GB)
    assert set(wide.keep_channels[9]) >= {0, 1, 2}

    tight = cut_mlpt(dump, ceiling_gb=0.0)
    assert 2 in tight.keep_channels[9]
    assert 0 not in tight.keep_channels[9]
    assert 1 not in tight.keep_channels[9]


def test_estimator_ffn_term():
    # 3 mats * hidden * kept * 0.5 bytes
    one = estimate_weight_gb(1, 0, 0)
    two = estimate_weight_gb(2, 0, 0)
    delta = (two - one) * (1024 ** 3)
    assert delta == pytest.approx(3 * 5120 * 0.5)
    assert DEFAULT_CEILING_GB == 12.0

