"""MLPT scores -> keep-mask: floor, dead FFN, packs, vocab remap, ceiling."""

from __future__ import annotations

import pytest

from export.cut import (
    BYTES_PER_PARAM_Q4KM,
    DEFAULT_CEILING_GB,
    N_PARAMS_27B_BILLION,
    WEAK_KEEP_MIN_27B,
    WEAK_KEEP_MIN_RECOVER_27B,
    CutCeilingError,
    bytes_per_param_from_model,
    count_gguf_params,
    cut_mlpt,
    estimate_weight_bytes,
    estimate_weight_gb,
    layer_missing_hook,
    weak_keep_min,
)
from export.mlpt import N_FFN, N_LAYERS, N_PACKS, VOCAB, make_mlpt
from export.names import (
    is_gated_attention_layer,
    pack_to_layer,
    protected_layers,
    protected_packs,
)

# Sparse dumps keep 17408 on missing-hook layers. Use a huge ceiling so
# those tests are not about the remnant budget.
WIDE = 1000.0


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
    table = cut_mlpt(dump, ceiling_gb=WIDE)
    assert 10 in table.keep_channels[5]
    assert 11 not in table.keep_channels[5]
    assert 12 in table.keep_channels[5]


def test_missing_hook_keeps_all_channels_never_kept_zero():
    dump = make_mlpt(vocab=[0])
    assert all(layer_missing_hook(dump, layer) for layer in range(N_LAYERS))
    table = cut_mlpt(dump, ceiling_gb=WIDE)
    for layer in range(N_LAYERS):
        kept = table.keep_channels[layer]
        assert kept == list(range(N_FFN)), f"layer {layer} not full width"
        assert kept != [0]
        assert len(kept) == 17408


def test_protected_layers_still_width_cut_not_dropped():
    dump = make_mlpt(
        channels={(0, 3): (0, 0.0, 0.0), (0, 4): (2, 1.0, 1.0)},
        vocab=[0],
    )
    table = cut_mlpt(dump, ceiling_gb=WIDE)
    assert set(protected_layers(N_LAYERS)) == {0, 1, 62, 63}
    for layer in (0, 1, 62, 63):
        assert table.keep_channels[layer], f"protected layer {layer} missing"
    assert 3 not in table.keep_channels[0]
    assert 4 in table.keep_channels[0]


def test_gated_attention_layers_still_have_ffn_rows():
    dump = make_mlpt(channels={(3, 1): (4, 2.0, 1.0)}, vocab=[0])
    table = cut_mlpt(dump, ceiling_gb=WIDE)
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
    table = cut_mlpt(dump, ceiling_gb=WIDE)
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
    table = cut_mlpt(dump, ceiling_gb=WIDE)
    assert table.vocab_remap == {3: 0, 7: 1, 100: 2, 248319: 3}
    assert table.old_ids_in_row_order() == [3, 7, 100, 248319]
    assert table.rows_in_row_order() == [0, 1, 2, 3]


def _all_live_dump(*, weak_layer: int = 9) -> object:
    dump = make_mlpt(vocab=[0])
    dump.n_fired[:, :] = 1
    dump.sumsq[:, :] = 10.0
    dump.maxabs[:, :] = 1.0
    dump.n_spike[:] = 1
    dump.sumsq[weak_layer, 0] = 0.001
    dump.sumsq[weak_layer, 1] = 0.002
    dump.floor[weak_layer, 2] = True
    dump.sumsq[weak_layer, 2] = 0.0
    dump.n_fired[weak_layer, 2] = 0
    return dump


def test_weak_cut_drops_lowest_sumsq_first():
    dump = _all_live_dump()
    wide = cut_mlpt(dump, ceiling_gb=WIDE)
    assert set(wide.keep_channels[9]) >= {0, 1, 2}

    est = estimate_weight_bytes(
        wide.keep_channels, wide.keep_packs, len(wide.vocab_remap) or VOCAB
    )
    one_ch = 3 * 5120 * BYTES_PER_PARAM_Q4KM
    # Drop the two weakest non-floor channels on layer 9 (0 then 1).
    ceiling_gb = (est - 2.5 * one_ch) / float(1024**3)
    tight = cut_mlpt(dump, ceiling_gb=ceiling_gb)
    assert 2 in tight.keep_channels[9]  # floor
    assert 0 not in tight.keep_channels[9]
    assert 1 not in tight.keep_channels[9]


def test_weak_cut_never_drops_floor():
    dump = _all_live_dump()
    est = estimate_weight_bytes(
        [list(range(N_FFN)) for _ in range(N_LAYERS)],
        list(range(N_PACKS)),
        1,
    )
    one_ch = 3 * 5120 * BYTES_PER_PARAM_Q4KM
    table = cut_mlpt(dump, ceiling_gb=(est - 3 * one_ch) / float(1024**3))
    assert 2 in table.keep_channels[9]


def test_weak_cap_25_then_ceiling_error():
    dump = _all_live_dump()
    assert weak_keep_min(N_FFN, recover=False) == WEAK_KEEP_MIN_27B == 13056
    with pytest.raises(CutCeilingError) as ei:
        cut_mlpt(dump, ceiling_gb=0.0, recover=False)
    table = ei.value.table
    assert table is not None
    for layer, chs in enumerate(table.keep_channels):
        assert len(chs) >= WEAK_KEEP_MIN_27B, f"layer {layer} hollowed to {len(chs)}"
        assert chs != [0]
        assert len(chs) > 1
    # Floor on layer 9 channel 2 cannot be weak-cut, so that layer keeps more.
    assert 2 in table.keep_channels[9]
    assert len(table.keep_channels[9]) == WEAK_KEEP_MIN_27B


def test_recover_allows_40_percent_then_ceiling_error():
    dump = _all_live_dump()
    assert weak_keep_min(N_FFN, recover=True) == WEAK_KEEP_MIN_RECOVER_27B == 10496
    with pytest.raises(CutCeilingError) as ei:
        cut_mlpt(dump, ceiling_gb=0.0, recover=True)
    table = ei.value.table
    assert table is not None
    assert ei.value.recover is True
    for layer, chs in enumerate(table.keep_channels):
        assert len(chs) >= WEAK_KEEP_MIN_RECOVER_27B
        assert len(chs) > 1
        assert chs != [0]


def test_cut_order_packs_and_vocab_before_weak():
    """Dead packs + unused vocab are free; they can avoid a weak FFN cut."""
    dump = make_mlpt(vocab=[0, 1])
    dump.n_fired[:, :] = 1
    dump.sumsq[:, :] = 10.0
    dump.maxabs[:, :] = 1.0
    # Only pack 6 spiked; protected 0,1,47 stay. Rest dead.
    dump.n_spike[:] = 0
    dump.n_spike[6] = 3
    dump.sumsq[9, 0] = 0.001

    after_free = estimate_weight_bytes(
        [list(range(N_FFN)) for _ in range(N_LAYERS)],
        sorted({0, 1, 6, 47}),
        2,
    )
    if_all_packs = estimate_weight_bytes(
        [list(range(N_FFN)) for _ in range(N_LAYERS)],
        list(range(N_PACKS)),
        VOCAB,
    )
    assert after_free < if_all_packs
    # Ceiling sits between "after free cuts" and "still holding dead packs/vocab".
    ceiling_gb = (after_free + if_all_packs) / 2.0 / float(1024**3)
    table = cut_mlpt(dump, ceiling_gb=ceiling_gb)
    assert 6 in table.keep_packs
    assert 5 not in table.keep_packs
    assert table.vocab_remap == {0: 0, 1: 1}
    # Weak not needed: channel 0 on layer 9 stays.
    assert 0 in table.keep_channels[9]
    assert len(table.keep_channels[9]) == N_FFN


def test_estimator_ffn_term():
    one = estimate_weight_gb(1, 0, 0)
    two = estimate_weight_gb(2, 0, 0)
    delta = (two - one) * (1024**3)
    assert delta == pytest.approx(3 * 5120 * BYTES_PER_PARAM_Q4KM)
    assert DEFAULT_CEILING_GB == 12.0
    assert BYTES_PER_PARAM_Q4KM == pytest.approx(17.1 / 28)
    assert BYTES_PER_PARAM_Q4KM == pytest.approx(17.1 / N_PARAMS_27B_BILLION)
    assert BYTES_PER_PARAM_Q4KM == pytest.approx(0.61, abs=0.02)


def test_bytes_per_param_from_model(tmp_path, full_gguf):
    from gguf import GGUFReader

    reader = GGUFReader(str(full_gguf))
    n = count_gguf_params(reader)
    assert n > 0
    bpp = bytes_per_param_from_model(str(full_gguf), n_params=n)
    size = full_gguf.stat().st_size
    assert bpp == pytest.approx(size / n)
    assert bpp != pytest.approx(0.5)
