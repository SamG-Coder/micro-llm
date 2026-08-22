"""Collapse an MLPT score dump into a keep-mask. C++ dumps; Export cuts.

Ranking rule (weak FFN channels)
--------------------------------
energy(layer, channel) = sumsq
If sumsq == 0 and n_fired > 0, energy = n_fired * 1e-12
  (fired but numerically-zero accumulator still ranks above true zeros).

Channels are compared by (energy, n_fired, maxabs, layer, channel) so the
order is deterministic. Lowest energy among non-floor survivors is cut first.

Floor bits are never ranked for drop. Dead (n_fired == 0 AND not floor) is
always dropped in phase 1, before packs / vocab / any weak cut.

Cut order (free cuts first, then ceiling):
  1. dead FFN          (n_fired == 0 AND not floor)
  2. dead DeltaNet     (n_spike == 0), except protected layer packs
  3. unused vocab      (bitset 0)
  4. weak FFN          (lowest energy among remaining non-floor)  [ceiling]

Missing hook: if a layer's total n_fired is 0 and that layer has no floor
bits, keep ALL n_ffn channels (17408 on 27B). Never emit kept=[0]. The
hooks did not run; "everything dead" is not evidence.

Weak cap: at most 25% of the layer width (keep >= 13056 of 17408).
recover=True allows 40% (keep >= 10496). Never hollow a layer to 1
channel. If the remnant is still over the ceiling after the max weak cut,
raise CutCeilingError.

Never drop layers 0, 1, n-2, n-1 as *layers* (their DeltaNet packs stay
even if n_spike == 0). FFN on those layers still width-cuts.
Never drop the 16 Gated Attention blocks (QKVO + KV); they are not in
this table and copy through on pack.

bytes/param defaults to 17.1/28 ≈ 0.61 (Q4_K_M 17.1GB / ~28B params).
When --model is given, use source_file_size / n_params.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from export.mlpt import N_FFN, MlptDump
from export.names import TENSOR_ALIGN
from export.prune_table import PruneTable

RANKING_RULE = (
    "energy = sumsq; if sumsq == 0 and n_fired > 0 then "
    "energy = n_fired * 1e-12. Weak cut = lowest energy among "
    "non-floor survivors, ties broken by (n_fired, maxabs, layer, channel)."
)

# Full Q4_K_M ~17.1GB / ~28B params. 0.5 was the old Q4-ish guess and
# under-counted the remnant. Prefer source_file_size / n_params when a
# --model path is available.
FULL_Q4KM_GB = 17.1
N_PARAMS_27B_BILLION = 28.0
BYTES_PER_PARAM_Q4KM = FULL_Q4KM_GB / N_PARAMS_27B_BILLION  # ≈ 0.610714
BYTES_PER_PARAM_Q4 = BYTES_PER_PARAM_Q4KM  # back-compat alias
HIDDEN_27B = 5120
FIRED_ZERO_EPS = 1e-12
DEFAULT_CEILING_GB = 12.0

# Weak FFN width-cut. 25% clean; 40% if we recover. Never hollow to 1.
WEAK_CUT_FRAC = 0.25
WEAK_CUT_FRAC_RECOVER = 0.40
WEAK_KEEP_MIN_27B = 13_056          # ceil(17408 * 0.75)
WEAK_KEEP_MIN_RECOVER_27B = 10_496  # 41*256; Q4_K superblocks line up


class CutCeilingError(RuntimeError):
    """Remnant still over the weight ceiling after free cuts + max weak FFN."""

    def __init__(
        self,
        message: str,
        *,
        table: PruneTable | None = None,
        estimate_bytes: int = 0,
        ceiling_bytes: int = 0,
        recover: bool = False,
    ) -> None:
        super().__init__(message)
        self.table = table
        self.estimate_bytes = int(estimate_bytes)
        self.ceiling_bytes = int(ceiling_bytes)
        self.recover = bool(recover)


def channel_energy(sumsq: float, n_fired: int) -> float:
    e = float(sumsq)
    if e == 0.0 and int(n_fired) > 0:
        return float(n_fired) * FIRED_ZERO_EPS
    return e


def weak_keep_min(n_ffn: int = N_FFN, *, recover: bool = False) -> int:
    """Minimum channels to keep per layer during the weak-cut phase."""
    if int(n_ffn) == N_FFN:
        return WEAK_KEEP_MIN_RECOVER_27B if recover else WEAK_KEEP_MIN_27B
    frac = WEAK_CUT_FRAC_RECOVER if recover else WEAK_CUT_FRAC
    return max(2, math.ceil(int(n_ffn) * (1.0 - frac)))


def layer_missing_hook(dump: MlptDump, layer: int) -> bool:
    """True when the FFN hook never ran (n_fired stayed 0, no floor).

    A streamed FFN with n_fired=0 is a missing hook, not a prune — keep
    all 17408 channels. Do not rank or width-cut that layer.
    """
    return int(np.asarray(dump.n_fired[layer]).sum()) == 0 and not bool(
        np.asarray(dump.floor[layer]).any()
    )


def estimate_weight_bytes(
    keep_channels: Sequence[Sequence[int]],
    keep_packs: Sequence[int],
    n_vocab_kept: int,
    hidden: int = HIDDEN_27B,
    n_gated: int = 16,
    bytes_per_param: float = BYTES_PER_PARAM_Q4KM,
) -> int:
    """Rough remnant weight bytes. FFN dominates; used to stop cutting at the ceiling.

    Per kept FFN channel: gate + up + down = 3 * hidden params.
    Per kept DeltaNet pack: treat as ~4 * hidden^2 (in/out + state).
    Per Gated Attention block: ~4 * hidden^2 (QKVO).
    Vocab: untied embed + lm_head = 2 * n_kept * hidden.
    """
    ffn = sum(len(chs) * 3 * hidden for chs in keep_channels)
    packs = len(keep_packs) * 4 * hidden * hidden
    attn = n_gated * 4 * hidden * hidden
    vocab = 2 * n_vocab_kept * hidden
    return int((ffn + packs + attn + vocab) * float(bytes_per_param))


def estimate_weight_gb(
    n_ffn_kept: int,
    n_packs_kept: int,
    n_vocab_kept: int,
    hidden: int = HIDDEN_27B,
    n_gated: int = 16,
    bytes_per_param: float = BYTES_PER_PARAM_Q4KM,
) -> float:
    """GiB for n kept FFN channels / packs / vocab rows."""
    ffn = int(n_ffn_kept) * 3 * hidden
    packs = int(n_packs_kept) * 4 * hidden * hidden
    attn = int(n_gated) * 4 * hidden * hidden
    vocab = 2 * int(n_vocab_kept) * hidden
    return (ffn + packs + attn + vocab) * float(bytes_per_param) / float(1024**3)


def bytes_per_param_from_model(
    model_path: str,
    n_params: int | None = None,
) -> float:
    """source_file_size / n_params. Falls back to 17.1/28 if n_params is 0."""
    from pathlib import Path

    size = int(Path(model_path).stat().st_size)
    n = int(n_params or 0)
    if n <= 0:
        return BYTES_PER_PARAM_Q4KM
    return float(size) / float(n)


def count_gguf_params(reader) -> int:
    """Prefer general.parameter_count; else sum logical tensor elements."""
    field = getattr(reader, "fields", {}).get("general.parameter_count")
    if field is not None:
        val = field.contents()
        if val:
            return int(val)
    n = 0
    for t in reader.tensors:
        n += int(t.n_elements)
    return n


def _protected_layers(n_layers: int) -> set[int]:
    return {0, 1, n_layers - 2, n_layers - 1}


def _n_vocab_for_estimate(vocab_remap: dict[int, int], vocab_size: int) -> int:
    # Empty remap means "keep the full tokenizer table" (missing vocab bits).
    return len(vocab_remap) if vocab_remap else int(vocab_size)


def cut_mlpt(
    dump: MlptDump,
    ceiling_gb: float = DEFAULT_CEILING_GB,
    hidden: int = HIDDEN_27B,
    keep_vision: bool = False,
    keep_mtp: bool = False,
    *,
    recover: bool = False,
    bytes_per_param: float | None = None,
) -> PruneTable:
    """Dead FFN, then dead packs, then unused vocab, then weak FFN.

    Floor bit forces keep. Layers 0,1,n-2,n-1 are never dropped as layers.
    Gated Attention blocks are not in this table and are never dropped.
    Ranking for weak channels is sumsq of |SiLU(gate)*up| (see RANKING_RULE).
    """
    n_layers, n_ffn = dump.n_layers, dump.n_ffn
    protected = _protected_layers(n_layers)
    bpp = float(BYTES_PER_PARAM_Q4KM if bytes_per_param is None else bytes_per_param)
    min_keep = weak_keep_min(n_ffn, recover=recover)

    keep_channels: list[list[int]] = []
    weak: list[tuple[float, int, float, int, int]] = []
    for layer in range(n_layers):
        if layer_missing_hook(dump, layer):
            # Missing hook (incl. streamed FFN n_fired=0): keep all 17408.
            keep_channels.append(list(range(n_ffn)))
            continue
        kept: list[int] = []
        for ch in range(n_ffn):
            floor = bool(dump.floor[layer, ch])
            n_fired = int(dump.n_fired[layer, ch])
            if floor or n_fired > 0:
                kept.append(ch)
                if not floor:
                    energy = channel_energy(float(dump.sumsq[layer, ch]), n_fired)
                    weak.append(
                        (energy, n_fired, float(dump.maxabs[layer, ch]), layer, ch)
                    )
        # Scored layer with evidence: empty kept cannot happen (floor or fire).
        # Still refuse the old kept=[0] hollow.
        if not kept:
            kept = list(range(n_ffn))
        keep_channels.append(kept)

    keep_packs: list[int] = []
    for p in range(dump.n_packs):
        layer = int(dump.pack_layer[p])
        if int(dump.n_spike[p]) > 0 or layer in protected:
            keep_packs.append(p)

    old_ids = [int(i) for i in np.flatnonzero(np.asarray(dump.vocab, dtype=bool))]
    vocab_remap = {oid: row for row, oid in enumerate(old_ids)}
    n_vocab_est = _n_vocab_for_estimate(vocab_remap, dump.vocab_size)

    ceiling = int(float(ceiling_gb) * (1024**3))

    def _est() -> int:
        return estimate_weight_bytes(
            keep_channels,
            keep_packs,
            n_vocab_est,
            hidden=hidden,
            bytes_per_param=bpp,
        )

    # Weak FFN last, only if free cuts left us over the ceiling.
    weak.sort()  # lowest energy first
    for _energy, _fired, _mx, layer, ch in weak:
        if _est() <= ceiling:
            break
        row = keep_channels[layer]
        if ch not in row:
            continue
        if len(row) <= min_keep:
            continue
        if len(row) <= 1:
            # Never hollow to 1 channel.
            continue
        row.remove(ch)

    table = PruneTable(
        keep_channels=[sorted(set(chs)) for chs in keep_channels],
        keep_packs=sorted(set(keep_packs)),
        vocab_remap=vocab_remap,
        keep_vision=keep_vision,
        keep_mtp=keep_mtp,
        n_layer=n_layers,
    )

    est = estimate_weight_bytes(
        table.keep_channels,
        table.keep_packs,
        n_vocab_est,
        hidden=hidden,
        bytes_per_param=bpp,
    )
    if est > ceiling:
        cap = "40%" if recover else "25%"
        raise CutCeilingError(
            f"remnant estimate {est / (1024**3):.3f} GiB exceeds ceiling "
            f"{float(ceiling_gb):.3f} GiB after dead FFN, dead packs, unused "
            f"vocab, and max weak FFN cut ({cap}; keep >= {min_keep} of {n_ffn}). "
            f"Refuse to hollow a layer to 1 channel.",
            table=table,
            estimate_bytes=est,
            ceiling_bytes=ceiling,
            recover=recover,
        )
    return table


# used by pack.py
cut = cut_mlpt
_ = TENSOR_ALIGN
