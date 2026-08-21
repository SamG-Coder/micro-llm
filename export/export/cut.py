"""Collapse an MLPT score dump into a keep-mask. C++ dumps; Export cuts.

Ranking rule (weak FFN channels)
--------------------------------
energy(layer, channel) = sumsq
If sumsq == 0 and n_fired > 0, energy = n_fired * 1e-12
  (fired but numerically-zero accumulator still ranks above true zeros).

Channels are compared by (energy, n_fired, maxabs, layer, channel) so the
order is deterministic. Lowest energy among non-floor survivors is cut first.

Floor bits are never ranked for drop. Dead (n_fired == 0 AND not floor) is
always dropped in phase 1, before any weak cut.

Cut order:
  1. dead FFN          (n_fired == 0 AND not floor)
  2. weak FFN          (lowest energy among remaining non-floor)  [ceiling]
  3. dead DeltaNet     (n_spike == 0), except protected layer packs
  4. unused vocab      (bitset 0)

Never drop layers 0, 1, n-2, n-1 as *layers* (their DeltaNet packs stay
even if n_spike == 0). FFN on those layers still width-cuts.
Never drop the 16 Gated Attention blocks (QKVO + KV); they are not in
this table and copy through on pack.
"""

from __future__ import annotations

import numpy as np

from export.mlpt import MlptDump
from export.names import TENSOR_ALIGN
from export.prune_table import PruneTable

RANKING_RULE = (
    "energy = sumsq; if sumsq == 0 and n_fired > 0 then "
    "energy = n_fired * 1e-12. Weak cut = lowest energy among "
    "non-floor survivors, ties broken by (n_fired, maxabs, layer, channel)."
)

# Q4_K ~ 0.5 byte/param. 12GB is the 8-16k remnant weight ceiling.
BYTES_PER_PARAM_Q4 = 0.5
HIDDEN_27B = 5120
FIRED_ZERO_EPS = 1e-12
DEFAULT_CEILING_GB = 12.0


def channel_energy(sumsq: float, n_fired: int) -> float:
    e = float(sumsq)
    if e == 0.0 and int(n_fired) > 0:
        return float(n_fired) * FIRED_ZERO_EPS
    return e


def estimate_weight_bytes(
    keep_channels: list[list[int]],
    keep_packs: list[int],
    n_vocab_kept: int,
    hidden: int = HIDDEN_27B,
    n_gated: int = 16,
) -> int:
    """Rough Q4 weight bytes. FFN dominates; used to stop cutting at the ceiling.

    Per kept FFN channel: gate + up + down = 3 * hidden params.
    Per kept DeltaNet pack: treat as ~4 * hidden^2 (in/out + state).
    Per Gated Attention block: ~4 * hidden^2 (QKVO).
    Vocab: untied embed + lm_head = 2 * n_kept * hidden.
    """
    ffn = sum(len(chs) * 3 * hidden for chs in keep_channels)
    packs = len(keep_packs) * 4 * hidden * hidden
    attn = n_gated * 4 * hidden * hidden
    vocab = 2 * n_vocab_kept * hidden
    return int((ffn + packs + attn + vocab) * BYTES_PER_PARAM_Q4)


def estimate_weight_gb(
    n_ffn_kept: int,
    n_packs_kept: int,
    n_vocab_kept: int,
    hidden: int = HIDDEN_27B,
    n_gated: int = 16,
) -> float:
    """GiB for n kept FFN channels / packs / vocab rows (Q4-ish 0.5 B/param)."""
    ffn = int(n_ffn_kept) * 3 * hidden
    packs = int(n_packs_kept) * 4 * hidden * hidden
    attn = int(n_gated) * 4 * hidden * hidden
    vocab = 2 * int(n_vocab_kept) * hidden
    return (ffn + packs + attn + vocab) * BYTES_PER_PARAM_Q4 / float(1024 ** 3)


def _protected_layers(n_layers: int) -> set[int]:
    return {0, 1, n_layers - 2, n_layers - 1}


def cut_mlpt(
    dump: MlptDump,
    ceiling_gb: float = DEFAULT_CEILING_GB,
    hidden: int = HIDDEN_27B,
    keep_vision: bool = False,
    keep_mtp: bool = False,
) -> PruneTable:
    """Dead FFN first, then weak (lowest sumsq), then dead packs, then unused vocab.

    Floor bit forces keep. Layers 0,1,n-2,n-1 are never dropped as layers.
    Gated Attention blocks are not in this table and are never dropped.
    Ranking for weak channels is sumsq of |SiLU(gate)*up| (see RANKING_RULE).
    """
    n_layers, n_ffn = dump.n_layers, dump.n_ffn
    protected = _protected_layers(n_layers)

    keep_channels: list[list[int]] = []
    weak: list[tuple[float, int, float, int, int]] = []
    for layer in range(n_layers):
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
        if not kept:
            kept = [0]
        keep_channels.append(kept)

    keep_packs: list[int] = []
    for p in range(dump.n_packs):
        layer = int(dump.pack_layer[p])
        if int(dump.n_spike[p]) > 0 or layer in protected:
            keep_packs.append(p)

    old_ids = [int(i) for i in np.flatnonzero(np.asarray(dump.vocab, dtype=bool))]
    vocab_remap = {oid: row for row, oid in enumerate(old_ids)}

    ceiling = int(ceiling_gb * (1024**3))
    weak.sort()  # lowest energy first
    for _energy, _fired, _mx, layer, ch in weak:
        if estimate_weight_bytes(keep_channels, keep_packs, len(vocab_remap), hidden=hidden) <= ceiling:
            break
        row = keep_channels[layer]
        if ch in row and len(row) > 1:
            row.remove(ch)

    return PruneTable(
        keep_channels=[sorted(set(chs)) for chs in keep_channels],
        keep_packs=sorted(set(keep_packs)),
        vocab_remap=vocab_remap,
        keep_vision=keep_vision,
        keep_mtp=keep_mtp,
        n_layer=n_layers,
    )


# used by pack.py
cut = cut_mlpt
_ = TENSOR_ALIGN
