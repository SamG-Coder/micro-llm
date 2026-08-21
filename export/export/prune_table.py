"""Load a prune-table JSON and encode/decode the GGUF KV block."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from export.names import (
    KV_ALIGN,
    KV_KEEP_CH_IDS,
    KV_KEEP_CH_N,
    KV_KEEP_MTP,
    KV_KEEP_PACKS,
    KV_KEEP_VISION,
    KV_PREFIX,
    KV_SERVE_OK,
    KV_VOCAB_OLD,
    KV_VOCAB_ROWS,
    KV_VERSION,
    TENSOR_ALIGN,
    n_packs_for_layers,
    pack_to_layer,
    protected_packs,
)


@dataclass
class PruneTable:
    """Effective keep-mask. This is what Export gathers and what gets baked."""

    keep_channels: list[list[int]]
    keep_packs: list[int]
    vocab_remap: dict[int, int]  # old tokenizer id -> dense remnant row
    keep_vision: bool = False
    keep_mtp: bool = False
    n_layer: int = 0
    serve_ok: bool = False  # C++ serve refuses unless present and true

    def old_ids_in_row_order(self) -> list[int]:
        if not self.vocab_remap:
            return []
        by_row = sorted(self.vocab_remap.items(), key=lambda kv: kv[1])
        return [old for old, _ in by_row]

    def rows_in_row_order(self) -> list[int]:
        return [self.vocab_remap[old] for old in self.old_ids_in_row_order()]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "keep_channels": {str(i): chs for i, chs in enumerate(self.keep_channels)},
            "keep_packs": list(self.keep_packs),
            "vocab_remap": {str(k): v for k, v in sorted(self.vocab_remap.items())},
            "keep_vision": self.keep_vision,
            "keep_mtp": self.keep_mtp,
            "serve_ok": self.serve_ok,
        }


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        value = json.loads(value)
    return [int(x) for x in value]


def _parse_keep_channels(raw: Any, n_layer: int) -> list[list[int]]:
    out: list[list[int]] = [[] for _ in range(n_layer)]
    if raw is None:
        return out
    if isinstance(raw, dict):
        for k, v in raw.items():
            layer = int(k)
            if layer < 0:
                raise ValueError(f"keep_channels layer {layer} is negative")
            if layer >= n_layer:
                # MTP / extra: ignore here; those layers are dropped unless keep_mtp
                continue
            out[layer] = sorted(set(int(c) for c in v))
        return out
    if isinstance(raw, list):
        for i, v in enumerate(raw):
            if i >= n_layer:
                break
            out[i] = sorted(set(int(c) for c in (v or [])))
        return out
    raise TypeError(f"keep_channels must be a dict or list, got {type(raw)}")


def _parse_vocab_remap(raw: Any) -> dict[int, int]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        if "old_ids" in raw and "rows" in raw:
            olds = _as_int_list(raw["old_ids"])
            rows = _as_int_list(raw["rows"])
            if len(olds) != len(rows):
                raise ValueError("vocab_remap old_ids and rows length mismatch")
            return {int(o): int(r) for o, r in zip(olds, rows)}
        return {int(k): int(v) for k, v in raw.items()}
    if isinstance(raw, list):
        remap: dict[int, int] = {}
        for item in raw:
            if isinstance(item, dict):
                remap[int(item["old_id"])] = int(item["row"])
            else:
                old, row = item
                remap[int(old)] = int(row)
        return remap
    raise TypeError(f"vocab_remap must be a dict or list, got {type(raw)}")


def _validate_vocab_remap(remap: dict[int, int]) -> dict[int, int]:
    if not remap:
        return {}
    rows = sorted(remap.values())
    if len(set(rows)) != len(rows):
        raise ValueError("vocab_remap rows must be unique")
    if rows[0] != 0 or rows[-1] != len(rows) - 1:
        raise ValueError(
            "vocab_remap rows must be a dense 0..N-1 range "
            f"(got min={rows[0]} max={rows[-1]} n={len(rows)})"
        )
    return dict(remap)


def load_prune_table(path: str | Path, n_layer: int) -> PruneTable:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return parse_prune_table(raw, n_layer=n_layer)


def parse_prune_table(raw: Mapping[str, Any], n_layer: int) -> PruneTable:
    if n_layer <= 0:
        raise ValueError("n_layer must be positive")

    keep_channels = _parse_keep_channels(raw.get("keep_channels"), n_layer)
    keep_packs = sorted(set(_as_int_list(raw.get("keep_packs"))))
    n_packs = n_packs_for_layers(n_layer)
    for p in keep_packs:
        if p < 0 or p >= n_packs:
            raise ValueError(f"keep_packs id {p} out of range 0..{n_packs - 1}")
        if pack_to_layer(p) >= n_layer:
            raise ValueError(f"keep_packs id {p} maps to layer >= n_layer={n_layer}")

    # First two / last two layers are never dropped as layers.
    forced = protected_packs(n_layer)
    keep_packs = sorted(set(keep_packs) | set(forced))

    vocab_remap = _validate_vocab_remap(_parse_vocab_remap(raw.get("vocab_remap")))
    keep_vision = bool(raw.get("keep_vision", False))
    keep_mtp = bool(raw.get("keep_mtp", False))

    return PruneTable(
        keep_channels=keep_channels,
        keep_packs=keep_packs,
        vocab_remap=vocab_remap,
        keep_vision=keep_vision,
        keep_mtp=keep_mtp,
        n_layer=n_layer,
        serve_ok=bool(raw.get("serve_ok", False)),
    )


def encode_kv(table: PruneTable) -> dict[str, Any]:
    """Flat KV payload. Arrays are Python lists of ints (UINT32 on write)."""
    ch_n = [len(chs) for chs in table.keep_channels]
    ch_ids: list[int] = []
    for chs in table.keep_channels:
        ch_ids.extend(int(c) for c in chs)
    return {
        KV_VERSION: 1,
        KV_ALIGN: TENSOR_ALIGN,
        KV_KEEP_PACKS: [int(p) for p in table.keep_packs],
        KV_KEEP_CH_N: ch_n,
        KV_KEEP_CH_IDS: ch_ids,
        KV_VOCAB_OLD: table.old_ids_in_row_order(),
        KV_VOCAB_ROWS: table.rows_in_row_order(),
        KV_KEEP_VISION: table.keep_vision,
        KV_KEEP_MTP: table.keep_mtp,
        KV_SERVE_OK: bool(table.serve_ok),
    }


def decode_kv(fields: Mapping[str, Any]) -> PruneTable:
    """Rebuild a PruneTable from GGUF field contents (name -> value)."""

    def _get(key: str, default: Any = None) -> Any:
        return fields.get(key, default)

    ch_n = _as_int_list(_get(KV_KEEP_CH_N, []))
    ch_ids = _as_int_list(_get(KV_KEEP_CH_IDS, []))
    keep_channels: list[list[int]] = []
    cursor = 0
    for n in ch_n:
        keep_channels.append([int(x) for x in ch_ids[cursor : cursor + n]])
        cursor += n
    if cursor != len(ch_ids):
        raise ValueError("keep_channels.ids length does not match keep_channels.n")

    olds = _as_int_list(_get(KV_VOCAB_OLD, []))
    rows = _as_int_list(_get(KV_VOCAB_ROWS, []))
    if len(olds) != len(rows):
        raise ValueError("vocab_remap old_ids/rows length mismatch")
    vocab_remap = {int(o): int(r) for o, r in zip(olds, rows)}

    return PruneTable(
        keep_channels=keep_channels,
        keep_packs=sorted(set(_as_int_list(_get(KV_KEEP_PACKS, [])))),
        vocab_remap=vocab_remap,
        keep_vision=bool(_get(KV_KEEP_VISION, False)),
        keep_mtp=bool(_get(KV_KEEP_MTP, False)),
        n_layer=len(keep_channels),
        serve_ok=bool(_get(KV_SERVE_OK, False)),
    )


def fields_from_reader(reader: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, field in reader.fields.items():
        if key.startswith(KV_PREFIX):
            out[key] = field.contents()
    return out
