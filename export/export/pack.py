"""CLI: full GGUF + prune table -> packed remnant GGUF.

    python -m export.pack --model full.gguf --prune-table dump.mlpt --out remnant.gguf
    python -m export.pack --model full.gguf --prune-table dump.mlpt --out remnant.gguf --ceiling-gb 12
    python -m export.pack --model full.gguf --prune-table keep.json --out remnant.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from export.cut import DEFAULT_CEILING_GB, cut_mlpt
from export.mlpt import is_mlpt_file, read_mlpt
from export.prune_table import PruneTable, load_prune_table
from export.writer import infer_dims, pack_gguf
from gguf import GGUFReader


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m export.pack",
        description="Pack a Qwen hybrid GGUF to a remnant file using an MLPT dump or JSON keep-mask.",
    )
    p.add_argument("--model", required=True, help="Full source GGUF (F16/F32; Q4_K gather needs --q4-k-to-f16)")
    p.add_argument(
        "--prune-table",
        required=True,
        help="MLPT dump (.mlpt / magic MLPT) or JSON keep-mask (.json)",
    )
    p.add_argument("--out", required=True, help="Output packed GGUF")
    p.add_argument(
        "--ceiling-gb",
        type=float,
        default=DEFAULT_CEILING_GB,
        help="Remnant weight ceiling in GiB (MLPT cut only). Default 12. Use 10 for 32k+vision.",
    )
    p.add_argument(
        "--q4-k-to-f16",
        action="store_true",
        help="Dequant Q4_K -> gather -> write F16. Default is to error (requant hook unimplemented).",
    )
    p.add_argument("--keep-vision", action="store_true", help="Keep vision tower (v1 default: omit)")
    p.add_argument("--keep-mtp", action="store_true", help="Keep MTP / nextn heads (v1 default: omit)")
    return p


def resolve_prune_table(
    path: str | Path,
    n_layer: int,
    *,
    ceiling_gb: float = DEFAULT_CEILING_GB,
    keep_vision: bool = False,
    keep_mtp: bool = False,
) -> PruneTable:
    """Load a JSON keep-mask, or read+cut an MLPT dump.

    .json -> existing keep-mask (tests). .mlpt or magic MLPT -> cut then pack.
    """
    table_path = Path(path)
    suffix = table_path.suffix.lower()
    if suffix == ".json":
        return load_prune_table(table_path, n_layer=n_layer)
    if suffix == ".mlpt" or is_mlpt_file(table_path):
        return cut_mlpt(read_mlpt(table_path), ceiling_gb=float(ceiling_gb), keep_vision=keep_vision, keep_mtp=keep_mtp)
    raise ValueError(f"prune table must be a .json keep-mask or MLPT dump: {table_path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model = Path(args.model)
    table_path = Path(args.prune_table)
    out = Path(args.out)
    if not model.is_file():
        print(f"error: model not found: {model}", file=sys.stderr)
        return 2
    if not table_path.is_file():
        print(f"error: prune table not found: {table_path}", file=sys.stderr)
        return 2

    dims = infer_dims(GGUFReader(str(model)))
    table = resolve_prune_table(table_path, n_layer=dims.n_layer, ceiling_gb=args.ceiling_gb, keep_vision=args.keep_vision, keep_mtp=args.keep_mtp)
    table.keep_vision = table.keep_vision or args.keep_vision
    table.keep_mtp = table.keep_mtp or args.keep_mtp
    pack_gguf(model, table, out, q4_k_to_f16=args.q4_k_to_f16)
    print(
        f"wrote {out}  layers={dims.n_layer} embd={dims.n_embd} "
        f"ffn {dims.n_ff}->per-layer keep_channels  "
        f"vocab {dims.n_vocab}->{len(table.vocab_remap) or dims.n_vocab}  "
        f"packs kept={len(table.keep_packs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
