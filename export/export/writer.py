"""Packed GGUF writer: gather FFN / vocab, drop dead packs, bake KV, 256-align."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from gguf import (
    GGMLQuantizationType,
    GGUFReader,
    GGUFValueType,
    GGUFWriter,
)

from export.gather import gather_ffn_channels, gather_vocab_rows
from export.names import (
    KV_ALIGN,
    KV_KEEP_CH_IDS,
    KV_KEEP_CH_N,
    KV_KEEP_MTP,
    KV_KEEP_PACKS,
    KV_KEEP_VISION,
    KV_PREFIX,
    KV_VOCAB_OLD,
    KV_VOCAB_ROWS,
    KV_VERSION,
    TENSOR_ALIGN,
    is_attention_tensor,
    is_deltanet_mixer,
    is_embed,
    is_ffn_tensor,
    is_lm_head,
    is_mtp_tensor,
    is_shared_norm,
    is_vision_tensor,
    layer_to_pack,
    parse_blk,
)
from export.prune_table import PruneTable, encode_kv
from export.quant import (
    Q4KRequantNotImplemented,
    from_f32,
    is_float_type,
    is_q4_k,
    to_f32,
)

SKIP_SRC_KV = {
    "GGUF.version",
    "GGUF.tensor_count",
    "GGUF.kv_count",
    "general.architecture",
    "general.alignment",
}


def infer_arch(reader: GGUFReader, default: str = "qwen35") -> str:
    field = reader.fields.get("general.architecture")
    if field is None:
        return default
    val = field.contents()
    return str(val) if val else default


def infer_n_layer(reader: GGUFReader, fallback: int) -> int:
    n = 0
    for t in reader.tensors:
        parsed = parse_blk(t.name)
        if parsed is not None:
            n = max(n, parsed[0] + 1)
    return n or fallback


def infer_n_ff(reader: GGUFReader, fallback: int) -> int:
    for t in reader.tensors:
        if is_ffn_tensor(t.name):
            # gate/up numpy (n_ff, n_embd); down (n_embd, n_ff)
            parsed = parse_blk(t.name)
            assert parsed is not None
            suffix = parsed[1]
            shp = tuple(int(x) for x in t.data.shape)
            if suffix.startswith("ffn_down"):
                return shp[-1] if len(shp) >= 2 else fallback
            return shp[0]
    return fallback


def infer_n_vocab(reader: GGUFReader, fallback: int) -> int:
    for t in reader.tensors:
        if is_embed(t.name) or is_lm_head(t.name):
            return int(t.data.shape[0])
    return fallback


def infer_n_embd(reader: GGUFReader, fallback: int = 5120) -> int:
    for t in reader.tensors:
        if is_embed(t.name) and t.data.ndim >= 2:
            return int(t.data.shape[-1])
    return fallback


def _py_value(val: Any) -> Any:
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, list):
        return [_py_value(x) for x in val]
    return val


def copy_source_kv(reader: GGUFReader, writer: GGUFWriter) -> None:
    """Copy source metadata. Tokenizer KV is copied verbatim (no rewrite)."""
    for key, field in reader.fields.items():
        if key in SKIP_SRC_KV or key.startswith(KV_PREFIX):
            continue
        if not field.types:
            continue
        main = field.types[0]
        val = _py_value(field.contents())
        if main == GGUFValueType.ARRAY:
            if not val:
                continue
            sub = field.types[-1] if len(field.types) > 1 else None
            writer.add_key_value(key, val, GGUFValueType.ARRAY, sub_type=sub)
        elif main == GGUFValueType.STRING:
            if val:
                writer.add_key_value(key, val, GGUFValueType.STRING)
        else:
            writer.add_key_value(key, val, main)


def write_prune_kv(writer: GGUFWriter, table: PruneTable) -> None:
    """Bake keep_channels / keep_packs / vocab_remap. Skip empty arrays."""
    payload = encode_kv(table)
    writer.add_uint32(KV_VERSION, int(payload[KV_VERSION]))
    writer.add_uint32(KV_ALIGN, int(payload[KV_ALIGN]))
    writer.add_bool(KV_KEEP_VISION, bool(payload[KV_KEEP_VISION]))
    writer.add_bool(KV_KEEP_MTP, bool(payload[KV_KEEP_MTP]))
    packs = [int(x) for x in payload[KV_KEEP_PACKS]]
    if packs:
        writer.add_array(KV_KEEP_PACKS, packs)
    ch_n = [int(x) for x in payload[KV_KEEP_CH_N]]
    if ch_n:
        writer.add_array(KV_KEEP_CH_N, ch_n)
    ch_ids = [int(x) for x in payload[KV_KEEP_CH_IDS]]
    if ch_ids:
        writer.add_array(KV_KEEP_CH_IDS, ch_ids)
    olds = [int(x) for x in payload[KV_VOCAB_OLD]]
    rows = [int(x) for x in payload[KV_VOCAB_ROWS]]
    if olds:
        writer.add_array(KV_VOCAB_OLD, olds)
    if rows:
        writer.add_array(KV_VOCAB_ROWS, rows)


def _logical_shape(tensor) -> tuple[int, ...]:
    # ReaderTensor.shape is GGUF dim order; numpy data is reversed.
    return tuple(int(x) for x in tensor.data.shape)


def _add_tensor(
    writer: GGUFWriter,
    name: str,
    array: np.ndarray,
    qtype: GGMLQuantizationType,
    logical_shape: tuple[int, ...] | None = None,
) -> None:
    array = np.ascontiguousarray(array)
    if is_float_type(qtype) or qtype in {
        GGMLQuantizationType.I8,
        GGMLQuantizationType.I16,
        GGMLQuantizationType.I32,
        GGMLQuantizationType.I64,
    }:
        writer.add_tensor(name, array)
        return
    shape = logical_shape if logical_shape is not None else tuple(int(x) for x in array.shape)
    writer.add_tensor(name, array, raw_shape=shape, raw_dtype=qtype)


def _gather_or_copy(
    name: str,
    array: np.ndarray,
    qtype: GGMLQuantizationType,
    *,
    gather_indices: list[int] | None,
    axis_size: int,
    q4_k_to_f16: bool,
) -> tuple[np.ndarray, GGMLQuantizationType]:
    if gather_indices is None:
        return np.ascontiguousarray(array), qtype

    if is_q4_k(qtype):
        f32 = to_f32(array, qtype)
        from export.gather import axis_matching, gather_axis

        axis = axis_matching(f32.shape, axis_size, name)
        gathered = gather_axis(f32, gather_indices, axis)
        return from_f32(gathered, qtype, q4_k_to_f16=q4_k_to_f16)

    if not is_float_type(qtype):
        raise TypeError(
            f"{name}: gather on {qtype.name} is not supported in v1 "
            f"(F16/F32 fully supported; Q4_K needs dequant->gather->requant)"
        )

    from export.gather import axis_matching, gather_axis

    axis = axis_matching(array.shape, axis_size, name)
    return gather_axis(array, gather_indices, axis), qtype


def should_omit(name: str, table: PruneTable, n_layer: int) -> bool:
    if is_vision_tensor(name) and not table.keep_vision:
        return True
    if is_mtp_tensor(name, n_layer) and not table.keep_mtp:
        return True
    if is_deltanet_mixer(name):
        parsed = parse_blk(name)
        if parsed is None:
            return False
        pack = layer_to_pack(parsed[0])
        if pack is None:
            return False
        return pack not in set(table.keep_packs)
    if is_ffn_tensor(name):
        parsed = parse_blk(name)
        if parsed is None:
            return False
        layer = parsed[0]
        if layer >= len(table.keep_channels):
            return True
        return len(table.keep_channels[layer]) == 0
    return False


def pack_tensors(
    reader: GGUFReader,
    table: PruneTable,
    *,
    q4_k_to_f16: bool = False,
) -> list[tuple[str, np.ndarray, GGMLQuantizationType]]:
    n_layer = table.n_layer or infer_n_layer(reader, 0)
    n_ff = infer_n_ff(reader, 0)
    n_vocab = infer_n_vocab(reader, 0)
    old_ids = table.old_ids_in_row_order()
    keep_pack_set = set(table.keep_packs)
    out: list[tuple[str, np.ndarray, GGMLQuantizationType, tuple[int, ...] | None]] = []

    for t in reader.tensors:
        name = t.name
        if should_omit(name, table, n_layer):
            continue

        qtype = t.tensor_type
        data = t.data
        gathered: np.ndarray
        out_type = qtype

        if is_ffn_tensor(name):
            parsed = parse_blk(name)
            assert parsed is not None
            layer = parsed[0]
            keep = table.keep_channels[layer]
            if n_ff <= 0:
                raise ValueError(f"{name}: cannot infer n_ff for gather")
            gathered, out_type = _gather_or_copy(
                name,
                data,
                qtype,
                gather_indices=keep,
                axis_size=n_ff,
                q4_k_to_f16=q4_k_to_f16,
            )
        elif is_embed(name) or is_lm_head(name):
            if not old_ids:
                gathered, out_type = np.ascontiguousarray(data), qtype
            else:
                if n_vocab <= 0:
                    raise ValueError(f"{name}: cannot infer n_vocab for gather")
                gathered, out_type = _gather_or_copy(
                    name,
                    data,
                    qtype,
                    gather_indices=old_ids,
                    axis_size=n_vocab,
                    q4_k_to_f16=q4_k_to_f16,
                )
        else:
            # Attention / norms / everything else: copy through, original shape.
            if is_q4_k(qtype):
                gathered = np.ascontiguousarray(data)
                out_type = qtype
            else:
                gathered = np.ascontiguousarray(data)
                out_type = qtype

        new_logical = tuple(int(x) for x in gathered.shape)
        out.append((name, gathered, out_type, new_logical))
    return out


def write_packed_gguf(
    src_path: str,
    out_path: str,
    table: PruneTable,
    *,
    q4_k_to_f16: bool = False,
    arch: str | None = None,
) -> list[str]:
    """Read full GGUF, pack, write remnant. Returns written tensor names."""
    reader = GGUFReader(src_path)
    arch_name = arch or infer_arch(reader)
    packed = pack_tensors(reader, table, q4_k_to_f16=q4_k_to_f16)

    writer = GGUFWriter(out_path, arch_name)
    writer.add_custom_alignment(TENSOR_ALIGN)
    copy_source_kv(reader, writer)
    write_prune_kv(writer, table)

    names: list[str] = []
    for name, array, qtype, logical in packed:
        _add_tensor(writer, name, array, qtype, logical_shape=logical)
        names.append(name)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return names



def infer_dims(reader: GGUFReader):
    from dataclasses import dataclass

    @dataclass
    class ModelDims:
        n_layer: int
        n_embd: int
        n_ff: int
        n_vocab: int
        arch: str

    arch = infer_arch(reader)
    return ModelDims(
        n_layer=infer_n_layer(reader, 0),
        n_embd=infer_n_embd(reader),
        n_ff=infer_n_ff(reader, 0),
        n_vocab=infer_n_vocab(reader, 0),
        arch=arch,
    )


def pack_gguf(model, table: PruneTable, out, q4_k_to_f16: bool = False) -> list[str]:
    return write_packed_gguf(str(model), str(out), table, q4_k_to_f16=q4_k_to_f16)


def list_tensors(path) -> dict:
    reader = GGUFReader(str(path))
    out = {}
    for t in reader.tensors:
        data = np.ascontiguousarray(t.data)
        out[t.name] = {
            "data": data,
            "shape": tuple(int(x) for x in data.shape),
            "type": t.tensor_type,
        }
    return out


def tensor_offsets_aligned(path, align: int = TENSOR_ALIGN):
    reader = GGUFReader(str(path))
    rows = []
    for t in reader.tensors:
        off = int(t.data_offset)
        rows.append((t.name, off, off % align == 0))
    return rows
