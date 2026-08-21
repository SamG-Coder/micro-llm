"""Packed GGUF writer: gather FFN / vocab, drop dead packs, bake KV, 256-align.

Stream-writes one tensor at a time (add_tensor_info for the header, then
gather + write_tensor_data + drop). Do not buffer the whole remnant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from gguf import (
    GGMLQuantizationType,
    GGUFReader,
    GGUFValueType,
    GGUFWriter,
)

from export.cut import HIDDEN_27B, estimate_weight_bytes
from export.gather import axis_matching, gather_axis
from export.names import (
    KV_ALIGN,
    KV_CUDA_SCRATCH_BYTES,
    KV_KEEP_CH_IDS,
    KV_KEEP_CH_N,
    KV_KEEP_MTP,
    KV_KEEP_PACKS,
    KV_KEEP_VISION,
    KV_PER_TOKEN_FP16,
    KV_PER_TOKEN_FP8,
    KV_PREFIX,
    KV_SERVE_OK,
    KV_SERVE_USABLE_BYTES,
    KV_VOCAB_OLD,
    KV_VOCAB_ROWS,
    KV_VERSION,
    KV_WEIGHT_BYTES,
    TENSOR_ALIGN,
    is_deltanet_mixer,
    is_embed,
    is_ffn_tensor,
    is_lm_head,
    is_mtp_tensor,
    is_vision_tensor,
    layer_to_pack,
    parse_blk,
)
from export.prune_table import PruneTable, encode_kv
from export.serve_budget import (
    CUDA_SCRATCH_BYTES,
    KV_BYTES_PER_TOKEN_FP16,
    KV_BYTES_PER_TOKEN_FP8,
    SERVE_USABLE_HEADLESS_BYTES,
)
from export.quant import (
    Q4_K_TYPE_SIZE,
    QK_K,
    from_f32,
    is_float_type,
    is_q4_k,
    to_f32,
)
from gguf.quants import quant_shape_to_byte_shape

SKIP_SRC_KV = {
    "GGUF.version",
    "GGUF.tensor_count",
    "GGUF.kv_count",
    "general.architecture",
    "general.alignment",
}


def remnant_serve_ok(*, q4_k_to_f16: bool, source_has_q4_k: bool) -> bool:
    """C++ serve refuses unless micro_llm.serve_ok is present and true.

    --q4-k-to-f16 is host debug (mixed F16 remnant) -> false.
    A real Q4 remnant (source has Q4_K and we did not take the debug path) -> true.
    """
    if q4_k_to_f16:
        return False
    return bool(source_has_q4_k)


def source_has_q4_k(reader: GGUFReader) -> bool:
    return any(is_q4_k(t.tensor_type) for t in reader.tensors)


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
            shp = _logical_numpy_shape(t)
            if suffix.startswith("ffn_down"):
                return shp[-1] if len(shp) >= 2 else fallback
            return shp[0]
    return fallback


def infer_n_vocab(reader: GGUFReader, fallback: int) -> int:
    for t in reader.tensors:
        if is_embed(t.name) or is_lm_head(t.name):
            return int(_logical_numpy_shape(t)[0])
    return fallback


def infer_n_embd(reader: GGUFReader, fallback: int = 5120) -> int:
    for t in reader.tensors:
        if is_embed(t.name):
            shp = _logical_numpy_shape(t)
            if len(shp) >= 2:
                return int(shp[-1])
    return fallback


def _py_value(val: Any) -> Any:
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, list):
        return [_py_value(x) for x in val]
    return val


def _logical_numpy_shape(tensor) -> tuple[int, ...]:
    # ReaderTensor.shape is GGUF dim order; numpy data is reversed.
    return tuple(int(x) for x in reversed(list(tensor.shape)))


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


def estimate_baked_weight_bytes(
    table: PruneTable,
    *,
    hidden: int = HIDDEN_27B,
    n_vocab: int = 0,
) -> int:
    """cut.estimate_weight_bytes for the KV block. Serve prefers on-disk file size."""
    n_kept = len(table.old_ids_in_row_order()) or int(n_vocab)
    return estimate_weight_bytes(
        table.keep_channels,
        table.keep_packs,
        n_kept,
        hidden=int(hidden),
    )


def patch_uint64_kv(path: str, key: str, value: int) -> bool:
    """Overwrite a UINT64 KV in place after stream-write. Not a second tensor rewrite."""
    reader = GGUFReader(path, "r+")
    try:
        field = reader.get_field(key)
        if field is None or not field.data:
            return False
        part_index = int(field.data[0])
        field.parts[part_index][0] = np.uint64(value)
        reader.data.flush()
        return True
    finally:
        data = getattr(reader, "data", None)
        if data is not None:
            del reader.data


def write_prune_kv(
    writer: GGUFWriter,
    table: PruneTable,
    *,
    serve_ok: bool | None = None,
    weight_bytes: int | None = None,
) -> None:
    """Bake keep_channels / keep_packs / vocab_remap / serve_ok / 5080 serve stack.

    GGUF KV is written before tensors, so weight_bytes here is
    cut.estimate_weight_bytes. After stream-write, write_packed_gguf patches
    micro_llm.weight_bytes to the on-disk file size in place. The serve path
    prefers that file size over this KV estimate.
    """
    if serve_ok is not None:
        table.serve_ok = bool(serve_ok)
    payload = encode_kv(table)
    writer.add_uint32(KV_VERSION, int(payload[KV_VERSION]))
    writer.add_uint32(KV_ALIGN, int(payload[KV_ALIGN]))
    writer.add_bool(KV_KEEP_VISION, bool(payload[KV_KEEP_VISION]))
    writer.add_bool(KV_KEEP_MTP, bool(payload[KV_KEEP_MTP]))
    # Always present. C++ serve refuses unless this is true.
    writer.add_bool(KV_SERVE_OK, bool(payload[KV_SERVE_OK]))
    writer.add_uint64(KV_CUDA_SCRATCH_BYTES, int(CUDA_SCRATCH_BYTES))
    writer.add_uint64(KV_PER_TOKEN_FP16, int(KV_BYTES_PER_TOKEN_FP16))
    writer.add_uint64(KV_PER_TOKEN_FP8, int(KV_BYTES_PER_TOKEN_FP8))
    writer.add_uint64(KV_SERVE_USABLE_BYTES, int(SERVE_USABLE_HEADLESS_BYTES))
    if weight_bytes is None:
        weight_bytes = estimate_baked_weight_bytes(table)
    writer.add_uint64(KV_WEIGHT_BYTES, int(weight_bytes))
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
        axis = axis_matching(f32.shape, axis_size, name)
        gathered = gather_axis(f32, gather_indices, axis)
        return from_f32(gathered, qtype, q4_k_to_f16=q4_k_to_f16)

    if not is_float_type(qtype):
        raise TypeError(
            f"{name}: gather on {qtype.name} is not supported in v1 "
            f"(F16/F32 fully supported; Q4_K needs dequant->gather->requant)"
        )

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


def _work_shape(tensor) -> tuple[int, ...]:
    """Shape used for gather planning (logical numpy order)."""
    if is_float_type(tensor.tensor_type) or tensor.tensor_type in {
        GGMLQuantizationType.I8,
        GGMLQuantizationType.I16,
        GGMLQuantizationType.I32,
        GGMLQuantizationType.I64,
    }:
        return tuple(int(x) for x in tensor.data.shape)
    return _logical_numpy_shape(tensor)


def _gathered_shape(shape: tuple[int, ...], axis_size: int, n_keep: int, name: str) -> tuple[int, ...]:
    axis = axis_matching(shape, axis_size, name)
    out = list(shape)
    out[axis] = int(n_keep)
    return tuple(out)


def _np_dtype_for_qtype(qtype: GGMLQuantizationType) -> np.dtype:
    if qtype == GGMLQuantizationType.F16:
        return np.dtype(np.float16)
    if qtype == GGMLQuantizationType.F32:
        return np.dtype(np.float32)
    if qtype == GGMLQuantizationType.F64:
        return np.dtype(np.float64)
    if qtype == GGMLQuantizationType.BF16:
        return np.dtype(np.uint16)
    if qtype == GGMLQuantizationType.I8:
        return np.dtype(np.int8)
    if qtype == GGMLQuantizationType.I16:
        return np.dtype(np.int16)
    if qtype == GGMLQuantizationType.I32:
        return np.dtype(np.int32)
    if qtype == GGMLQuantizationType.I64:
        return np.dtype(np.int64)
    return np.dtype(np.uint8)


@dataclass
class _OutPlan:
    name: str
    gather_indices: list[int] | None
    axis_size: int
    src_qtype: GGMLQuantizationType
    out_qtype: GGMLQuantizationType
    info_shape: tuple[int, ...]
    nbytes: int
    np_dtype: np.dtype
    raw_dtype: GGMLQuantizationType | None


def _plan_one(
    tensor,
    table: PruneTable,
    *,
    n_ff: int,
    n_vocab: int,
    old_ids: list[int],
    q4_k_to_f16: bool,
) -> _OutPlan:
    name = tensor.name
    qtype = tensor.tensor_type
    gather_indices: list[int] | None = None
    axis_size = 0

    if is_ffn_tensor(name):
        parsed = parse_blk(name)
        assert parsed is not None
        keep = table.keep_channels[parsed[0]]
        if n_ff <= 0:
            raise ValueError(f"{name}: cannot infer n_ff for gather")
        gather_indices = keep
        axis_size = n_ff
    elif is_embed(name) or is_lm_head(name):
        if old_ids:
            if n_vocab <= 0:
                raise ValueError(f"{name}: cannot infer n_vocab for gather")
            gather_indices = old_ids
            axis_size = n_vocab

    if gather_indices is None:
        if is_q4_k(qtype):
            return _OutPlan(
                name=name,
                gather_indices=None,
                axis_size=0,
                src_qtype=qtype,
                out_qtype=qtype,
                info_shape=tuple(int(x) for x in tensor.data.shape),
                nbytes=int(tensor.n_bytes),
                np_dtype=np.dtype(np.uint8),
                raw_dtype=qtype,
            )
        return _OutPlan(
            name=name,
            gather_indices=None,
            axis_size=0,
            src_qtype=qtype,
            out_qtype=qtype,
            info_shape=tuple(int(x) for x in tensor.data.shape),
            nbytes=int(tensor.data.nbytes),
            np_dtype=tensor.data.dtype,
            raw_dtype=None if is_float_type(qtype) else qtype,
        )

    work = _work_shape(tensor)
    out_shape = _gathered_shape(work, axis_size, len(gather_indices), name)
    out_qtype = qtype
    if is_q4_k(qtype):
        if q4_k_to_f16:
            out_qtype = GGMLQuantizationType.F16
        else:
            out_qtype = GGMLQuantizationType.Q4_K
    elif not is_float_type(qtype):
        raise TypeError(
            f"{name}: gather on {qtype.name} is not supported in v1 "
            f"(F16/F32 fully supported; Q4_K needs dequant->gather->requant)"
        )

    n_elem = 1
    for d in out_shape:
        n_elem *= int(d)

    if out_qtype == GGMLQuantizationType.Q4_K:
        if out_shape[-1] % QK_K != 0:
            raise ValueError(
                f"{name}: Q4_K requant needs last dim a multiple of {QK_K}, "
                f"gathered shape {out_shape}. Keep-channel counts for ffn_down "
                f"must be a multiple of 256, or pass --q4-k-to-f16 (host debug)."
            )
        info_shape = quant_shape_to_byte_shape(out_shape, GGMLQuantizationType.Q4_K)
        nbytes = int(n_elem * Q4_K_TYPE_SIZE // QK_K)
        return _OutPlan(
            name=name,
            gather_indices=gather_indices,
            axis_size=axis_size,
            src_qtype=qtype,
            out_qtype=out_qtype,
            info_shape=info_shape,
            nbytes=nbytes,
            np_dtype=np.dtype(np.uint8),
            raw_dtype=GGMLQuantizationType.Q4_K,
        )

    np_dtype = _np_dtype_for_qtype(out_qtype)
    nbytes = int(n_elem * np_dtype.itemsize)
    return _OutPlan(
        name=name,
        gather_indices=gather_indices,
        axis_size=axis_size,
        src_qtype=qtype,
        out_qtype=out_qtype,
        info_shape=out_shape,
        nbytes=nbytes,
        np_dtype=np_dtype,
        raw_dtype=None,
    )


def _materialize(tensor, plan: _OutPlan, *, q4_k_to_f16: bool) -> np.ndarray:
    array, _out_type = _gather_or_copy(
        plan.name,
        tensor.data,
        plan.src_qtype,
        gather_indices=plan.gather_indices,
        axis_size=plan.axis_size,
        q4_k_to_f16=q4_k_to_f16,
    )
    return np.ascontiguousarray(array)


def pack_tensors(
    reader: GGUFReader,
    table: PruneTable,
    *,
    q4_k_to_f16: bool = False,
) -> list[tuple[str, np.ndarray, GGMLQuantizationType]]:
    """Test helper: materialize every kept tensor. Pack path streams instead."""
    n_layer = table.n_layer or infer_n_layer(reader, 0)
    n_ff = infer_n_ff(reader, 0)
    n_vocab = infer_n_vocab(reader, 0)
    old_ids = table.old_ids_in_row_order()
    out: list[tuple[str, np.ndarray, GGMLQuantizationType]] = []

    for t in reader.tensors:
        if should_omit(t.name, table, n_layer):
            continue
        plan = _plan_one(
            t, table, n_ff=n_ff, n_vocab=n_vocab, old_ids=old_ids, q4_k_to_f16=q4_k_to_f16
        )
        gathered = _materialize(t, plan, q4_k_to_f16=q4_k_to_f16)
        out.append((t.name, gathered, plan.out_qtype))
    return out


def write_packed_gguf(
    src_path: str,
    out_path: str,
    table: PruneTable,
    *,
    q4_k_to_f16: bool = False,
    arch: str | None = None,
    serve_ok: bool | None = None,
) -> list[str]:
    """Read full GGUF, pack, stream-write remnant one tensor at a time."""
    reader = GGUFReader(src_path)
    arch_name = arch or infer_arch(reader)
    n_layer = table.n_layer or infer_n_layer(reader, 0)
    n_ff = infer_n_ff(reader, 0)
    n_vocab = infer_n_vocab(reader, 0)
    old_ids = table.old_ids_in_row_order()

    if serve_ok is None:
        serve_ok = remnant_serve_ok(
            q4_k_to_f16=q4_k_to_f16,
            source_has_q4_k=source_has_q4_k(reader),
        )
    table.serve_ok = bool(serve_ok)

    hidden = infer_n_embd(reader, HIDDEN_27B)
    weight_est = estimate_baked_weight_bytes(table, hidden=hidden, n_vocab=n_vocab)

    writer = GGUFWriter(out_path, arch_name)
    writer.add_custom_alignment(TENSOR_ALIGN)
    copy_source_kv(reader, writer)
    write_prune_kv(writer, table, serve_ok=serve_ok, weight_bytes=weight_est)

    plans: list[tuple[Any, _OutPlan]] = []
    for t in reader.tensors:
        if should_omit(t.name, table, n_layer):
            continue
        plan = _plan_one(
            t, table, n_ff=n_ff, n_vocab=n_vocab, old_ids=old_ids, q4_k_to_f16=q4_k_to_f16
        )
        if plan.raw_dtype is not None:
            writer.add_tensor_info(
                plan.name, plan.info_shape, plan.np_dtype, plan.nbytes, raw_dtype=plan.raw_dtype
            )
        else:
            writer.add_tensor_info(plan.name, plan.info_shape, plan.np_dtype, plan.nbytes)
        plans.append((t, plan))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    names: list[str] = []
    for t, plan in plans:
        array = _materialize(t, plan, q4_k_to_f16=q4_k_to_f16)
        if array.nbytes != plan.nbytes:
            raise ValueError(
                f"{plan.name}: streamed nbytes {array.nbytes} != planned {plan.nbytes}"
            )
        writer.write_tensor_data(array)
        names.append(plan.name)
        del array

    writer.close()
    # Serve prefers on-disk file size over the KV estimate written before tensors.
    file_size = int(Path(out_path).stat().st_size)
    patch_uint64_kv(out_path, KV_WEIGHT_BYTES, file_size)
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


def pack_gguf(
    model,
    table: PruneTable,
    out,
    q4_k_to_f16: bool = False,
    serve_ok: bool | None = None,
) -> list[str]:
    return write_packed_gguf(
        str(model),
        str(out),
        table,
        q4_k_to_f16=q4_k_to_f16,
        serve_ok=serve_ok,
    )


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
