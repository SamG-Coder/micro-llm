"""Read/write the hour-end binary MLPT prune-table dump.

Little-endian. Fixed 27B shape. Total 17,997,328 bytes.

In-memory arrays are convenient 2D / bool views. On disk, floor and vocab
are packed bitsets (bit i in byte i>>3, bit i&7).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from export.names import (
    N_FFN_27B,
    N_LAYER_27B,
    N_PACKS_27B,
    N_VOCAB_27B,
    TENSOR_ALIGN,
    pack_to_layer,
)

MAGIC = b"MLPT"
VERSION = 1
N_LAYERS = N_LAYER_27B
N_FFN = N_FFN_CHANNELS = N_FFN_27B
N_PACKS = N_PACKS_27B
VOCAB = VOCAB_SIZE = N_VOCAB_27B
HEADER_SIZE = 80
FLAG_HAS_FLOOR = 1 << 0
DEFAULT_FIRE_EPS = 1.0e-6
DEFAULT_SPIKE_EPS = 1.0e-6

CHANNEL_RECORD_SIZE = 16
PACK_RECORD_SIZE = 24
FLOOR_BYTES = (N_LAYERS * N_FFN) // 8  # 139264
VOCAB_BYTES = (VOCAB + 7) // 8  # 31040
TOTAL_BYTES = MLPT_SIZE = (
    HEADER_SIZE
    + N_LAYERS * N_FFN * CHANNEL_RECORD_SIZE
    + N_PACKS * PACK_RECORD_SIZE
    + FLOOR_BYTES
    + VOCAB_BYTES
)
assert TOTAL_BYTES == 17_997_328

CHANNEL_DTYPE = np.dtype(
    [("n_fired", "<u8"), ("sumsq", "<f4"), ("maxabs", "<f4")],
    align=False,
)
PACK_DTYPE = np.dtype(
    [
        ("pack", "<u4"),
        ("layer", "<u4"),
        ("n_spike", "<u8"),
        ("sumsq_residual", "<f8"),
    ],
    align=False,
)
assert CHANNEL_DTYPE.itemsize == CHANNEL_RECORD_SIZE
assert PACK_DTYPE.itemsize == PACK_RECORD_SIZE

_HEADER_STRUCT = struct.Struct("<4s7I2fQI7I")
assert _HEADER_STRUCT.size == HEADER_SIZE


@dataclass(frozen=True)
class ChannelStat:
    n_fired: int
    sumsq: float
    maxabs: float


@dataclass(frozen=True)
class PackStat:
    pack: int
    layer: int
    n_spike: int
    sumsq_residual: float


@dataclass(frozen=True)
class MlptHeader:
    magic: bytes
    version: int
    n_layers: int
    n_ffn_channels: int
    n_packs: int
    vocab_size: int
    tensor_align: int
    header_size: int
    fire_eps: float
    spike_eps: float
    n_tokens: int
    flags: int
    reserved: tuple[int, ...]


def _pack_bits(flags: np.ndarray) -> np.ndarray:
    flat = np.asarray(flags, dtype=bool).ravel()
    return np.packbits(flat, bitorder="little")


def _unpack_bits(buf: np.ndarray, n_bits: int, shape: tuple[int, ...]) -> np.ndarray:
    bits = np.unpackbits(np.asarray(buf, dtype=np.uint8), bitorder="little")[:n_bits]
    return bits.reshape(shape).astype(bool)


class MlptDump:
    """In-memory MLPT dump with 2D channel views and bool floor/vocab."""

    def __init__(
        self,
        *,
        n_fired: np.ndarray | None = None,
        sumsq: np.ndarray | None = None,
        maxabs: np.ndarray | None = None,
        n_spike: np.ndarray | None = None,
        sumsq_residual: np.ndarray | None = None,
        pack: np.ndarray | None = None,
        pack_layer: np.ndarray | None = None,
        floor: np.ndarray | None = None,
        vocab: np.ndarray | None = None,
        n_tokens: int = 0,
        fire_eps: float = DEFAULT_FIRE_EPS,
        spike_eps: float = DEFAULT_SPIKE_EPS,
        flags: int = FLAG_HAS_FLOOR,
        header: MlptHeader | None = None,
    ) -> None:
        self.n_fired = (
            n_fired if n_fired is not None else np.zeros((N_LAYERS, N_FFN), dtype=np.uint64)
        )
        self.sumsq = (
            sumsq if sumsq is not None else np.zeros((N_LAYERS, N_FFN), dtype=np.float32)
        )
        self.maxabs = (
            maxabs if maxabs is not None else np.zeros((N_LAYERS, N_FFN), dtype=np.float32)
        )
        self.n_spike = n_spike if n_spike is not None else np.zeros(N_PACKS, dtype=np.uint64)
        self.sumsq_residual = (
            sumsq_residual
            if sumsq_residual is not None
            else np.zeros(N_PACKS, dtype=np.float64)
        )
        self._pack_ids = (
            pack if pack is not None else np.arange(N_PACKS, dtype=np.uint32)
        )
        self.pack_layer = (
            pack_layer
            if pack_layer is not None
            else np.array([pack_to_layer(p) for p in range(N_PACKS)], dtype=np.uint32)
        )
        self.floor = (
            floor if floor is not None else np.zeros((N_LAYERS, N_FFN), dtype=bool)
        )
        self.vocab = vocab if vocab is not None else np.zeros(VOCAB, dtype=bool)
        self.n_tokens = int(n_tokens)
        self.fire_eps = float(fire_eps)
        self.spike_eps = float(spike_eps)
        self.flags = int(flags)
        self._header = header

    @property
    def n_layers(self) -> int:
        return N_LAYERS

    @property
    def n_ffn(self) -> int:
        return N_FFN

    @property
    def n_ffn_channels(self) -> int:
        return N_FFN

    @property
    def n_packs(self) -> int:
        return N_PACKS

    @property
    def vocab_size(self) -> int:
        return VOCAB

    @property
    def header(self) -> MlptHeader:
        if self._header is not None:
            # keep n_tokens in sync if the caller mutated it
            h = self._header
            return MlptHeader(
                magic=h.magic,
                version=h.version,
                n_layers=h.n_layers,
                n_ffn_channels=h.n_ffn_channels,
                n_packs=h.n_packs,
                vocab_size=h.vocab_size,
                tensor_align=h.tensor_align,
                header_size=h.header_size,
                fire_eps=self.fire_eps,
                spike_eps=self.spike_eps,
                n_tokens=self.n_tokens,
                flags=self.flags,
                reserved=h.reserved,
            )
        return MlptHeader(
            magic=MAGIC,
            version=VERSION,
            n_layers=N_LAYERS,
            n_ffn_channels=N_FFN,
            n_packs=N_PACKS,
            vocab_size=VOCAB,
            tensor_align=TENSOR_ALIGN,
            header_size=HEADER_SIZE,
            fire_eps=self.fire_eps,
            spike_eps=self.spike_eps,
            n_tokens=self.n_tokens,
            flags=self.flags,
            reserved=(0, 0, 0, 0, 0, 0, 0),
        )

    def channel(self, layer: int, channel: int) -> ChannelStat:
        return ChannelStat(
            int(self.n_fired[layer, channel]),
            float(self.sumsq[layer, channel]),
            float(self.maxabs[layer, channel]),
        )

    @property
    def pack_ids(self) -> np.ndarray:
        return self._pack_ids

    @property
    def packs(self) -> np.ndarray:
        arr = np.empty(N_PACKS, dtype=PACK_DTYPE)
        arr["pack"] = self._pack_ids
        arr["layer"] = self.pack_layer
        arr["n_spike"] = self.n_spike
        arr["sumsq_residual"] = self.sumsq_residual
        return arr

    def pack(self, pack_id: int) -> PackStat:
        return self.pack_stat(pack_id)

    def pack_stat(self, pack_id: int) -> PackStat:
        return PackStat(
            int(self._pack_ids[pack_id]),
            int(self.pack_layer[pack_id]),
            int(self.n_spike[pack_id]),
            float(self.sumsq_residual[pack_id]),
        )

    def floor_keep(self, layer: int, channel: int) -> bool:
        return bool(self.floor[layer, channel])

    def vocab_seen(self, token_id: int) -> bool:
        return bool(self.vocab[token_id])

    def has_floor(self) -> bool:
        return bool(self.flags & FLAG_HAS_FLOOR)


def empty_dump() -> MlptDump:
    return MlptDump()


def make_mlpt(
    *,
    fire_eps: float = DEFAULT_FIRE_EPS,
    spike_eps: float = DEFAULT_SPIKE_EPS,
    n_tokens: int = 0,
    flags: int = FLAG_HAS_FLOOR,
    channels: Mapping[tuple[int, int], ChannelStat | tuple] | None = None,
    packs: Mapping[int, PackStat | tuple] | None = None,
    floor: Iterable[tuple[int, int]] | None = None,
    vocab: Iterable[int] | None = None,
) -> MlptDump:
    dump = MlptDump(fire_eps=fire_eps, spike_eps=spike_eps, n_tokens=n_tokens, flags=flags)
    if channels:
        for (layer, ch), stat in channels.items():
            if isinstance(stat, ChannelStat):
                dump.n_fired[layer, ch] = stat.n_fired
                dump.sumsq[layer, ch] = np.float32(stat.sumsq)
                dump.maxabs[layer, ch] = np.float32(stat.maxabs)
            else:
                n_fired, sumsq, maxabs = stat
                dump.n_fired[layer, ch] = n_fired
                dump.sumsq[layer, ch] = np.float32(sumsq)
                dump.maxabs[layer, ch] = np.float32(maxabs)
    if packs:
        for pack_id, stat in packs.items():
            if isinstance(stat, PackStat):
                dump.n_spike[pack_id] = stat.n_spike
                dump.sumsq_residual[pack_id] = stat.sumsq_residual
            else:
                dump.n_spike[int(pack_id)] = stat[0]
                dump.sumsq_residual[int(pack_id)] = stat[1]
    if floor:
        for layer, ch in floor:
            dump.floor[int(layer), int(ch)] = True
    if vocab:
        for tok in vocab:
            dump.vocab[int(tok)] = True
    return dump


def is_mlpt_file(path: str | Path) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    with open(p, "rb") as fh:
        return fh.read(4) == MAGIC


def is_mlpt_path(path: str | Path) -> bool:
    return is_mlpt_file(path)


def _unpack_header(blob: bytes) -> MlptHeader:
    if len(blob) < HEADER_SIZE:
        raise ValueError(f"MLPT header truncated ({len(blob)} bytes, want {HEADER_SIZE})")
    tup = _HEADER_STRUCT.unpack(blob[:HEADER_SIZE])
    return MlptHeader(
        magic=tup[0],
        version=int(tup[1]),
        n_layers=int(tup[2]),
        n_ffn_channels=int(tup[3]),
        n_packs=int(tup[4]),
        vocab_size=int(tup[5]),
        tensor_align=int(tup[6]),
        header_size=int(tup[7]),
        fire_eps=float(tup[8]),
        spike_eps=float(tup[9]),
        n_tokens=int(tup[10]),
        flags=int(tup[11]),
        reserved=tuple(int(x) for x in tup[12:19]),
    )


def validate_header(header: MlptHeader) -> None:
    if header.magic != MAGIC:
        raise ValueError(f"bad MLPT magic {header.magic!r}, want {MAGIC!r}")
    if header.version != VERSION:
        raise ValueError(f"unsupported MLPT version {header.version}, want {VERSION}")
    if header.n_layers != N_LAYERS:
        raise ValueError(f"n_layers={header.n_layers}, want {N_LAYERS}")
    if header.n_ffn_channels != N_FFN:
        raise ValueError(f"n_ffn_channels={header.n_ffn_channels}, want {N_FFN}")
    if header.n_packs != N_PACKS:
        raise ValueError(f"n_packs={header.n_packs}, want {N_PACKS}")
    if header.vocab_size != VOCAB:
        raise ValueError(f"vocab_size={header.vocab_size}, want {VOCAB}")
    if header.tensor_align != TENSOR_ALIGN:
        raise ValueError(f"tensor_align={header.tensor_align}, want {TENSOR_ALIGN}")
    if header.header_size != HEADER_SIZE:
        raise ValueError(f"header_size={header.header_size}, want {HEADER_SIZE}")


def read_mlpt_header(path: str | Path) -> MlptHeader:
    with open(path, "rb") as fh:
        blob = fh.read(HEADER_SIZE)
    header = _unpack_header(blob)
    validate_header(header)
    return header


def _validate_packs(pack: np.ndarray, pack_layer: np.ndarray) -> None:
    for p in range(N_PACKS):
        if int(pack[p]) != p:
            raise ValueError(
                f"pack id is not global sequential 0..47 (got pack={int(pack[p])} at index {p})"
            )
        expect_layer = pack_to_layer(p)
        if int(pack_layer[p]) != expect_layer:
            raise ValueError(
                f"pack {p} layer is not 4*group+slot (got {int(pack_layer[p])}, want {expect_layer})"
            )


def read_mlpt(path: str | Path) -> MlptDump:
    path = Path(path)
    data = path.read_bytes()
    if len(data) < HEADER_SIZE:
        raise ValueError(f"MLPT truncated ({len(data)} bytes)")
    header = _unpack_header(data[:HEADER_SIZE])
    validate_header(header)
    if len(data) != TOTAL_BYTES:
        raise ValueError(f"MLPT size {len(data)} != {TOTAL_BYTES}")

    off = HEADER_SIZE
    n_ch = N_LAYERS * N_FFN
    ch_bytes = n_ch * CHANNEL_RECORD_SIZE
    channels = np.frombuffer(data[off : off + ch_bytes], dtype=CHANNEL_DTYPE).copy()
    off += ch_bytes
    pack_bytes = N_PACKS * PACK_RECORD_SIZE
    packs = np.frombuffer(data[off : off + pack_bytes], dtype=PACK_DTYPE).copy()
    off += pack_bytes
    _validate_packs(packs["pack"], packs["layer"])
    floor_buf = np.frombuffer(data[off : off + FLOOR_BYTES], dtype=np.uint8).copy()
    off += FLOOR_BYTES
    vocab_buf = np.frombuffer(data[off : off + VOCAB_BYTES], dtype=np.uint8).copy()

    return MlptDump(
        n_fired=channels["n_fired"].reshape(N_LAYERS, N_FFN),
        sumsq=channels["sumsq"].reshape(N_LAYERS, N_FFN),
        maxabs=channels["maxabs"].reshape(N_LAYERS, N_FFN),
        n_spike=packs["n_spike"].copy(),
        sumsq_residual=packs["sumsq_residual"].copy(),
        pack=packs["pack"].copy(),
        pack_layer=packs["layer"].copy(),
        floor=_unpack_bits(floor_buf, N_LAYERS * N_FFN, (N_LAYERS, N_FFN)),
        vocab=_unpack_bits(vocab_buf, VOCAB, (VOCAB,)),
        n_tokens=header.n_tokens,
        fire_eps=header.fire_eps,
        spike_eps=header.spike_eps,
        flags=header.flags,
        header=header,
    )


def write_mlpt(path: str | Path, dump: MlptDump | None = None, **kwargs) -> MlptDump:
    """Write a full 17,997,328-byte MLPT file so tests can build fixtures."""
    if dump is None:
        dump = make_mlpt(**kwargs)
    elif kwargs:
        raise TypeError("write_mlpt: pass either dump or fixture kwargs, not both")

    h = dump.header
    validate_header(h)
    _validate_packs(dump._pack_ids, dump.pack_layer)

    channels = np.empty(N_LAYERS * N_FFN, dtype=CHANNEL_DTYPE)
    channels["n_fired"] = np.asarray(dump.n_fired, dtype=np.uint64).reshape(-1)
    channels["sumsq"] = np.asarray(dump.sumsq, dtype=np.float32).reshape(-1)
    channels["maxabs"] = np.asarray(dump.maxabs, dtype=np.float32).reshape(-1)

    packs = np.empty(N_PACKS, dtype=PACK_DTYPE)
    packs["pack"] = np.asarray(dump._pack_ids, dtype=np.uint32)
    packs["layer"] = np.asarray(dump.pack_layer, dtype=np.uint32)
    packs["n_spike"] = np.asarray(dump.n_spike, dtype=np.uint64)
    packs["sumsq_residual"] = np.asarray(dump.sumsq_residual, dtype=np.float64)

    floor_buf = _pack_bits(dump.floor)
    if floor_buf.size != FLOOR_BYTES:
        raise ValueError(f"floor bitset size {floor_buf.size} != {FLOOR_BYTES}")
    vocab_buf = _pack_bits(dump.vocab)
    if vocab_buf.size != VOCAB_BYTES:
        raise ValueError(f"vocab bitset size {vocab_buf.size} != {VOCAB_BYTES}")

    reserved = list(h.reserved)
    if len(reserved) < 7:
        reserved.extend([0] * (7 - len(reserved)))
    header_bytes = _HEADER_STRUCT.pack(
        h.magic,
        h.version,
        h.n_layers,
        h.n_ffn_channels,
        h.n_packs,
        h.vocab_size,
        h.tensor_align,
        h.header_size,
        h.fire_eps,
        h.spike_eps,
        h.n_tokens,
        h.flags,
        *reserved[:7],
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(header_bytes)
        fh.write(np.ascontiguousarray(channels).tobytes())
        fh.write(np.ascontiguousarray(packs).tobytes())
        fh.write(np.ascontiguousarray(floor_buf).tobytes())
        fh.write(np.ascontiguousarray(vocab_buf).tobytes())
    return dump


# Back-compat aliases used by pack.py / older tests
load_mlpt = read_mlpt
parse_mlpt = None  # binary files only
