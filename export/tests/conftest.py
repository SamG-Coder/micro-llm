"""Synthetic hybrid GGUF + prune table for unit tests. No 27B file."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from gguf import GGUFWriter

from export.names import (
    ATTN_GATE,
    ATTN_K,
    ATTN_K_NORM,
    ATTN_NORM,
    ATTN_OUT,
    ATTN_Q,
    ATTN_Q_NORM,
    ATTN_QKV,
    ATTN_V,
    FFN_DOWN,
    FFN_GATE,
    FFN_UP,
    OUTPUT,
    OUTPUT_NORM,
    POST_ATTN_NORM,
    SSM_A,
    SSM_ALPHA,
    SSM_BETA,
    SSM_CONV1D,
    SSM_DT,
    SSM_NORM,
    SSM_OUT,
    TENSOR_ALIGN,
    TOKEN_EMBD,
    blk,
    is_gated_attention_layer,
)
from export.prune_table import parse_prune_table


# Tiny hybrid: 8 layers = 2 groups. Packs 0..5 -> layers 0,1,2,4,5,6.
N_LAYER = 8
N_EMBD = 16
N_FF = 32
N_VOCAB = 20
N_HEAD = 4
N_KV = 2
HEAD_DIM = 4
N_TOKENS = ["tok-%02d" % i for i in range(N_VOCAB)]


def _fill(shape, start, dtype=np.float16):
    n = int(np.prod(shape))
    # Stay inside exact F16 integers so copy-through tests can compare bitwise.
    arr = (np.arange(n, dtype=np.float32) + float(start % 1024)).reshape(shape)
    return arr.astype(dtype)


def write_synthetic_gguf(path: Path, *, extra_vision: bool = True, extra_mtp: bool = True) -> Path:
    w = GGUFWriter(str(path), "qwen35")
    w.add_name("synthetic-hybrid")
    w.add_block_count(N_LAYER)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_KV)
    w.add_vocab_size(N_VOCAB)
    w.add_full_attention_interval(4)
    w.add_custom_alignment(TENSOR_ALIGN)
    w.add_token_list(N_TOKENS)
    w.add_tokenizer_model("gpt2")

    seq = 1
    w.add_tensor(TOKEN_EMBD, _fill((N_VOCAB, N_EMBD), seq))
    seq += N_VOCAB * N_EMBD
    w.add_tensor(OUTPUT, _fill((N_VOCAB, N_EMBD), seq))
    seq += N_VOCAB * N_EMBD
    w.add_tensor(OUTPUT_NORM, _fill((N_EMBD,), seq))
    seq += N_EMBD

    for i in range(N_LAYER):
        w.add_tensor(blk(i, ATTN_NORM), _fill((N_EMBD,), seq))
        seq += N_EMBD
        w.add_tensor(blk(i, POST_ATTN_NORM), _fill((N_EMBD,), seq))
        seq += N_EMBD

        if is_gated_attention_layer(i):
            w.add_tensor(blk(i, ATTN_Q), _fill((N_HEAD * HEAD_DIM * 2, N_EMBD), seq))
            seq += N_HEAD * HEAD_DIM * 2 * N_EMBD
            w.add_tensor(blk(i, ATTN_K), _fill((N_KV * HEAD_DIM, N_EMBD), seq))
            seq += N_KV * HEAD_DIM * N_EMBD
            w.add_tensor(blk(i, ATTN_V), _fill((N_KV * HEAD_DIM, N_EMBD), seq))
            seq += N_KV * HEAD_DIM * N_EMBD
            w.add_tensor(blk(i, ATTN_OUT), _fill((N_EMBD, N_HEAD * HEAD_DIM), seq))
            seq += N_EMBD * N_HEAD * HEAD_DIM
            w.add_tensor(blk(i, ATTN_Q_NORM), _fill((HEAD_DIM,), seq))
            seq += HEAD_DIM
            w.add_tensor(blk(i, ATTN_K_NORM), _fill((HEAD_DIM,), seq))
            seq += HEAD_DIM
        else:
            w.add_tensor(blk(i, ATTN_QKV), _fill((48, N_EMBD), seq))
            seq += 48 * N_EMBD
            w.add_tensor(blk(i, ATTN_GATE), _fill((32, N_EMBD), seq))
            seq += 32 * N_EMBD
            w.add_tensor(blk(i, SSM_CONV1D), _fill((4, 48), seq))
            seq += 4 * 48
            w.add_tensor(blk(i, SSM_DT), _fill((8,), seq))
            seq += 8
            w.add_tensor(blk(i, SSM_A), _fill((8,), seq))
            seq += 8
            w.add_tensor(blk(i, SSM_BETA), _fill((8, N_EMBD), seq))
            seq += 8 * N_EMBD
            w.add_tensor(blk(i, SSM_ALPHA), _fill((8, N_EMBD), seq))
            seq += 8 * N_EMBD
            w.add_tensor(blk(i, SSM_NORM), _fill((4,), seq))
            seq += 4
            w.add_tensor(blk(i, SSM_OUT), _fill((N_EMBD, 32), seq))
            seq += N_EMBD * 32

        gate = np.zeros((N_FF, N_EMBD), dtype=np.float16)
        up = np.zeros((N_FF, N_EMBD), dtype=np.float16)
        down = np.zeros((N_EMBD, N_FF), dtype=np.float16)
        for c in range(N_FF):
            # Exact F16 integers; same channel id on gate/up/down.
            gate[c, :] = np.float16(i * 64 + c)
            up[c, :] = np.float16(i * 64 + 256 + c)
            down[:, c] = np.float16(i * 64 + 512 + c)
        w.add_tensor(blk(i, FFN_GATE), gate)
        w.add_tensor(blk(i, FFN_UP), up)
        w.add_tensor(blk(i, FFN_DOWN), down)

    if extra_vision:
        w.add_tensor("v.patch_embd.weight", _fill((8, N_EMBD), 200))
    if extra_mtp:
        w.add_tensor("blk.8.nextn.eh_proj.weight", _fill((N_EMBD, 2 * N_EMBD), 300))
        w.add_tensor(blk(8, FFN_GATE), _fill((N_FF, N_EMBD), 400))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def default_prune_dict() -> dict:
    # Drop packs 2 and 4 (layers 2 and 5). Packs 0,1,5 are protected (layers 0,1,6).
    keep_channels = {}
    for i in range(N_LAYER):
        keep_channels[str(i)] = [2, 5, 7, 11] if i % 2 == 0 else [1, 4, 9]
    return {
        "keep_channels": keep_channels,
        "keep_packs": [0, 3],
        "vocab_remap": {"1": 0, "5": 1, "9": 2, "14": 3},
        "keep_vision": False,
        "keep_mtp": False,
    }


@pytest.fixture
def synth_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def full_gguf(tmp_path: Path) -> Path:
    return write_synthetic_gguf(tmp_path / "full.gguf")


@pytest.fixture
def prune_path(tmp_path: Path) -> Path:
    p = tmp_path / "keep.json"
    p.write_text(json.dumps(default_prune_dict()), encoding="utf-8")
    return p


@pytest.fixture
def prune_table():
    return parse_prune_table(default_prune_dict(), n_layer=N_LAYER)


@pytest.fixture
def remnant_gguf(tmp_path: Path, full_gguf: Path, prune_table) -> Path:
    from export.writer import pack_gguf

    out = tmp_path / "remnant.gguf"
    pack_gguf(full_gguf, prune_table, out)
    return out
