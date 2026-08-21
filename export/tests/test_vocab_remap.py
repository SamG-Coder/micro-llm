"""Vocab remap is old_id -> dense row; unused rows gone; no tokenizer rewrite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from gguf import GGUFReader

from export.names import OUTPUT, TOKEN_EMBD
from export.writer import list_tensors
from tests.conftest import N_EMBD, N_TOKENS, N_VOCAB


def test_unused_rows_gone_and_dense_order(remnant_gguf, prune_table):
    tensors = list_tensors(remnant_gguf)
    emb = tensors[TOKEN_EMBD]["data"]
    head = tensors[OUTPUT]["data"]
    old_ids = prune_table.old_ids_in_row_order()
    assert emb.shape[0] == len(old_ids)
    assert head.shape[0] == len(old_ids)
    assert emb.shape[0] < N_VOCAB

    # token_embd was filled as arange starting at 1, so row r, col 0 = 1 + r * n_embd
    for i, old in enumerate(old_ids):
        expected = np.float16(1 + old * N_EMBD)
        assert emb[i, 0] == expected
        assert prune_table.vocab_remap[old] == i


def test_tokenizer_kv_not_rewritten(full_gguf, remnant_gguf):
    src = GGUFReader(str(full_gguf))
    dst = GGUFReader(str(remnant_gguf))
    src_tokens = src.get_field("tokenizer.ggml.tokens").contents()
    dst_tokens = dst.get_field("tokenizer.ggml.tokens").contents()
    assert src_tokens == dst_tokens == N_TOKENS
    # original ids still occupy their original slots in the tokenizer list
    assert dst_tokens[5] == "tok-05"
    assert dst_tokens[1] == "tok-01"


def test_no_tokenizer_file_written(remnant_gguf):
    folder = Path(remnant_gguf).parent
    extras = [
        p
        for p in folder.iterdir()
        if p.name != remnant_gguf.name and "tokenizer" in p.name.lower()
    ]
    assert extras == []
    assert not (folder / "tokenizer.json").exists()
    assert not (folder / "tokenizer.gguf").exists()
