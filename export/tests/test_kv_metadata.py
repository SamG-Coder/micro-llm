"""KV metadata round-trips keep_channels / keep_packs / vocab_remap."""

from __future__ import annotations

from gguf import GGUFReader

from export.names import KV_KEEP_CH_IDS, KV_KEEP_CH_N, KV_KEEP_PACKS, KV_VOCAB_OLD, KV_VOCAB_ROWS
from export.prune_table import decode_kv, fields_from_reader


def test_kv_round_trip(remnant_gguf, prune_table):
    reader = GGUFReader(str(remnant_gguf))
    baked = decode_kv(fields_from_reader(reader))

    assert baked.keep_packs == prune_table.keep_packs
    assert baked.keep_channels == prune_table.keep_channels
    assert baked.vocab_remap == prune_table.vocab_remap
    assert baked.keep_vision is False
    assert baked.keep_mtp is False

    # Arrays are present as typed KV, not only a JSON blob.
    assert reader.get_field(KV_KEEP_PACKS) is not None
    assert reader.get_field(KV_KEEP_CH_N) is not None
    assert reader.get_field(KV_KEEP_CH_IDS) is not None
    assert reader.get_field(KV_VOCAB_OLD) is not None
    assert reader.get_field(KV_VOCAB_ROWS) is not None

    olds = reader.get_field(KV_VOCAB_OLD).contents()
    rows = reader.get_field(KV_VOCAB_ROWS).contents()
    assert olds == prune_table.old_ids_in_row_order()
    assert rows == prune_table.rows_in_row_order()


def test_alignment_key(remnant_gguf):
    reader = GGUFReader(str(remnant_gguf))
    assert reader.alignment == 256
    assert reader.get_field("general.alignment").contents() == 256
    assert reader.get_field("micro_llm.tensor_align").contents() == 256
    assert reader.get_field("micro_llm.version").contents() == 1
