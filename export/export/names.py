"""Stable Qwen hybrid tensor names and pack-id mapping.

C++ serve-path loader must match these strings exactly.
Names follow llama.cpp LLM_ARCH_QWEN35 (Qwen3.5 / 3.6 / 3.8 hybrid GGUF).

64-layer 27B pattern: groups of 4 = 3 DeltaNet + 1 Gated Attention.
  layer = 4 * group + slot
  slot 0,1,2 -> DeltaNet pack  (global pack id 0..47)
  slot 3     -> Gated Attention (never dropped)
  pack p     -> layer 4 * (p // 3) + (p % 3)
"""

from __future__ import annotations

import re
from typing import Optional

# --- 27B product constants (synthetic tests use smaller dims) ---
N_LAYER_27B = 64
N_FFN_27B = 17408
N_PACKS_27B = 48
N_GATED_ATTN_27B = 16
N_VOCAB_27B = 248320
GROUP_SIZE = 4
DELTANET_PER_GROUP = 3
TENSOR_ALIGN = 256

# --- top-level tensors ---
TOKEN_EMBD = "token_embd.weight"
OUTPUT = "output.weight"  # lm_head; absent when tied to token_embd
OUTPUT_NORM = "output_norm.weight"

# --- per-block suffixes (llama.cpp QWEN35) ---
ATTN_NORM = "attn_norm.weight"
POST_ATTN_NORM = "post_attention_norm.weight"

# Gated Attention (QKVO + 4 KV heads). Copied through, never dropped.
ATTN_Q = "attn_q.weight"  # Q and gate concatenated on Qwen3.5
ATTN_K = "attn_k.weight"
ATTN_V = "attn_v.weight"
ATTN_OUT = "attn_output.weight"
ATTN_Q_NORM = "attn_q_norm.weight"
ATTN_K_NORM = "attn_k_norm.weight"
ATTN_V_NORM = "attn_v_norm.weight"  # optional

# DeltaNet / Gated DeltaNet pack tensors. Dropped as a unit with the pack.
ATTN_QKV = "attn_qkv.weight"
ATTN_GATE = "attn_gate.weight"  # z-gate on DeltaNet layers
SSM_IN = "ssm_in.weight"  # alternate fused in-proj (qwen3next-style)
SSM_BA = "ssm_ba.weight"
SSM_CONV1D = "ssm_conv1d.weight"
SSM_DT = "ssm_dt.bias"
SSM_A = "ssm_a"  # no .weight suffix in llama.cpp
SSM_BETA = "ssm_beta.weight"
SSM_ALPHA = "ssm_alpha.weight"
SSM_NORM = "ssm_norm.weight"
SSM_OUT = "ssm_out.weight"

# FFN: one channel index gathered from all three.
FFN_GATE = "ffn_gate.weight"
FFN_UP = "ffn_up.weight"
FFN_DOWN = "ffn_down.weight"

# MTP / NextN (omitted unless prune table keep_mtp)
NEXTN_MARKERS = ("nextn.", ".nextn.", "shared_head")

# Vision tower (omitted unless prune table keep_vision)
VISION_PREFIXES = ("v.", "mm.", "vision.", "clip.")

BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

# --- KV metadata keys baked into the remnant GGUF ---
KV_VERSION = "micro_llm.version"
KV_ALIGN = "micro_llm.tensor_align"
KV_KEEP_PACKS = "micro_llm.keep_packs"
KV_KEEP_CH_N = "micro_llm.keep_channels.n"
KV_KEEP_CH_IDS = "micro_llm.keep_channels.ids"
KV_VOCAB_OLD = "micro_llm.vocab_remap.old_ids"
KV_VOCAB_ROWS = "micro_llm.vocab_remap.rows"
KV_KEEP_VISION = "micro_llm.keep_vision"
KV_KEEP_MTP = "micro_llm.keep_mtp"
KV_PREFIX = "micro_llm."


def blk(layer: int, suffix: str) -> str:
    return f"blk.{layer}.{suffix}"


def parse_blk(name: str) -> Optional[tuple[int, str]]:
    m = BLK_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def pack_to_layer(pack_id: int) -> int:
    """Global pack 0..47 -> layer. layer = 4 * group + slot."""
    if pack_id < 0:
        raise ValueError(f"pack id must be >= 0, got {pack_id}")
    group, slot = divmod(int(pack_id), DELTANET_PER_GROUP)
    return GROUP_SIZE * group + slot


def layer_to_pack(layer: int) -> Optional[int]:
    """Layer -> global pack id, or None if this layer is Gated Attention."""
    group, slot = divmod(int(layer), GROUP_SIZE)
    if slot == GROUP_SIZE - 1:
        return None
    return group * DELTANET_PER_GROUP + slot


def is_gated_attention_layer(layer: int) -> bool:
    return int(layer) % GROUP_SIZE == GROUP_SIZE - 1


def is_deltanet_layer(layer: int) -> bool:
    return not is_gated_attention_layer(layer)


def n_packs_for_layers(n_layer: int) -> int:
    n_groups = (n_layer + GROUP_SIZE - 1) // GROUP_SIZE
    # last group may be short; count DeltaNet slots that exist
    count = 0
    for p in range(n_groups * DELTANET_PER_GROUP):
        if pack_to_layer(p) < n_layer:
            count += 1
    return count


def protected_layers(n_layer: int) -> frozenset[int]:
    """First two and last two layers are never dropped as layers."""
    if n_layer <= 0:
        return frozenset()
    ids = {0, 1, max(0, n_layer - 2), max(0, n_layer - 1)}
    return frozenset(i for i in ids if i < n_layer)


def protected_packs(n_layer: int) -> frozenset[int]:
    packs = set()
    for layer in protected_layers(n_layer):
        p = layer_to_pack(layer)
        if p is not None:
            packs.add(p)
    return frozenset(packs)


def is_ffn_tensor(name: str) -> bool:
    parsed = parse_blk(name)
    if parsed is None:
        return False
    suffix = parsed[1]
    return suffix.startswith("ffn_gate") or suffix.startswith("ffn_up") or suffix.startswith("ffn_down")


def is_shared_norm(name: str) -> bool:
    parsed = parse_blk(name)
    if parsed is None:
        return False
    return parsed[1] in {ATTN_NORM, POST_ATTN_NORM, "ffn_norm.weight"}


def is_attention_tensor(name: str) -> bool:
    """Gated Attention QKVO / norms (not DeltaNet qkv/gate, not FFN)."""
    parsed = parse_blk(name)
    if parsed is None:
        return False
    s = parsed[1]
    if s.startswith("attn_qkv"):
        return False
    if s.startswith("attn_gate"):
        return False
    return (
        s.startswith("attn_q")
        or s.startswith("attn_k")
        or s.startswith("attn_v")
        or s.startswith("attn_output")
        or s.startswith("attn_out")
    )


def is_deltanet_mixer(name: str) -> bool:
    """Tensors that belong to a DeltaNet pack (dropped with the pack)."""
    parsed = parse_blk(name)
    if parsed is None:
        return False
    s = parsed[1]
    if s.startswith("ssm_") or s.startswith("attn_qkv") or s.startswith("attn_gate"):
        return True
    return False


def is_vision_tensor(name: str) -> bool:
    lower = name.lower()
    return lower.startswith(VISION_PREFIXES) or ".vision." in lower


def is_mtp_tensor(name: str, n_layer: int) -> bool:
    if any(m in name for m in NEXTN_MARKERS):
        return True
    parsed = parse_blk(name)
    if parsed is not None and parsed[0] >= n_layer:
        return True
    return False


def is_embed(name: str) -> bool:
    return name == TOKEN_EMBD or name == "token_embd.weight"


def is_lm_head(name: str) -> bool:
    return name == OUTPUT or name == "output.weight"


def is_tokenizer_kv(key: str) -> bool:
    return key.startswith("tokenizer.")
