#pragma once

// Milestone 1 card + split + GEMM ledgers. Slots A/B are reserved FIRST,
// then leftover under 14 GiB is parked FFN. KV 20k is reserved before
// sitting at 14. Park weights, not KV. Compile-tested without llama.cpp.

#include "micro_llm/types.hpp"

#include <cstdint>
#include <cstdio>
#include <string>

namespace micro_llm {

struct VramLedger {
    uint64_t ga_bytes = kPinnedGaWeightBytes;
    uint64_t kv20k_bytes = kHourKvReserveBytes;
    uint64_t scratch_bytes = kCudaScratchBytes;  // 0.9 GiB
    uint64_t slot_a_bytes = 0;
    uint64_t slot_b_bytes = 0;
    uint64_t parked_ffn_bytes = 0;
    uint64_t usable_bytes = kServeUsableBytes;  // 15.2 GiB hard
    uint32_t n_parked_ffn = 0;
    uint32_t n_streamed_ffn = 0;
};

inline constexpr uint64_t vram_ledger_used(const VramLedger& v) {
    return v.ga_bytes + v.kv20k_bytes + v.scratch_bytes + v.slot_a_bytes + v.slot_b_bytes +
           v.parked_ffn_bytes;
}

inline constexpr uint64_t vram_ledger_free_to_hard(const VramLedger& v) {
    const uint64_t used = vram_ledger_used(v);
    return v.usable_bytes > used ? v.usable_bytes - used : 0;
}

// Slots A/B first, then leftover under soft-14 becomes parked FFN.
inline constexpr VramLedger vram_ledger_slots_first(
    uint64_t layer_bytes = kQ4FfnLayerBytesMeasured5080,
    uint64_t soft = kHourCardSoftBytes) {
    VramLedger v;
    v.slot_a_bytes = layer_bytes;
    v.slot_b_bytes = layer_bytes;
    const uint64_t fixed = v.ga_bytes + v.kv20k_bytes + v.scratch_bytes + v.slot_a_bytes +
                           v.slot_b_bytes;
    uint32_t n_park = 0;
    if (fixed < soft && layer_bytes != 0) {
        n_park = static_cast<uint32_t>((soft - fixed) / layer_bytes);
        if (n_park > kMaxParkedFfnLayers) {
            n_park = kMaxParkedFfnLayers;
        }
    }
    v.n_parked_ffn = n_park;
    v.n_streamed_ffn = ffn_stream_layers(n_park);
    v.parked_ffn_bytes = static_cast<uint64_t>(n_park) * layer_bytes;
    return v;
}

inline std::string format_vram_ledger(const VramLedger& v) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "VRAM_LEDGER ga_MiB=%.1f kv20k_MiB=%.1f scratch_MiB=%.1f "
                  "slot_A_MiB=%.1f slot_B_MiB=%.1f parked_ffn_MiB=%.1f "
                  "used_MiB=%.1f free_to_15_2_MiB=%.1f park=%u stream=%u",
                  static_cast<double>(v.ga_bytes) / (1024.0 * 1024.0),
                  static_cast<double>(v.kv20k_bytes) / (1024.0 * 1024.0),
                  static_cast<double>(v.scratch_bytes) / (1024.0 * 1024.0),
                  static_cast<double>(v.slot_a_bytes) / (1024.0 * 1024.0),
                  static_cast<double>(v.slot_b_bytes) / (1024.0 * 1024.0),
                  static_cast<double>(v.parked_ffn_bytes) / (1024.0 * 1024.0),
                  static_cast<double>(vram_ledger_used(v)) / (1024.0 * 1024.0),
                  static_cast<double>(vram_ledger_free_to_hard(v)) / (1024.0 * 1024.0),
                  v.n_parked_ffn, v.n_streamed_ffn);
    return buf;
}

// Graph-split causes. callback/hooks MUST be 0 when cb_eval is nullptr.
struct SplitLedger {
    uint32_t callback_hooks = 0;
    uint32_t buffer_type = 0;
    uint32_t backend = 0;
    uint32_t op = 0;
    bool trace_off = true;
};

inline constexpr uint32_t split_ledger_total(const SplitLedger& s) {
    const uint32_t hooks = s.trace_off ? 0u : s.callback_hooks;
    return hooks + s.buffer_type + s.backend + s.op;
}

// One-backend CUDA FFN+DeltaNet+GA: only the host lm_head crossing remains.
inline constexpr SplitLedger split_ledger_trace_off_cuda_ffn() {
    SplitLedger s;
    s.trace_off = true;
    s.callback_hooks = 0;
    s.buffer_type = 1;  // CUDA weights -> host lm_head
    s.backend = 1;
    s.op = 1;  // MUL_MAT host logits
    return s;
}

inline std::string format_split_ledger(const SplitLedger& s) {
    const uint32_t hooks = s.trace_off ? 0u : s.callback_hooks;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "SPLIT_LEDGER callback/hooks=%u buffer_type=%u backend=%u op=%u "
                  "total=%u trace_off=%d",
                  hooks, s.buffer_type, s.backend, s.op, hooks + s.buffer_type + s.backend + s.op,
                  s.trace_off ? 1 : 0);
    return buf;
}

// 64 layers * (gate, up, down) = 192 FFN MUL_MATs per decode token.
inline constexpr uint32_t kFfnGemmPerLayer = 3;
inline constexpr uint32_t kFfnGemmPerToken = kNLayers * kFfnGemmPerLayer;

struct FfnGemmCounts {
    uint64_t cuda = 0;
    uint64_t cpu = 0;
};

inline constexpr FfnGemmCounts ffn_gemm_all_cuda() {
    return FfnGemmCounts{kFfnGemmPerToken, 0};
}

inline std::string format_ffn_gemm_line(const FfnGemmCounts& g) {
    char buf[192];
    std::snprintf(buf, sizeof(buf),
                  "FFN_GEMM cuda=%llu cpu=%llu per_token (host_ffn_binds must be 0)",
                  static_cast<unsigned long long>(g.cuda),
                  static_cast<unsigned long long>(g.cpu));
    return buf;
}

inline std::string format_bench_line(double tok_per_sec, double prefill_s, uint32_t prefill_tok,
                                    uint64_t host_ffn_binds, uint64_t cuda_ffn_binds,
                                    uint64_t h2d_per_tok, uint64_t kv20k_bytes) {
    char buf[384];
    std::snprintf(buf, sizeof(buf),
                  "BENCH tok/s=%.2f prefill_s=%.2f prefill_tok=%u host_ffn_binds=%llu "
                  "cuda_ffn_binds=%llu h2d_B/tok=%llu kv20k_MiB=%.1f",
                  tok_per_sec, prefill_s, prefill_tok,
                  static_cast<unsigned long long>(host_ffn_binds),
                  static_cast<unsigned long long>(cuda_ffn_binds),
                  static_cast<unsigned long long>(h2d_per_tok),
                  static_cast<double>(kv20k_bytes) / (1024.0 * 1024.0));
    return buf;
}

}  // namespace micro_llm
