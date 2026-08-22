#pragma once

// Milestone 1 card + split + GEMM ledgers. Slots A/B are reserved FIRST,
// then leftover under 14 GiB after KV@20k parks FFN 0–56 only. Layers
// 57–63 stream through A/B. Park 64 is illegal. Park weights, not KV.
// Compile-tested without llama.cpp.

#include "micro_llm/perf.hpp"
#include "micro_llm/types.hpp"

#include <cstdint>
#include <cstdio>
#include <string>

namespace micro_llm {

struct VramLedger {
    uint64_t ga_bytes = kPinnedGaWeightBytes;
    uint64_t kv20k_bytes = kHourKvReserveBytes;
    uint64_t scratch_bytes = kCudaScratchBytes;  // 0.9 GiB
    uint64_t graph_reserve_bytes = kHourGraphReserveBytes;  // before extra park
    uint64_t slot_a_bytes = 0;
    uint64_t slot_b_bytes = 0;
    uint64_t parked_ffn_bytes = 0;
    uint64_t usable_bytes = kServeUsableBytes;  // 15.2 GiB hard
    uint32_t n_parked_ffn = 0;
    uint32_t n_streamed_ffn = 0;
};

inline constexpr uint64_t vram_ledger_used(const VramLedger& v) {
    return v.ga_bytes + v.kv20k_bytes + v.scratch_bytes + v.graph_reserve_bytes + v.slot_a_bytes +
           v.slot_b_bytes + v.parked_ffn_bytes;
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
    v.slot_a_bytes = kStreamSlotBytes;  // full layer ~160 MiB, not 80
    v.slot_b_bytes = kStreamSlotBytes;
    v.graph_reserve_bytes = kHourGraphReserveBytes;  // BEFORE any extra park
    const uint64_t fixed = v.ga_bytes + v.kv20k_bytes + v.scratch_bytes + v.graph_reserve_bytes +
                           v.slot_a_bytes + v.slot_b_bytes;
    uint32_t n_park = 0;
    if (fixed < soft && layer_bytes != 0) {
        n_park = static_cast<uint32_t>((soft - fixed) / layer_bytes);
        if (n_park > kMaxParkedFfnLayers) {
            n_park = kMaxParkedFfnLayers;  // 57. never 64.
        }
    }
    v.n_parked_ffn = n_park;
    v.n_streamed_ffn = ffn_stream_layers(n_park);
    v.parked_ffn_bytes = static_cast<uint64_t>(n_park) * layer_bytes;
    return v;
}

// After measure: slot_A/B are that layer’s bytes, not the 160 floor.
// Graph reserve + CUDA0 13110 stay on the ledger BEFORE park. Do not
// park overflow (57–63 stay streamed). extra_park stays 0.
inline constexpr VramLedger vram_ledger_measured_slots(uint64_t slot_a, uint64_t slot_b) {
    VramLedger v = vram_ledger_slots_first();
    if (slot_a != 0) {
        v.slot_a_bytes = slot_a;
    }
    if (slot_b != 0) {
        v.slot_b_bytes = slot_b;
    }
    return v;
}

inline std::string format_ffn_slot_bytes_line(uint32_t layer, uint64_t gate, uint64_t up,
                                              uint64_t down, uint64_t slot) {
    char buf[384];
    const uint64_t sum = gate + up + down;
    std::snprintf(buf, sizeof(buf),
                  "FFN_SLOT_BYTES layer=%u gate=%llu up=%llu down=%llu sum=%llu "
                  "align=%u slot=%llu floor=%llu (slot==measured+align; grow if >160)",
                  layer, static_cast<unsigned long long>(gate),
                  static_cast<unsigned long long>(up), static_cast<unsigned long long>(down),
                  static_cast<unsigned long long>(sum), kTensorAlign,
                  static_cast<unsigned long long>(slot),
                  static_cast<unsigned long long>(kStreamSlotBytes));
    return buf;
}

inline std::string format_vram_ledger(const VramLedger& v) {
    char buf[704];
    std::snprintf(buf, sizeof(buf),
                  "VRAM_LEDGER ga_MiB=%.1f kv20k_MiB=%.1f scratch_MiB=%.1f "
                  "graph_reserve_MiB=%.1f slot_A_MiB=%.1f slot_B_MiB=%.1f "
                  "parked_ffn_MiB=%.1f used_MiB=%.1f free_to_15_2_MiB=%.1f "
                  "park=%u stream=%u extra_park=0 cuda0_bind_MiB=%.1f",
                  static_cast<double>(v.ga_bytes) / (1024.0 * 1024.0),
                  static_cast<double>(v.kv20k_bytes) / (1024.0 * 1024.0),
                  static_cast<double>(v.scratch_bytes) / (1024.0 * 1024.0),
                  static_cast<double>(v.graph_reserve_bytes) / (1024.0 * 1024.0),
                  static_cast<double>(v.slot_a_bytes) / (1024.0 * 1024.0),
                  static_cast<double>(v.slot_b_bytes) / (1024.0 * 1024.0),
                  static_cast<double>(v.parked_ffn_bytes) / (1024.0 * 1024.0),
                  static_cast<double>(vram_ledger_used(v)) / (1024.0 * 1024.0),
                  static_cast<double>(vram_ledger_free_to_hard(v)) / (1024.0 * 1024.0),
                  v.n_parked_ffn, v.n_streamed_ffn,
                  static_cast<double>(kMeasured5080Cuda0BindMiB));
    return buf;
}

// Graph-split causes. All six print every run. hook/callback MUST be 0
// when cb_eval is nullptr (--trace-off). CUDA_Host_to_CUDA must be 0:
// streamed 57–63 are CPU-resident + H2D into A/B, not CUDA_Host.
struct SplitLedger {
    uint32_t callback_hooks = 0;       // hook/callback
    uint32_t cuda_host_to_cuda = 0;    // CUDA_Host_to_CUDA
    uint32_t backend_transition = 0;   // backend
    uint32_t unsupported_op = 0;
    uint32_t placement_buffer = 0;     // placement/buffer
    uint32_t other = 0;
    bool trace_off = true;
};

inline constexpr uint32_t split_ledger_total(const SplitLedger& s) {
    const uint32_t hooks = s.trace_off ? 0u : s.callback_hooks;
    return hooks + s.cuda_host_to_cuda + s.backend_transition + s.unsupported_op +
           s.placement_buffer + s.other;
}

inline constexpr SplitLedger split_ledger_trace_off_park_stream(
    uint32_t n_stream = kMeasured5080Stream7, uint32_t cuda_host_ffn = 0,
    uint32_t cpu_ffn_srcs = 0) {
    SplitLedger s;
    s.trace_off = true;
    s.callback_hooks = 0;
    s.cuda_host_to_cuda = cuda_host_ffn;
    s.backend_transition = 1;  // CUDA stack -> host lm_head
    s.unsupported_op = 0;
    (void)n_stream;
    // Leftover CPU / CPU_Mapped MUL_MAT srcs (blk.63 ffn_gate/up). 0 after bind.
    s.placement_buffer = cpu_ffn_srcs;
    s.other = 0;
    return s;
}

// Kept name: hour path is park-57 + stream-7, not park-all-64.
inline constexpr SplitLedger split_ledger_trace_off_cuda_ffn() {
    return split_ledger_trace_off_park_stream(kMeasured5080Stream7, 0);
}

inline std::string format_split_ledger(const SplitLedger& s) {
    const uint32_t hooks = s.trace_off ? 0u : s.callback_hooks;
    char buf[384];
    std::snprintf(buf, sizeof(buf),
                  "SPLIT_LEDGER callback/hooks=%u hook/callback=%u "
                  "CUDA_Host_to_CUDA=%u backend_transition=%u unsupported_op=%u "
                  "placement/buffer=%u other=%u total=%u trace_off=%d",
                  hooks, hooks, s.cuda_host_to_cuda, s.backend_transition, s.unsupported_op,
                  s.placement_buffer, s.other, split_ledger_total(s), s.trace_off ? 1 : 0);
    return buf;
}

// Why a graph split exists. Print every kind every run (n may be 0).
enum class SplitCauseKind : uint8_t {
    HookCallback = 0,
    CudaHostToCuda = 1,
    BackendTransition = 2,
    UnsupportedOp = 3,
    Placement = 4,
    Other = 5,
};

inline const char* split_cause_kind_name(SplitCauseKind k) {
    switch (k) {
        case SplitCauseKind::HookCallback:
            return "hook/callback";
        case SplitCauseKind::CudaHostToCuda:
            return "CUDA_Host_to_CUDA";
        case SplitCauseKind::BackendTransition:
            return "backend_transition";
        case SplitCauseKind::UnsupportedOp:
            return "unsupported_op";
        case SplitCauseKind::Placement:
            return "placement/buffer";
        case SplitCauseKind::Other:
            return "other";
    }
    return "other";
}

inline bool name_is_ffn_mul_mat_src(const char* tensor) {
    if (!tensor || tensor[0] == '\0') {
        return false;
    }
    const std::string t(tensor);
    return t.find("ffn_gate") != std::string::npos || t.find("ffn_up") != std::string::npos ||
           t.find("ffn_down") != std::string::npos || t.find("ffn_out") != std::string::npos;
}

// The three Q4 GEMM weights only — not scales, not norms. These must all
// fit in one ~160 MiB slot. 9a5f0df packed extras first; down stayed
// CPU_Mapped.
inline bool name_is_slot_q4_weight(const char* tensor) {
    if (!name_is_ffn_mul_mat_src(tensor)) {
        return false;
    }
    const std::string t(tensor);
    if (t.find("scale") != std::string::npos) {
        return false;
    }
    if (t.size() >= 2 && t.compare(t.size() - 2, 2, "_s") == 0) {
        return false;
    }
    return t.find("ffn_norm") == std::string::npos;
}

inline bool buft_is_cpu_mapped(const char* n) {
    if (!n || n[0] == '\0') {
        return false;
    }
    const std::string s(n);
    return s.find("Mapped") != std::string::npos || s.find("MAPPED") != std::string::npos ||
           s.find("mapped") != std::string::npos;
}

inline bool name_is_ffn_input_norm(const char* tensor) {
    if (!tensor || tensor[0] == '\0') {
        return false;
    }
    const std::string t(tensor);
    return t.find("attn_post_norm") != std::string::npos ||
           t.find("post_attention_norm") != std::string::npos ||
           t.find("ffn_norm") != std::string::npos;
}

inline std::string format_split_cause_line(SplitCauseKind k, uint32_t n, const char* tensor,
                                          const char* buft, int32_t layer) {
    char buf[384];
    std::snprintf(buf, sizeof(buf),
                  "SPLIT_CAUSE kind=%s n=%u tensor=%s buft=%s layer=%d "
                  "(hook/callback must be 0 on --trace-off)",
                  split_cause_kind_name(k), n, tensor && tensor[0] ? tensor : "-",
                  buft && buft[0] ? buft : "-", layer);
    return buf;
}

inline std::string format_split_why_line(uint32_t n_splits, const char* last_tensor,
                                        const char* last_buft) {
    char buf[448];
    std::snprintf(buf, sizeof(buf),
                  "SPLIT_WHY n=%u bs=1 last=%s last_buft=%s "
                  "(532>340 = CPU src/view on blk.63 ffn_gate/up; "
                  "638 = down CPU_Mapped)",
                  n_splits, last_tensor && last_tensor[0] ? last_tensor : "-",
                  last_buft && last_buft[0] ? last_buft : "-");
    return buf;
}

// Name the tensor + backend still off CUDA. view_src is the usual 532 hop:
// weight poked to a slot, parent mmap (CPU_Mapped) still hosts MUL_MAT.
inline std::string format_ffn_hop_line(int32_t layer, const char* tensor, const char* buft,
                                      const char* view_src, const char* view_buft,
                                      SplitCauseKind cause) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "FFN_HOP layer=%d tensor=%s buft=%s view_src=%s view_buft=%s "
                  "cause=%s (532 last crossing = gate/up src or view still host)",
                  layer, tensor && tensor[0] ? tensor : "-", buft && buft[0] ? buft : "-",
                  view_src && view_src[0] ? view_src : "none",
                  view_buft && view_buft[0] ? view_buft : "-", split_cause_kind_name(cause));
    return buf;
}

inline std::string format_split_causes_block(const SplitLedger& s, const char* place_tensor,
                                            const char* place_buft, int32_t place_layer) {
    std::string out;
    out += format_split_cause_line(SplitCauseKind::HookCallback, s.trace_off ? 0u : s.callback_hooks,
                                   "cb_eval", s.trace_off ? "nullptr" : "set", -1);
    out += '\n';
    out += format_split_cause_line(SplitCauseKind::CudaHostToCuda, s.cuda_host_to_cuda, place_tensor,
                                   place_buft, place_layer);
    out += '\n';
    out += format_split_cause_line(SplitCauseKind::BackendTransition, s.backend_transition,
                                   "output.weight", "CPU", -1);
    out += '\n';
    out += format_split_cause_line(SplitCauseKind::UnsupportedOp, s.unsupported_op, "-", "-", -1);
    out += '\n';
    out += format_split_cause_line(SplitCauseKind::Placement, s.placement_buffer, place_tensor,
                                   place_buft, place_layer);
    out += '\n';
    out += format_split_cause_line(SplitCauseKind::Other, s.other, "-", "-", -1);
    return out;
}

// qwen35 build_ffn reads ffn_{gate,up,down} plus optional *_s scales.
// attn_post_norm is the FFN input; a CPU norm pulls MUL_MAT onto host.
inline constexpr const char* kStreamedBindSuffixes[] = {
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
    "ffn_gate.scale",
    "ffn_up.scale",
    "ffn_down.scale",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
    "ffn_out.weight",
    "ffn_out",
    "ffn_gate_s",
    "ffn_up_s",
    "ffn_down_s",
    "ffn_norm.weight",
    "attn_post_norm.weight",
    "post_attention_norm.weight",
};

inline constexpr uint32_t kStreamedBindSuffixCount =
    static_cast<uint32_t>(sizeof(kStreamedBindSuffixes) / sizeof(kStreamedBindSuffixes[0]));

// CUDA_Host before CUDA (the Host substring lives inside some CUDA names).
enum class BuftKind : uint8_t { Cpu = 0, Cuda = 1, CudaHost = 2, Other = 3 };

inline BuftKind classify_backend_buft_name(const char* n) {
    if (!n || n[0] == '\0') {
        return BuftKind::Other;
    }
    std::string s(n);
    if (s.find("Host") != std::string::npos || s.find("HOST") != std::string::npos ||
        s.find("host") != std::string::npos) {
        return BuftKind::CudaHost;
    }
    if (s.find("CUDA") != std::string::npos || s.find("cuda") != std::string::npos ||
        s.find("GPU") != std::string::npos || s.find("gpu") != std::string::npos) {
        return BuftKind::Cuda;
    }
    if (s.find("CPU") != std::string::npos || s.find("cpu") != std::string::npos) {
        return BuftKind::Cpu;
    }
    return BuftKind::Other;
}

inline SplitCauseKind classify_split_cause(const char* tensor, const char* buft, bool trace_off) {
    if (!trace_off) {
        return SplitCauseKind::HookCallback;
    }
    const BuftKind bk = classify_backend_buft_name(buft);
    if (bk == BuftKind::CudaHost) {
        return SplitCauseKind::CudaHostToCuda;
    }
    // CPU_Mapped is a host alias of the GGUF mmap, not an unsupported op
    // and not a CUDA_Host↔CUDA hop. Graph reserve still sees host.
    if (buft_is_cpu_mapped(buft)) {
        return SplitCauseKind::Placement;
    }
    const std::string t = tensor ? tensor : "";
    if (name_is_ffn_mul_mat_src(tensor) || name_is_ffn_input_norm(tensor)) {
        if (bk != BuftKind::Cuda) {
            return SplitCauseKind::Placement;
        }
    }
    if (t.find("output") != std::string::npos && t.find("ffn") == std::string::npos) {
        return SplitCauseKind::BackendTransition;
    }
    if (bk == BuftKind::Cpu) {
        return SplitCauseKind::BackendTransition;
    }
    return SplitCauseKind::Other;
}

inline std::string format_ffn_place_line(uint32_t parked_cuda, uint32_t streamed_cpu,
                                         uint32_t streamed_cuda, uint32_t streamed_cuda_host) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "FFN_PLACE parked_cuda=%u streamed_cpu=%u streamed_cuda=%u "
                  "streamed_cuda_host=%u (want host=0 stream_cuda_gemm=1)",
                  parked_cuda, streamed_cpu, streamed_cuda, streamed_cuda_host);
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

struct BenchLine {
    bool trace_on = false;
    double tok_per_sec = 0.0;
    double decode_s = 0.0;
    double prefill_s = 0.0;
    uint32_t prefill_tok = 0;
    uint64_t host_ffn_binds = 0;
    uint64_t cuda_ffn_binds = 0;
    uint64_t cuda0_model = 0;
    uint64_t cuda0_compute = 0;
    uint64_t nvidia_used = 0;
    uint64_t h2d_per_tok = 0;
    uint64_t kv20k_bytes = kHourKvReserveBytes;
    uint32_t real_h2d = 0;
};

inline constexpr bool bench_swap_7780(double tok_per_sec) {
    return tok_per_sec > static_cast<double>(kMeasured7780TokPerSec192);
}

inline BenchLine bench_from_snapshot(const PerfSnapshot& s) {
    BenchLine b;
    b.trace_on = !s.trace_off;
    b.tok_per_sec = s.tok_per_sec;
    b.decode_s = s.decode_s;
    b.prefill_s = s.prefill_s;
    b.prefill_tok = s.prefill_tok;
    b.host_ffn_binds = s.host_ffn_binds;
    b.cuda_ffn_binds = s.cuda_ffn_binds;
    b.cuda0_model = s.cuda0_model;
    b.cuda0_compute = s.cuda0_compute;
    b.nvidia_used = s.nvidia_used;
    b.h2d_per_tok = s.h2d_bytes_per_tok;
    b.kv20k_bytes = kHourKvReserveBytes;
    b.real_h2d = (s.real_h2d || s.h2d_bytes > 0) ? 1u : 0u;
    return b;
}

inline std::string format_bench_line(const BenchLine& b) {
    char buf[640];
    std::snprintf(buf, sizeof(buf),
                  "BENCH TRACE=%s tok/s=%.2f decode_s=%.2f prefill_s=%.2f "
                  "prefill_tok=%u host_ffn_binds=%llu cuda_ffn_binds=%llu "
                  "cuda0_model_MiB=%.1f cuda0_compute_MiB=%.1f nvidia_used_MiB=%.1f "
                  "h2d_B/tok=%llu real_h2d=%u kv20k_MiB=%.1f swap_7780=%d",
                  b.trace_on ? "on" : "off", b.tok_per_sec, b.decode_s, b.prefill_s,
                  b.prefill_tok, static_cast<unsigned long long>(b.host_ffn_binds),
                  static_cast<unsigned long long>(b.cuda_ffn_binds),
                  static_cast<double>(b.cuda0_model) / (1024.0 * 1024.0),
                  static_cast<double>(b.cuda0_compute) / (1024.0 * 1024.0),
                  static_cast<double>(b.nvidia_used) / (1024.0 * 1024.0),
                  static_cast<unsigned long long>(b.h2d_per_tok), b.real_h2d,
                  static_cast<double>(b.kv20k_bytes) / (1024.0 * 1024.0),
                  bench_swap_7780(b.tok_per_sec) ? 1 : 0);
    return buf;
}

inline std::string format_bench_line(double tok_per_sec, double prefill_s, uint32_t prefill_tok,
                                    uint64_t host_ffn_binds, uint64_t cuda_ffn_binds,
                                    uint64_t h2d_per_tok, uint64_t kv20k_bytes) {
    BenchLine b;
    b.tok_per_sec = tok_per_sec;
    b.prefill_s = prefill_s;
    b.prefill_tok = prefill_tok;
    b.host_ffn_binds = host_ffn_binds;
    b.cuda_ffn_binds = cuda_ffn_binds;
    b.h2d_per_tok = h2d_per_tok;
    b.kv20k_bytes = kv20k_bytes;
    return format_bench_line(b);
}

}  // namespace micro_llm
