#pragma once

// ggml CUDA tensors are buffer-base + offset, not a raw float*.
// Passing t->data (an alloc offset) to a CUDA kernel AVs. Resolve first.
// Compile-tested without llama.cpp.

#include "micro_llm/types.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

namespace micro_llm {

// ggml often stores the alloc offset in tensor->data before / instead of a VA.
// A 1 MiB cutoff is NOT enough: a Q4 tensor at 80 MiB into a 10 GiB buffer
// has offset 0x5000000, which is still an offset, not a device VA.
inline constexpr uintptr_t kMinDeviceVaHint = 0x100000ull;

inline bool ptr_looks_like_integer_offset(const void* p,
                                         size_t buf_size = kMinDeviceVaHint) {
    const auto u = reinterpret_cast<uintptr_t>(p);
    return p == nullptr || (buf_size != 0 && u < buf_size);
}

// Map (base, view_offs, optional data) to a pointer inside the buffer.
// data < buf_size is an alloc offset (even when > 1 MiB). A real VA is
// accepted only if it sits inside [base, base+buf_size). Does not D2H.
inline const float* resolve_f32_in_buffer(void* base, size_t buf_size, void* data,
                                          size_t view_offs, size_t nbytes, bool* ok) {
    if (ok) {
        *ok = false;
    }
    if (!base || nbytes == 0 || buf_size < nbytes) {
        return nullptr;
    }
    char* b = static_cast<char*>(base);
    char* p = nullptr;
    const auto u = reinterpret_cast<uintptr_t>(data);

    if (data && u >= buf_size) {
        char* d = static_cast<char*>(data);
        if (d >= b && d + nbytes <= b + buf_size) {
            p = d;
        }
    }
    if (!p) {
        size_t off = view_offs;
        if (data && u < buf_size) {
            off = u;
            if (view_offs != 0 && u != view_offs && u + view_offs + nbytes <= buf_size) {
                off = u + view_offs;
            }
        }
        if (off + nbytes > buf_size) {
            return nullptr;
        }
        p = b + off;
    }
    if (p < b || p + nbytes > b + buf_size) {
        return nullptr;
    }
    if (ok) {
        *ok = true;
    }
    return reinterpret_cast<const float*>(p);
}

// ggml cannot rebind a loaded Q4 tensor onto a different CUDA buffer mid-graph
// (the sched pins buffer type at graph build). Private cudaMalloc slots that
// the sched never sees are not a bind (5080: h2d_B=0). Load-time
// ggml_backend_buffer_init_tensor + ggml_backend_tensor_set onto the slot
// the MUL_MAT reads is the bind. Park-all-64 is illegal.
inline constexpr bool ggml_can_rebind_q4_midgraph() { return false; }

// Load-time attach is allowed. Mid-graph share of 2 slots across 7 Q4
// layers is not (see ggml_can_rebind_q4_midgraph).
inline constexpr bool ggml_can_bind_q4_at_load() { return true; }

inline constexpr bool ggml_slot_pack_ok(uint64_t off, uint64_t nbytes, uint64_t cap) {
    return nbytes != 0 && cap != 0 && off <= cap && nbytes <= cap - off;
}

// After a CUDA rebind, t->data may still be a GGUF mmap VA or an alloc
// offset. Passing either to a kernel AVs (0xC0000005). Classify first.
enum class TensorDataKind : uint8_t {
    None = 0,
    IntegerOffset = 1,
    InBuffer = 2,
    StaleHost = 3,
};

inline const char* tensor_data_kind_name(TensorDataKind k) {
    switch (k) {
        case TensorDataKind::None:
            return "none";
        case TensorDataKind::IntegerOffset:
            return "integer_offset";
        case TensorDataKind::InBuffer:
            return "in_buffer";
        case TensorDataKind::StaleHost:
            return "stale_host";
    }
    return "none";
}

inline TensorDataKind classify_tensor_data_ptr(const void* data, const void* buf_base,
                                              size_t buf_size) {
    if (!data) {
        return TensorDataKind::None;
    }
    if (ptr_looks_like_integer_offset(data, buf_size != 0 ? buf_size : kMinDeviceVaHint)) {
        return TensorDataKind::IntegerOffset;
    }
    if (buf_base && buf_size != 0) {
        const auto d = reinterpret_cast<uintptr_t>(data);
        const auto b = reinterpret_cast<uintptr_t>(buf_base);
        if (d >= b && d < b + buf_size) {
            return TensorDataKind::InBuffer;
        }
    }
    return TensorDataKind::StaleHost;
}

inline bool tensor_data_is_av_risk(TensorDataKind k) {
    return k == TensorDataKind::IntegerOffset || k == TensorDataKind::StaleHost ||
           k == TensorDataKind::None;
}

// The one A/B bind must leave this, not an alloc offset. MUL_MAT that
// treats t->data as a raw pointer AVs on the offset (first-decode class).
inline void* device_va_from_buffer(void* base, size_t off) {
    return base ? static_cast<char*>(base) + off : nullptr;
}

// Print-only. Name the reserve 334/642 node: op + each src buft + whether
// data is a host ptr or a ggml CUDA alloc offset. Not a split-ledger line.
inline std::string format_reserve_av_node_line(const char* tag, const char* name, const char* op,
                                              const char* buft, const char* data_kind,
                                              uintptr_t data_u, const char* src0,
                                              const char* src0_buft, const char* src0_data,
                                              const char* src1, const char* src1_buft,
                                              const char* src1_data) {
    char buf[768];
    std::snprintf(buf, sizeof(buf),
                  "RESERVE_AV_NODE tag=%s name=%s op=%s buft=%s data=%s data_u=0x%llx "
                  "src0=%s src0_buft=%s src0_data=%s src1=%s src1_buft=%s src1_data=%s",
                  tag && tag[0] ? tag : "-", name && name[0] ? name : "-",
                  op && op[0] ? op : "-", buft && buft[0] ? buft : "-",
                  data_kind && data_kind[0] ? data_kind : "-",
                  static_cast<unsigned long long>(data_u), src0 && src0[0] ? src0 : "none",
                  src0_buft && src0_buft[0] ? src0_buft : "-",
                  src0_data && src0_data[0] ? src0_data : "-", src1 && src1[0] ? src1 : "none",
                  src1_buft && src1_buft[0] ? src1_buft : "-",
                  src1_data && src1_data[0] ? src1_data : "-");
    return buf;
}

// James 974b0c3 named the AV: #638 CUDA0 KV, #639 CPU residual,
// #640 CUDA0 layer-norm, #641 CPU output_norm 180K. Print those
// four. Not a split-ledger rewrite.
inline std::string format_tail_split_line(uint32_t n, const char* name, const char* op,
                                         const char* buft, const char* data_kind,
                                         const char* src0, const char* src0_buft,
                                         const char* src0_data, const char* src1,
                                         const char* src1_buft, const char* src1_data) {
    char buf[768];
    std::snprintf(buf, sizeof(buf),
                  "TAIL_SPLIT n=%u name=%s op=%s buft=%s data=%s "
                  "src0=%s src0_buft=%s src0_data=%s src1=%s src1_buft=%s src1_data=%s",
                  n, name && name[0] ? name : "-", op && op[0] ? op : "-",
                  buft && buft[0] ? buft : "-", data_kind && data_kind[0] ? data_kind : "-",
                  src0 && src0[0] ? src0 : "none", src0_buft && src0_buft[0] ? src0_buft : "-",
                  src0_data && src0_data[0] ? src0_data : "-", src1 && src1[0] ? src1 : "none",
                  src1_buft && src1_buft[0] ? src1_buft : "-",
                  src1_data && src1_data[0] ? src1_data : "-");
    return buf;
}

inline std::string format_tail_collapse_line(const char* output_norm_buft,
                                            const char* output_weight_buft) {
    char buf[384];
    std::snprintf(buf, sizeof(buf),
                  "TAIL_COLLAPSE residual->norm->output_norm want=CUDA0 "
                  "output_norm.weight=%s output.weight=%s "
                  "(host lm_head uses a real host buffer; no CPU 180K norm "
                  "after CUDA0 layer-norm; extra_park=0 hooks=0)",
                  output_norm_buft && output_norm_buft[0] ? output_norm_buft : "-",
                  output_weight_buft && output_weight_buft[0] ? output_weight_buft : "-");
    return buf;
}

// One A/B slot = one streamed layer’s gate+up+down. 254e10c: down
// offset past 160. If 160 < aligned sum, the slot MUST grow. Extra
// align slack so down is not flush against the cap.
inline constexpr uint64_t ffn_slot_bytes_for_triplet(uint64_t gate, uint64_t up, uint64_t down,
                                                     uint64_t align = kTensorAlign) {
    if (align == 0) {
        align = 1;
    }
    const uint64_t g = (gate + align - 1u) / align * align;
    const uint64_t u = (up + align - 1u) / align * align;
    const uint64_t d = (down + align - 1u) / align * align;
    return g + u + d + align;
}

// Streamed layer → ggml slot. 63 → A, 62 → B. Other overflow stays CPU
// (op_offload stream). 2 (extra park buffer) is illegal. -1 = parked.
// All three Q4s (gate, up, down) must fit in the slot they use.
inline constexpr int kStreamSlotParked = -1;
inline constexpr int kStreamSlotA = 0;
inline constexpr int kStreamSlotB = 1;
inline constexpr int kStreamSlotCpu = 3;

inline constexpr int ggml_stream_slot_kind(uint32_t layer, uint32_t n_park) {
    if (layer < n_park) {
        return kStreamSlotParked;
    }
    if (layer + 1u == kNLayers) {
        return kStreamSlotA;  // blk.63 GEMM from slot A
    }
    if (layer + 2u == kNLayers) {
        return kStreamSlotB;  // blk.62 in B (N+1 overlap)
    }
    return kStreamSlotCpu;
}

}  // namespace micro_llm
