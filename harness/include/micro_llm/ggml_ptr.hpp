#pragma once

// ggml CUDA tensors are buffer-base + offset, not a raw float*.
// Passing t->data (an alloc offset) to a CUDA kernel AVs. Resolve first.
// Compile-tested without llama.cpp.

#include "micro_llm/types.hpp"

#include <cstddef>
#include <cstdint>

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

// Streamed layer → ggml slot. 63 → A, 62 → B. Other overflow stays CPU
// (op_offload stream). 2 (extra park buffer) is illegal. -1 = parked.
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
