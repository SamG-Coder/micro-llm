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
// the sched never sees are not a bind (5080: h2d_B=0). Load-time copy into
// the CUDA buffer the tensor uses, or op_offload H2D of CPU Q4 into A/B,
// is the supported path. Park-all-64 is illegal.
inline constexpr bool ggml_can_rebind_q4_midgraph() { return false; }

}  // namespace micro_llm
