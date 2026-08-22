#pragma once

// Per-token FFN reduce: tap |SiLU(gate)*up| from the live forward.
// Do NOT dequant a whole FFN to FP16.
// Do NOT materialize a [chunk x 17408] scratch.
// Per-token warp reduce, then evict.
//
// CUDA kernel is optional. Without nvcc this is the CPU fallback.

#include "micro_llm/types.hpp"

#include <cmath>
#include <cstdint>

namespace micro_llm {

inline float silu(float x) {
    // silu(x) = x * sigmoid(x)
    return x / (1.f + std::exp(-x));
}

enum class ReduceBackend : uint8_t { Cpu = 0, Cuda = 1 };

bool ffn_reduce_cuda_available();
ReduceBackend ffn_reduce_backend();

// Persistent CUDA scratch for the hour. Allocated once; never cudaMalloc of
// gate/up per token. Device-tap uses the caller's live d_gate/d_up.
class CudaReduceContext {
public:
    CudaReduceContext() = default;
    ~CudaReduceContext();
    CudaReduceContext(const CudaReduceContext&) = delete;
    CudaReduceContext& operator=(const CudaReduceContext&) = delete;

    bool ensure(uint32_t n_channels);
    void release();
    bool ready() const { return cap_ != 0; }
    uint32_t capacity() const { return cap_; }

    // d_gate / d_up are live device pointers. No H2D of activations.
    int reduce_device(const float* d_gate, const float* d_up, float* abs_out,
                      uint8_t* fired_bits, uint32_t n_channels, float eps);

    // Async: launch on a private stream. No device-wide sync. Call
    // sync_stream() once per token before reading host outputs.
    int reduce_device_async(const float* d_gate, const float* d_up, float* abs_out,
                            uint8_t* fired_bits, uint32_t n_channels, float eps);
    bool sync_stream();
    uint64_t d2h_bytes() const { return d2h_bytes_; }
    uint64_t sync_count() const { return sync_count_; }

    // Host activations: persistent device scratch (not per-token malloc).
    int reduce_host(const float* gate, const float* up, float* abs_out,
                    uint8_t* fired_bits, uint32_t n_channels, float eps);

private:
    void* d_abs_ = nullptr;
    void* d_bits_ = nullptr;
    void* d_nf_ = nullptr;
    void* d_gate_scratch_ = nullptr;
    void* d_up_scratch_ = nullptr;
    void* stream_ = nullptr;  // cudaStream_t
    uint32_t cap_ = 0;
    uint64_t d2h_bytes_ = 0;
    uint64_t sync_count_ = 0;
};

CudaReduceContext& persistent_cuda_reduce();

// gate, up: n_channels live activations for ONE token (not weights).
// abs_out:  n_channels, written then owned by the caller who must evict.
// fired_bits: optional bitset, bit c set when |act| > eps. May be null.
// Returns the number of fired channels.
uint32_t ffn_reduce_token(const float* gate, const float* up, float* abs_out,
                          uint8_t* fired_bits, uint32_t n_channels, float eps);

uint32_t ffn_reduce_token_cpu(const float* gate, const float* up, float* abs_out,
                              uint8_t* fired_bits, uint32_t n_channels,
                              float eps);

// CUDA launch (host pointers). Uses persistent scratch. Returns -1 if no CUDA.
int ffn_reduce_token_cuda(const float* gate, const float* up, float* abs_out,
                          uint8_t* fired_bits, uint32_t n_channels, float eps);

// CUDA launch on live device pointers. No per-token malloc/H2D of gate/up.
int ffn_reduce_token_cuda_device(const float* d_gate, const float* d_up,
                                 float* abs_out, uint8_t* fired_bits,
                                 uint32_t n_channels, float eps);

}  // namespace micro_llm
