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

// gate, up: n_channels live activations for ONE token (not weights).
// abs_out:  n_channels, written then owned by the caller who must evict.
// fired_bits: optional bitset, bit c set when |act| > eps. May be null.
// Returns the number of fired channels.
uint32_t ffn_reduce_token(const float* gate, const float* up, float* abs_out,
                          uint8_t* fired_bits, uint32_t n_channels, float eps);

uint32_t ffn_reduce_token_cpu(const float* gate, const float* up, float* abs_out,
                              uint8_t* fired_bits, uint32_t n_channels,
                              float eps);

// CUDA launch. Returns -1 if CUDA is not built / not usable.
int ffn_reduce_token_cuda(const float* gate, const float* up, float* abs_out,
                          uint8_t* fired_bits, uint32_t n_channels, float eps);

}  // namespace micro_llm
