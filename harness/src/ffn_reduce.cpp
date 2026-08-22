#include "micro_llm/ffn_reduce.hpp"

#include <cmath>

namespace micro_llm {

uint32_t ffn_reduce_token_cpu(const float* gate, const float* up, float* abs_out,
                              uint8_t* fired_bits, uint32_t n_channels, float eps) {
    uint32_t n_fired = 0;
    for (uint32_t c = 0; c < n_channels; ++c) {
        // Same channel index across gate and up. Down uses this index at export.
        const float act = silu(gate[c]) * up[c];
        const float a = std::fabs(act);
        abs_out[c] = a;
        if (a > eps) {
            ++n_fired;
            if (fired_bits) {
                bit_set(fired_bits, c);
            }
        }
    }
    return n_fired;
}

uint32_t ffn_reduce_token(const float* gate, const float* up, float* abs_out,
                          uint8_t* fired_bits, uint32_t n_channels, float eps) {
#if defined(MICRO_LLM_HAS_CUDA)
    if (ffn_reduce_cuda_available()) {
        const int rc =
            ffn_reduce_token_cuda(gate, up, abs_out, fired_bits, n_channels, eps);
        if (rc >= 0) {
            return static_cast<uint32_t>(rc);
        }
    }
#endif
    return ffn_reduce_token_cpu(gate, up, abs_out, fired_bits, n_channels, eps);
}

#if !defined(MICRO_LLM_HAS_CUDA)
bool ffn_reduce_cuda_available() { return false; }

int ffn_reduce_token_cuda(const float* /*gate*/, const float* /*up*/, float* /*abs_out*/,
                          uint8_t* /*fired_bits*/, uint32_t /*n_channels*/,
                          float /*eps*/) {
    return -1;
}

int ffn_reduce_token_cuda_device(const float* /*d_gate*/, const float* /*d_up*/,
                                 float* /*abs_out*/, uint8_t* /*fired_bits*/,
                                 uint32_t /*n_channels*/, float /*eps*/) {
    return -1;
}

ReduceBackend ffn_reduce_backend() { return ReduceBackend::Cpu; }

CudaReduceContext::~CudaReduceContext() = default;

bool CudaReduceContext::ensure(uint32_t /*n_channels*/) { return false; }

void CudaReduceContext::release() {}

int CudaReduceContext::reduce_device(const float*, const float*, float*, uint8_t*,
                                     uint32_t, float) {
    return -1;
}

int CudaReduceContext::reduce_device_async(const float*, const float*, float*, uint8_t*,
                                           uint32_t, float) {
    return -1;
}

bool CudaReduceContext::sync_stream() { return false; }

int CudaReduceContext::reduce_host(const float*, const float*, float*, uint8_t*,
                                   uint32_t, float) {
    return -1;
}

CudaReduceContext& persistent_cuda_reduce() {
    static CudaReduceContext ctx;
    return ctx;
}
#endif

}  // namespace micro_llm
