// Optional CUDA kernel. Compiles only when MICRO_LLM_CUDA=ON and nvcc exists.
// Per-token warp map over channels. No [chunk x 17408] allocation.
// Do not dequant a whole FFN to FP16 ? gate/up are live activations.

#include "micro_llm/ffn_reduce.hpp"

#include <cuda_runtime.h>

namespace micro_llm {
namespace {

__device__ __forceinline__ float silu_dev(float x) {
    return x / (1.f + expf(-x));
}

__global__ void ffn_reduce_token_kernel(const float* gate, const float* up,
                                        float* abs_out, uint8_t* fired_bits,
                                        uint32_t n_channels, float eps,
                                        unsigned int* n_fired) {
    // One thread per channel. A warp covers 32 consecutive channels, then
    // the caller evicts. No chunk dimension.
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_channels) {
        return;
    }
    const float act = silu_dev(gate[c]) * up[c];
    const float a = fabsf(act);
    abs_out[c] = a;
    if (a > eps) {
        atomicAdd(n_fired, 1u);
        if (fired_bits) {
            atomicOr(reinterpret_cast<unsigned int*>(fired_bits + (c >> 3)),
                     1u << (c & 7u));
        }
    }
}

}  // namespace

bool ffn_reduce_cuda_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

ReduceBackend ffn_reduce_backend() {
    return ffn_reduce_cuda_available() ? ReduceBackend::Cuda : ReduceBackend::Cpu;
}

int ffn_reduce_token_cuda(const float* gate, const float* up, float* abs_out,
                          uint8_t* fired_bits, uint32_t n_channels, float eps) {
    if (!gate || !up || !abs_out || n_channels == 0) {
        return -1;
    }

    float* d_gate = nullptr;
    float* d_up = nullptr;
    float* d_abs = nullptr;
    uint8_t* d_bits = nullptr;
    unsigned int* d_nf = nullptr;
    const size_t fbytes = sizeof(float) * n_channels;
    const size_t bbytes = (static_cast<size_t>(n_channels) + 7u) / 8u;

    auto fail = [&]() {
        cudaFree(d_gate);
        cudaFree(d_up);
        cudaFree(d_abs);
        cudaFree(d_bits);
        cudaFree(d_nf);
        return -1;
    };

    if (cudaMalloc(&d_gate, fbytes) != cudaSuccess) return fail();
    if (cudaMalloc(&d_up, fbytes) != cudaSuccess) return fail();
    if (cudaMalloc(&d_abs, fbytes) != cudaSuccess) return fail();
    if (cudaMalloc(&d_nf, sizeof(unsigned int)) != cudaSuccess) return fail();
    if (fired_bits) {
        if (cudaMalloc(&d_bits, bbytes) != cudaSuccess) return fail();
        cudaMemset(d_bits, 0, bbytes);
    }
    cudaMemset(d_nf, 0, sizeof(unsigned int));
    cudaMemcpy(d_gate, gate, fbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_up, up, fbytes, cudaMemcpyHostToDevice);

    const int threads = 128;  // 4 warps; each warp maps 32 channels
    const int blocks = static_cast<int>((n_channels + threads - 1u) / threads);
    ffn_reduce_token_kernel<<<blocks, threads>>>(d_gate, d_up, d_abs, d_bits,
                                                 n_channels, eps, d_nf);
    if (cudaDeviceSynchronize() != cudaSuccess) return fail();

    cudaMemcpy(abs_out, d_abs, fbytes, cudaMemcpyDeviceToHost);
    unsigned int nf = 0;
    cudaMemcpy(&nf, d_nf, sizeof(nf), cudaMemcpyDeviceToHost);
    if (fired_bits && d_bits) {
        cudaMemcpy(fired_bits, d_bits, bbytes, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_abs);
    cudaFree(d_bits);
    cudaFree(d_nf);
    return static_cast<int>(nf);
}

}  // namespace micro_llm
