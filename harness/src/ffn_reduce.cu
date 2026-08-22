// Optional CUDA kernel. Compiles only when MICRO_LLM_CUDA=ON and nvcc exists.
// Persistent context: scratch allocated once. Device-tap uses live d_gate/d_up
// with no per-token cudaMalloc or H2D of activations.

#include "micro_llm/ffn_reduce.hpp"

#include <cuda_runtime.h>

#include <vector>

namespace micro_llm {
namespace {

__device__ __forceinline__ float silu_dev(float x) {
    return x / (1.f + expf(-x));
}

__global__ void ffn_reduce_token_kernel(const float* gate, const float* up,
                                        float* abs_out, uint8_t* fired_bits,
                                        uint32_t n_channels, float eps,
                                        unsigned int* n_fired) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_channels) {
        return;
    }
    const float act = silu_dev(gate[c]) * up[c];
    const float a = fabsf(act);
    if (abs_out) {
        abs_out[c] = a;
    }
    if (a > eps) {
        atomicAdd(n_fired, 1u);
        if (fired_bits) {
            atomicOr(reinterpret_cast<unsigned int*>(fired_bits + (c >> 3)),
                     1u << (c & 7u));
        }
    }
}

// Hour path: GPU accumulators. No 17408-float D2H.
__global__ void ffn_accum_token_kernel(const float* gate, const float* up,
                                       uint32_t* n_fired, float* sumsq, float* maxabs,
                                       uint8_t* token_bits, uint32_t layer,
                                       uint32_t n_channels, float eps) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_channels) {
        return;
    }
    const float act = silu_dev(gate[c]) * up[c];
    const float a = fabsf(act);
    if (a <= eps) {
        return;
    }
    const size_t idx = static_cast<size_t>(layer) * n_channels + c;
    atomicAdd(n_fired + idx, 1u);
    atomicAdd(sumsq + idx, a * a);
    // maxabs: CAS loop
    int* as_i = reinterpret_cast<int*>(maxabs + idx);
    int old = *as_i;
    while (true) {
        const float old_f = __int_as_float(old);
        if (a <= old_f) {
            break;
        }
        const int assumed = old;
        old = atomicCAS(as_i, assumed, __float_as_int(a));
        if (old == assumed) {
            break;
        }
    }
    if (token_bits) {
        const size_t bit = static_cast<size_t>(layer) * n_channels + c;
        atomicOr(reinterpret_cast<unsigned int*>(token_bits + (bit >> 3)),
                 1u << (bit & 7u));
    }
}

void cuda_free_ptr(void*& p) {
    if (p) {
        cudaFree(p);
        p = nullptr;
    }
}

}  // namespace

CudaReduceContext::~CudaReduceContext() { release(); }

bool CudaReduceContext::ensure(uint32_t n_channels) {
    if (n_channels == 0) {
        return false;
    }
    if (cap_ >= n_channels && d_abs_ && d_nf_) {
        return true;
    }
    release();
    const size_t fbytes = sizeof(float) * n_channels;
    const size_t bbytes = (static_cast<size_t>(n_channels) + 7u) / 8u;
    float* abs = nullptr;
    uint8_t* bits = nullptr;
    unsigned int* nf = nullptr;
    float* gscratch = nullptr;
    float* uscratch = nullptr;
    if (cudaMalloc(&abs, fbytes) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&bits, bbytes) != cudaSuccess) {
        cudaFree(abs);
        return false;
    }
    if (cudaMalloc(&nf, sizeof(unsigned int)) != cudaSuccess) {
        cudaFree(abs);
        cudaFree(bits);
        return false;
    }
    if (cudaMalloc(&gscratch, fbytes) != cudaSuccess) {
        cudaFree(abs);
        cudaFree(bits);
        cudaFree(nf);
        return false;
    }
    if (cudaMalloc(&uscratch, fbytes) != cudaSuccess) {
        cudaFree(abs);
        cudaFree(bits);
        cudaFree(nf);
        cudaFree(gscratch);
        return false;
    }
    d_abs_ = abs;
    d_bits_ = bits;
    d_nf_ = nf;
    d_gate_scratch_ = gscratch;
    d_up_scratch_ = uscratch;
    cap_ = n_channels;
    return true;
}

void CudaReduceContext::release() {
    cuda_free_ptr(d_abs_);
    cuda_free_ptr(d_bits_);
    cuda_free_ptr(d_nf_);
    cuda_free_ptr(d_gate_scratch_);
    cuda_free_ptr(d_up_scratch_);
    cuda_free_ptr(d_n_fired_);
    cuda_free_ptr(d_sumsq_);
    cuda_free_ptr(d_maxabs_);
    cuda_free_ptr(d_token_bits_);
    cap_ = 0;
}

bool CudaReduceContext::ensure_accums() {
    if (!ensure(kFfnIntermediate)) {
        return false;
    }
    if (d_n_fired_ && d_sumsq_ && d_maxabs_ && d_token_bits_) {
        return true;
    }
    const size_t n = static_cast<size_t>(kNLayers) * kFfnIntermediate;
    uint32_t* nf = nullptr;
    float* ss = nullptr;
    float* mx = nullptr;
    uint8_t* bits = nullptr;
    if (cudaMalloc(&nf, n * sizeof(uint32_t)) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&ss, n * sizeof(float)) != cudaSuccess) {
        cudaFree(nf);
        return false;
    }
    if (cudaMalloc(&mx, n * sizeof(float)) != cudaSuccess) {
        cudaFree(nf);
        cudaFree(ss);
        return false;
    }
    if (cudaMalloc(&bits, kFloorBitsetBytes) != cudaSuccess) {
        cudaFree(nf);
        cudaFree(ss);
        cudaFree(mx);
        return false;
    }
    cudaMemset(nf, 0, n * sizeof(uint32_t));
    cudaMemset(ss, 0, n * sizeof(float));
    cudaMemset(mx, 0, n * sizeof(float));
    cudaMemset(bits, 0, kFloorBitsetBytes);
    d_n_fired_ = nf;
    d_sumsq_ = ss;
    d_maxabs_ = mx;
    d_token_bits_ = bits;
    return true;
}

void CudaReduceContext::begin_token_device() {
    if (d_token_bits_) {
        cudaMemsetAsync(d_token_bits_, 0, kFloorBitsetBytes, 0);
    }
}

int CudaReduceContext::accum_device(uint32_t layer, const float* d_gate, const float* d_up,
                                    uint32_t n_channels, float eps) {
    if (!d_gate || !d_up || layer >= kNLayers || n_channels == 0) {
        return -1;
    }
    if (!ensure_accums()) {
        return -1;
    }
    const int threads = 128;
    const int blocks = static_cast<int>((n_channels + threads - 1u) / threads);
    ffn_accum_token_kernel<<<blocks, threads>>>(
        d_gate, d_up, static_cast<uint32_t*>(d_n_fired_), static_cast<float*>(d_sumsq_),
        static_cast<float*>(d_maxabs_), static_cast<uint8_t*>(d_token_bits_), layer,
        n_channels, eps);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

bool CudaReduceContext::async_d2h_bitset(uint8_t* host_bits) {
    if (!host_bits || !d_token_bits_) {
        return false;
    }
    return cudaMemcpyAsync(host_bits, d_token_bits_, kFloorBitsetBytes,
                           cudaMemcpyDeviceToHost, 0) == cudaSuccess;
}

bool CudaReduceContext::sync_d2h() { return cudaDeviceSynchronize() == cudaSuccess; }

bool CudaReduceContext::d2h_layer_accums(uint32_t layer, uint64_t* n_fired, float* sumsq,
                                         float* maxabs, uint32_t n_channels) {
    if (!n_fired || !sumsq || !maxabs || layer >= kNLayers || !d_n_fired_) {
        return false;
    }
    std::vector<uint32_t> nf(n_channels, 0);
    const size_t off = static_cast<size_t>(layer) * kFfnIntermediate;
    if (cudaMemcpy(nf.data(), static_cast<uint32_t*>(d_n_fired_) + off,
                   n_channels * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(sumsq, static_cast<float*>(d_sumsq_) + off, n_channels * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(maxabs, static_cast<float*>(d_maxabs_) + off, n_channels * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    for (uint32_t c = 0; c < n_channels; ++c) {
        n_fired[c] = nf[c];
    }
    return true;
}

int CudaReduceContext::reduce_device(const float* d_gate, const float* d_up,
                                     float* abs_out, uint8_t* fired_bits,
                                     uint32_t n_channels, float eps) {
    if (!d_gate || !d_up || !abs_out || n_channels == 0) {
        return -1;
    }
    if (!ensure(n_channels)) {
        return -1;
    }
    const size_t fbytes = sizeof(float) * n_channels;
    const size_t bbytes = (static_cast<size_t>(n_channels) + 7u) / 8u;
    cudaMemset(d_nf_, 0, sizeof(unsigned int));
    if (fired_bits) {
        cudaMemset(d_bits_, 0, bbytes);
    }
    const int threads = 128;
    const int blocks = static_cast<int>((n_channels + threads - 1u) / threads);
    ffn_reduce_token_kernel<<<blocks, threads>>>(
        d_gate, d_up, static_cast<float*>(d_abs_),
        fired_bits ? static_cast<uint8_t*>(d_bits_) : nullptr, n_channels, eps,
        static_cast<unsigned int*>(d_nf_));
    if (cudaDeviceSynchronize() != cudaSuccess) {
        return -1;
    }
    cudaMemcpy(abs_out, d_abs_, fbytes, cudaMemcpyDeviceToHost);
    unsigned int nf = 0;
    cudaMemcpy(&nf, d_nf_, sizeof(nf), cudaMemcpyDeviceToHost);
    if (fired_bits) {
        cudaMemcpy(fired_bits, d_bits_, bbytes, cudaMemcpyDeviceToHost);
    }
    return static_cast<int>(nf);
}

int CudaReduceContext::reduce_host(const float* gate, const float* up, float* abs_out,
                                   uint8_t* fired_bits, uint32_t n_channels,
                                   float eps) {
    if (!gate || !up || !abs_out || n_channels == 0) {
        return -1;
    }
    if (!ensure(n_channels)) {
        return -1;
    }
    const size_t fbytes = sizeof(float) * n_channels;
    cudaMemcpy(d_gate_scratch_, gate, fbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_up_scratch_, up, fbytes, cudaMemcpyHostToDevice);
    return reduce_device(static_cast<const float*>(d_gate_scratch_),
                         static_cast<const float*>(d_up_scratch_), abs_out,
                         fired_bits, n_channels, eps);
}

CudaReduceContext& persistent_cuda_reduce() {
    static CudaReduceContext ctx;
    return ctx;
}

bool ffn_reduce_cuda_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

ReduceBackend ffn_reduce_backend() {
    return ffn_reduce_cuda_available() ? ReduceBackend::Cuda : ReduceBackend::Cpu;
}

int ffn_reduce_token_cuda(const float* gate, const float* up, float* abs_out,
                          uint8_t* fired_bits, uint32_t n_channels, float eps) {
    if (!ffn_reduce_cuda_available()) {
        return -1;
    }
    return persistent_cuda_reduce().reduce_host(gate, up, abs_out, fired_bits,
                                                n_channels, eps);
}

int ffn_reduce_token_cuda_device(const float* d_gate, const float* d_up,
                                 float* abs_out, uint8_t* fired_bits,
                                 uint32_t n_channels, float eps) {
    if (!ffn_reduce_cuda_available()) {
        return -1;
    }
    return persistent_cuda_reduce().reduce_device(d_gate, d_up, abs_out, fired_bits,
                                                  n_channels, eps);
}

}  // namespace micro_llm
