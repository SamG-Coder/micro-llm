// Optional CUDA kernel. Compiles only when MICRO_LLM_CUDA=ON and nvcc exists.
// Persistent context: scratch allocated once. Device-tap uses live d_gate/d_up
// with no per-token cudaMalloc or H2D of activations.

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
    if (!stream_) {
        cudaStream_t st = nullptr;
        if (cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) == cudaSuccess) {
            stream_ = st;
        }
    }
    return true;
}

void CudaReduceContext::release() {
    if (stream_) {
        cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
        stream_ = nullptr;
    }
    cuda_free_ptr(d_abs_);
    cuda_free_ptr(d_bits_);
    cuda_free_ptr(d_nf_);
    cuda_free_ptr(d_gate_scratch_);
    cuda_free_ptr(d_up_scratch_);
    cap_ = 0;
}

int CudaReduceContext::reduce_device_async(const float* d_gate, const float* d_up,
                                           float* abs_out, uint8_t* fired_bits,
                                           uint32_t n_channels, float eps) {
    if (!d_gate || !d_up || !abs_out || n_channels == 0) {
        return -1;
    }
    if (!ensure(n_channels)) {
        return -1;
    }
    cudaStream_t st = static_cast<cudaStream_t>(stream_);
    const size_t fbytes = sizeof(float) * n_channels;
    const size_t bbytes = (static_cast<size_t>(n_channels) + 7u) / 8u;
    cudaMemsetAsync(d_nf_, 0, sizeof(unsigned int), st);
    if (fired_bits) {
        cudaMemsetAsync(d_bits_, 0, bbytes, st);
    }
    const int threads = 128;
    const int blocks = static_cast<int>((n_channels + threads - 1u) / threads);
    ffn_reduce_token_kernel<<<blocks, threads, 0, st>>>(
        d_gate, d_up, static_cast<float*>(d_abs_),
        fired_bits ? static_cast<uint8_t*>(d_bits_) : nullptr, n_channels, eps,
        static_cast<unsigned int*>(d_nf_));
    cudaMemcpyAsync(abs_out, d_abs_, fbytes, cudaMemcpyDeviceToHost, st);
    d2h_bytes_ += fbytes;
    if (fired_bits) {
        cudaMemcpyAsync(fired_bits, d_bits_, bbytes, cudaMemcpyDeviceToHost, st);
        d2h_bytes_ += bbytes;
    }
    // n_fired is read in sync_stream / reduce_device. Stash on host after sync.
    return 0;
}

bool CudaReduceContext::sync_stream() {
    if (!stream_) {
        return false;
    }
    ++sync_count_;
    return cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)) == cudaSuccess;
}

int CudaReduceContext::reduce_device(const float* d_gate, const float* d_up,
                                     float* abs_out, uint8_t* fired_bits,
                                     uint32_t n_channels, float eps) {
    const int rc = reduce_device_async(d_gate, d_up, abs_out, fired_bits, n_channels, eps);
    if (rc < 0) {
        return -1;
    }
    if (!sync_stream()) {
        return -1;
    }
    unsigned int nf = 0;
    cudaMemcpy(&nf, d_nf_, sizeof(nf), cudaMemcpyDeviceToHost);
    d2h_bytes_ += sizeof(nf);
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
