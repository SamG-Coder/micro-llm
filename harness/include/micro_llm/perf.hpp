#pragma once

// Decode PERFORMANCE telemetry. Real clocks / CUDA events, not guesses.
// Printed at end and periodically. Not in the token hot path except the
// event record itself (cudaEventRecord is cheap).

#include "micro_llm/types.hpp"

#include <cstdint>
#include <string>

namespace micro_llm {

enum class PerfSpan : uint8_t {
    Gpu = 0,
    Pcie = 1,
    Cpu = 2,
    Sync = 3,
    Trace = 4,
};

struct PerfSnapshot {
    uint32_t n_tokens = 0;
    double wall_s = 0.0;
    double tok_per_sec = 0.0;
    double latency_ms = 0.0;  // wall_s * 1000 / n_tokens
    double gpu_ms = 0.0;
    double pcie_ms = 0.0;
    double cpu_ms = 0.0;
    double sync_ms = 0.0;
    double trace_ms = 0.0;
    uint64_t h2d_bytes = 0;
    uint64_t d2h_bytes = 0;
    uint64_t h2d_bytes_per_tok = 0;
    uint64_t d2h_bytes_per_tok = 0;
    uint64_t vram_weights = 0;
    uint64_t vram_kv = 0;
    uint64_t vram_scratch = 0;
    uint64_t vram_free = 0;
    uint32_t n_parked_ffn = 0;
    uint32_t n_streamed_ffn = 0;
    uint64_t cuda_ffn_binds = 0;
    uint64_t host_ffn_binds = 0;
    uint64_t overlap_prefetches = 0;
    uint32_t graph_splits = 0;  // backend switches this token (5080 cb-off: 340)
    uint32_t split_callback_hooks = 0;
    uint32_t split_buffer_type = 0;
    uint32_t split_backend = 0;
    uint32_t split_op = 0;
    uint64_t ffn_cuda_gemm = 0;
    uint64_t ffn_cpu_gemm = 0;
    double prefill_s = 0.0;
    uint32_t prefill_tok = 0;
    double decode_s = 0.0;  // generate wall only; tok/s uses this, not prefill
    uint64_t cuda0_model = 0;
    uint64_t cuda0_compute = 0;
    uint64_t nvidia_used = 0;
    bool host_pages_pinned = false;
    bool cuda_events = false;
    bool trace_off = true;
    bool real_h2d = false;  // ggml tensor_set into the buffer MUL_MAT reads
};

class PerfClocks {
public:
    void reset();
    void begin_session();
    void end_token();

    void begin_span(PerfSpan span);
    void end_span(PerfSpan span);

    void add_ms(PerfSpan span, double ms);
    void add_h2d(uint64_t bytes);
    void add_d2h(uint64_t bytes);
    void add_cuda_ffn_bind();
    void add_host_ffn_bind();
    void add_overlap_prefetch();
    void note_backend(bool on_host);
    void begin_decode();  // reset per-decode split counter
    void begin_decode_wall();  // start generate tok/s clock (after prefill)
    uint32_t graph_splits() const { return graph_splits_; }

    void set_plan(uint32_t n_park, uint32_t n_stream, bool host_pinned);
    void set_vram(uint64_t weights, uint64_t kv, uint64_t scratch, uint64_t free_b);
    void set_cuda_events(bool yes);
    void set_trace_off(bool off);
    void set_prefill(double seconds, uint32_t tokens);
    void set_ffn_gemm(uint64_t cuda, uint64_t cpu);
    void set_split_ledger(uint32_t hooks, uint32_t buffer_type, uint32_t backend, uint32_t op);
    void set_cuda0(uint64_t model_b, uint64_t compute_b);
    void set_nvidia_used(uint64_t used_b);
    void set_real_h2d(bool yes);
    void add_cuda_ffn_binds(uint64_t n);

    // Query CUDA free/total when built with nvcc. Zeros otherwise.
    static bool query_vram(uint64_t* free_b, uint64_t* total_b);

    PerfSnapshot snapshot() const;
    std::string format_line() const;
    std::string format_bottlenecks() const;
    uint32_t tokens() const { return n_tokens_; }

private:
    double span_ms_[5] = {};
    uint64_t span_t0_ns_[5] = {};
    bool span_open_[5] = {};
    uint64_t h2d_bytes_ = 0;
    uint64_t d2h_bytes_ = 0;
    uint64_t vram_weights_ = 0;
    uint64_t vram_kv_ = 0;
    uint64_t vram_scratch_ = 0;
    uint64_t vram_free_ = 0;
    uint64_t cuda_ffn_binds_ = 0;
    uint64_t host_ffn_binds_ = 0;
    uint64_t overlap_prefetches_ = 0;
    uint32_t graph_splits_ = 0;
    uint32_t last_backend_ = 2;  // 2 = none, 0 = gpu, 1 = host
    uint64_t t0_ns_ = 0;
    uint64_t decode_t0_ns_ = 0;
    uint32_t n_tokens_ = 0;
    uint32_t n_parked_ = 0;
    uint32_t n_streamed_ = 0;
    uint32_t split_callback_hooks_ = 0;
    uint32_t split_buffer_type_ = 0;
    uint32_t split_backend_ = 0;
    uint32_t split_op_ = 0;
    uint64_t ffn_cuda_gemm_ = 0;
    uint64_t ffn_cpu_gemm_ = 0;
    double prefill_s_ = 0.0;
    uint32_t prefill_tok_ = 0;
    uint64_t cuda0_model_ = 0;
    uint64_t cuda0_compute_ = 0;
    uint64_t nvidia_used_ = 0;
    bool host_pinned_ = false;
    bool cuda_events_ = false;
    bool trace_off_ = true;
    bool real_h2d_ = false;
};

std::string format_performance_line(const PerfSnapshot& s);
std::string format_performance_bottlenecks(const PerfSnapshot& s);
std::string format_tokens_per_sec_line(double tps, uint32_t n, double elapsed_s);
std::string format_pcie_bound_line(uint32_t n_stream);

// Ranked bottlenecks from measured ms + bind counts. Not a guess of tok/s.
const char* top_bottleneck(const PerfSnapshot& s);

}  // namespace micro_llm
