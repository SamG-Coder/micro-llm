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
    bool host_pages_pinned = false;
    bool cuda_events = false;
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

    void set_plan(uint32_t n_park, uint32_t n_stream, bool host_pinned);
    void set_vram(uint64_t weights, uint64_t kv, uint64_t scratch, uint64_t free_b);
    void set_cuda_events(bool yes);

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
    uint64_t t0_ns_ = 0;
    uint32_t n_tokens_ = 0;
    uint32_t n_parked_ = 0;
    uint32_t n_streamed_ = 0;
    bool host_pinned_ = false;
    bool cuda_events_ = false;
};

std::string format_performance_line(const PerfSnapshot& s);
std::string format_performance_bottlenecks(const PerfSnapshot& s);
std::string format_tokens_per_sec_line(double tps, uint32_t n, double elapsed_s);

// Ranked bottlenecks from measured ms + bind counts. Not a guess of tok/s.
const char* top_bottleneck(const PerfSnapshot& s);

}  // namespace micro_llm
