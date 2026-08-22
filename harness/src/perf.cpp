#include "micro_llm/perf.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>

#if defined(MICRO_LLM_HAS_CUDA)
#include <cuda_runtime.h>
#endif

namespace micro_llm {
namespace {

uint64_t now_ns() {
    using clock = std::chrono::steady_clock;
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch())
            .count());
}

double ns_to_ms(uint64_t ns) { return static_cast<double>(ns) / 1.0e6; }

}  // namespace

void PerfClocks::reset() {
    for (int i = 0; i < 5; ++i) {
        span_ms_[i] = 0.0;
        span_t0_ns_[i] = 0;
        span_open_[i] = false;
    }
    h2d_bytes_ = 0;
    d2h_bytes_ = 0;
    vram_weights_ = 0;
    vram_kv_ = 0;
    vram_scratch_ = 0;
    vram_free_ = 0;
    cuda_ffn_binds_ = 0;
    host_ffn_binds_ = 0;
    overlap_prefetches_ = 0;
    t0_ns_ = 0;
    n_tokens_ = 0;
    n_parked_ = 0;
    n_streamed_ = 0;
    host_pinned_ = false;
    cuda_events_ = false;
}

void PerfClocks::begin_session() {
    reset();
    t0_ns_ = now_ns();
}

void PerfClocks::end_token() { ++n_tokens_; }

void PerfClocks::begin_span(PerfSpan span) {
    const int i = static_cast<int>(span);
    span_t0_ns_[i] = now_ns();
    span_open_[i] = true;
}

void PerfClocks::end_span(PerfSpan span) {
    const int i = static_cast<int>(span);
    if (!span_open_[i]) {
        return;
    }
    span_ms_[i] += ns_to_ms(now_ns() - span_t0_ns_[i]);
    span_open_[i] = false;
}

void PerfClocks::add_ms(PerfSpan span, double ms) {
    if (ms > 0.0) {
        span_ms_[static_cast<int>(span)] += ms;
    }
}

void PerfClocks::add_h2d(uint64_t bytes) { h2d_bytes_ += bytes; }
void PerfClocks::add_d2h(uint64_t bytes) { d2h_bytes_ += bytes; }
void PerfClocks::add_cuda_ffn_bind() { ++cuda_ffn_binds_; }
void PerfClocks::add_host_ffn_bind() { ++host_ffn_binds_; }
void PerfClocks::add_overlap_prefetch() { ++overlap_prefetches_; }

void PerfClocks::set_plan(uint32_t n_park, uint32_t n_stream, bool host_pinned) {
    n_parked_ = n_park;
    n_streamed_ = n_stream;
    host_pinned_ = host_pinned;
}

void PerfClocks::set_vram(uint64_t weights, uint64_t kv, uint64_t scratch, uint64_t free_b) {
    vram_weights_ = weights;
    vram_kv_ = kv;
    vram_scratch_ = scratch;
    vram_free_ = free_b;
}

void PerfClocks::set_cuda_events(bool yes) { cuda_events_ = yes; }

bool PerfClocks::query_vram(uint64_t* free_b, uint64_t* total_b) {
#if defined(MICRO_LLM_HAS_CUDA)
    size_t free_sz = 0;
    size_t total_sz = 0;
    if (cudaMemGetInfo(&free_sz, &total_sz) != cudaSuccess) {
        return false;
    }
    if (free_b) {
        *free_b = static_cast<uint64_t>(free_sz);
    }
    if (total_b) {
        *total_b = static_cast<uint64_t>(total_sz);
    }
    return true;
#else
    if (free_b) {
        *free_b = 0;
    }
    if (total_b) {
        *total_b = 0;
    }
    return false;
#endif
}

PerfSnapshot PerfClocks::snapshot() const {
    PerfSnapshot s;
    s.n_tokens = n_tokens_;
    const uint64_t now = now_ns();
    s.wall_s = t0_ns_ == 0 ? 0.0 : static_cast<double>(now - t0_ns_) / 1.0e9;
    s.tok_per_sec = s.wall_s > 0.0 ? static_cast<double>(n_tokens_) / s.wall_s : 0.0;
    s.latency_ms = n_tokens_ > 0 ? (s.wall_s * 1000.0) / static_cast<double>(n_tokens_) : 0.0;
    s.gpu_ms = n_tokens_ > 0 ? span_ms_[0] / static_cast<double>(n_tokens_) : 0.0;
    s.pcie_ms = n_tokens_ > 0 ? span_ms_[1] / static_cast<double>(n_tokens_) : 0.0;
    s.cpu_ms = n_tokens_ > 0 ? span_ms_[2] / static_cast<double>(n_tokens_) : 0.0;
    s.sync_ms = n_tokens_ > 0 ? span_ms_[3] / static_cast<double>(n_tokens_) : 0.0;
    s.trace_ms = n_tokens_ > 0 ? span_ms_[4] / static_cast<double>(n_tokens_) : 0.0;
    s.h2d_bytes = h2d_bytes_;
    s.d2h_bytes = d2h_bytes_;
    s.h2d_bytes_per_tok = n_tokens_ > 0 ? h2d_bytes_ / n_tokens_ : 0;
    s.d2h_bytes_per_tok = n_tokens_ > 0 ? d2h_bytes_ / n_tokens_ : 0;
    s.vram_weights = vram_weights_;
    s.vram_kv = vram_kv_;
    s.vram_scratch = vram_scratch_;
    s.vram_free = vram_free_;
    s.n_parked_ffn = n_parked_;
    s.n_streamed_ffn = n_streamed_;
    s.cuda_ffn_binds = cuda_ffn_binds_;
    s.host_ffn_binds = host_ffn_binds_;
    s.overlap_prefetches = overlap_prefetches_;
    s.host_pages_pinned = host_pinned_;
    s.cuda_events = cuda_events_;
    return s;
}

std::string format_performance_line(const PerfSnapshot& s) {
    char buf[768];
    std::snprintf(
        buf, sizeof(buf),
        "PERFORMANCE tok/s=%.2f latency_ms=%.2f gpu_ms=%.2f pcie_ms=%.2f "
        "cpu_ms=%.2f sync_ms=%.2f trace_ms=%.2f h2d_B=%llu d2h_B=%llu "
        "vram_w_MiB=%.1f vram_kv_MiB=%.1f vram_scratch_MiB=%.1f vram_free_MiB=%.1f "
        "park=%u stream=%u cuda_ffn=%llu host_ffn=%llu overlap=%llu pinned=%d "
        "cuda_events=%d tokens=%u elapsed_s=%.2f",
        s.tok_per_sec, s.latency_ms, s.gpu_ms, s.pcie_ms, s.cpu_ms, s.sync_ms, s.trace_ms,
        static_cast<unsigned long long>(s.h2d_bytes_per_tok),
        static_cast<unsigned long long>(s.d2h_bytes_per_tok),
        static_cast<double>(s.vram_weights) / (1024.0 * 1024.0),
        static_cast<double>(s.vram_kv) / (1024.0 * 1024.0),
        static_cast<double>(s.vram_scratch) / (1024.0 * 1024.0),
        static_cast<double>(s.vram_free) / (1024.0 * 1024.0), s.n_parked_ffn, s.n_streamed_ffn,
        static_cast<unsigned long long>(s.cuda_ffn_binds),
        static_cast<unsigned long long>(s.host_ffn_binds),
        static_cast<unsigned long long>(s.overlap_prefetches), s.host_pages_pinned ? 1 : 0,
        s.cuda_events ? 1 : 0, s.n_tokens, s.wall_s);
    return buf;
}

const char* top_bottleneck(const PerfSnapshot& s) {
    // Host FFN compute is the 7780 3.5 tok/s trap. Call it first when present.
    if (s.host_ffn_binds > 0 && s.n_streamed_ffn > 0 &&
        s.host_ffn_binds >= s.cuda_ffn_binds && s.cpu_ms >= s.gpu_ms &&
        s.cpu_ms >= s.pcie_ms) {
        return "host_ffn";
    }
    struct Row {
        const char* name;
        double ms;
    } rows[] = {
        {"gpu_compute", s.gpu_ms}, {"pcie_h2d", s.pcie_ms}, {"cpu", s.cpu_ms},
        {"sync", s.sync_ms},       {"trace", s.trace_ms},
    };
    const char* best = rows[0].name;
    double best_ms = rows[0].ms;
    for (const auto& r : rows) {
        if (r.ms > best_ms) {
            best_ms = r.ms;
            best = r.name;
        }
    }
    if (best_ms <= 0.0 && s.host_ffn_binds > 0) {
        return "host_ffn";
    }
    return best;
}

std::string format_performance_bottlenecks(const PerfSnapshot& s) {
    char buf[192];
    std::snprintf(buf, sizeof(buf),
                  "PERFORMANCE bottlenecks=%s host_ffn_binds=%llu cuda_ffn_binds=%llu "
                  "overlap=%llu pinned=%d",
                  top_bottleneck(s), static_cast<unsigned long long>(s.host_ffn_binds),
                  static_cast<unsigned long long>(s.cuda_ffn_binds),
                  static_cast<unsigned long long>(s.overlap_prefetches),
                  s.host_pages_pinned ? 1 : 0);
    return buf;
}

std::string PerfClocks::format_line() const { return format_performance_line(snapshot()); }

std::string PerfClocks::format_bottlenecks() const {
    return format_performance_bottlenecks(snapshot());
}

std::string format_tokens_per_sec_line(double tps, uint32_t n, double elapsed_s) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "tokens/s=%.2f generated=%u elapsed_s=%.2f", tps, n,
                  elapsed_s);
    return buf;
}

}  // namespace micro_llm
