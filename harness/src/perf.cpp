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
    graph_splits_ = 0;
    last_backend_ = 2;
    t0_ns_ = 0;
    decode_t0_ns_ = 0;
    n_tokens_ = 0;
    n_parked_ = 0;
    n_streamed_ = 0;
    split_callback_hooks_ = 0;
    split_buffer_type_ = 0;
    split_backend_ = 0;
    split_op_ = 0;
    ffn_cuda_gemm_ = 0;
    ffn_cpu_gemm_ = 0;
    prefill_s_ = 0.0;
    prefill_tok_ = 0;
    cuda0_model_ = 0;
    cuda0_compute_ = 0;
    nvidia_used_ = 0;
    host_pinned_ = false;
    cuda_events_ = false;
    trace_off_ = true;
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

void PerfClocks::begin_decode() {
    graph_splits_ = 0;
    last_backend_ = 2;
}

void PerfClocks::begin_decode_wall() { decode_t0_ns_ = now_ns(); }

void PerfClocks::note_backend(bool on_host) {
    const uint32_t b = on_host ? 1u : 0u;
    if (last_backend_ != 2 && last_backend_ != b) {
        ++graph_splits_;
    }
    last_backend_ = b;
}

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

void PerfClocks::set_trace_off(bool off) { trace_off_ = off; }

void PerfClocks::set_prefill(double seconds, uint32_t tokens) {
    prefill_s_ = seconds;
    prefill_tok_ = tokens;
}

void PerfClocks::set_ffn_gemm(uint64_t cuda, uint64_t cpu) {
    ffn_cuda_gemm_ = cuda;
    ffn_cpu_gemm_ = cpu;
}

void PerfClocks::set_split_ledger(uint32_t hooks, uint32_t buffer_type, uint32_t backend,
                                 uint32_t op) {
    split_callback_hooks_ = trace_off_ ? 0u : hooks;
    split_buffer_type_ = buffer_type;
    split_backend_ = backend;
    split_op_ = op;
}

void PerfClocks::set_cuda0(uint64_t model_b, uint64_t compute_b) {
    cuda0_model_ = model_b;
    cuda0_compute_ = compute_b;
}

void PerfClocks::set_nvidia_used(uint64_t used_b) { nvidia_used_ = used_b; }

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
    s.decode_s = decode_t0_ns_ == 0 ? s.wall_s : static_cast<double>(now - decode_t0_ns_) / 1.0e9;
    s.tok_per_sec = s.decode_s > 0.0 ? static_cast<double>(n_tokens_) / s.decode_s : 0.0;
    s.latency_ms = n_tokens_ > 0 ? (s.decode_s * 1000.0) / static_cast<double>(n_tokens_) : 0.0;
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
    s.graph_splits = graph_splits_;
    s.split_callback_hooks = trace_off_ ? 0u : split_callback_hooks_;
    s.split_buffer_type = split_buffer_type_;
    s.split_backend = split_backend_;
    s.split_op = split_op_;
    s.ffn_cuda_gemm = ffn_cuda_gemm_;
    s.ffn_cpu_gemm = ffn_cpu_gemm_;
    s.prefill_s = prefill_s_;
    s.prefill_tok = prefill_tok_;
    s.cuda0_model = cuda0_model_;
    s.cuda0_compute = cuda0_compute_;
    s.nvidia_used = nvidia_used_;
    s.host_pages_pinned = host_pinned_;
    s.cuda_events = cuda_events_;
    s.trace_off = trace_off_;
    return s;
}

std::string format_performance_line(const PerfSnapshot& s) {
    char buf[768];
    std::snprintf(
        buf, sizeof(buf),
        "PERFORMANCE tok/s=%.2f latency_ms=%.2f gpu_ms=%.2f pcie_ms=%.2f "
        "cpu_ms=%.2f sync_ms=%.2f trace_ms=%.2f h2d_B=%llu d2h_B=%llu "
        "vram_w_MiB=%.1f vram_kv_MiB=%.1f vram_scratch_MiB=%.1f vram_free_MiB=%.1f "
        "park=%u stream=%u cuda_ffn=%llu host_ffn=%llu overlap=%llu splits=%u "
        "pinned=%d cuda_events=%d tokens=%u elapsed_s=%.2f",
        s.tok_per_sec, s.latency_ms, s.gpu_ms, s.pcie_ms, s.cpu_ms, s.sync_ms, s.trace_ms,
        static_cast<unsigned long long>(s.h2d_bytes_per_tok),
        static_cast<unsigned long long>(s.d2h_bytes_per_tok),
        static_cast<double>(s.vram_weights) / (1024.0 * 1024.0),
        static_cast<double>(s.vram_kv) / (1024.0 * 1024.0),
        static_cast<double>(s.vram_scratch) / (1024.0 * 1024.0),
        static_cast<double>(s.vram_free) / (1024.0 * 1024.0), s.n_parked_ffn, s.n_streamed_ffn,
        static_cast<unsigned long long>(s.cuda_ffn_binds),
        static_cast<unsigned long long>(s.host_ffn_binds),
        static_cast<unsigned long long>(s.overlap_prefetches), s.graph_splits,
        s.host_pages_pinned ? 1 : 0, s.cuda_events ? 1 : 0, s.n_tokens, s.wall_s);
    return buf;
}

const char* top_bottleneck(const PerfSnapshot& s) {
    // Host FFN compute is the 7780 3.5 tok/s trap. Call it first when present.
    if (s.graph_splits > 16) {
        return "graph_splits";
    }
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
                  "overlap=%llu splits=%u pinned=%d",
                  top_bottleneck(s), static_cast<unsigned long long>(s.host_ffn_binds),
                  static_cast<unsigned long long>(s.cuda_ffn_binds),
                  static_cast<unsigned long long>(s.overlap_prefetches), s.graph_splits,
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

std::string format_pcie_bound_line(uint32_t n_stream) {
    const uint64_t bpt = stream_pcie_bytes_per_tok(n_stream);
    const double tok = stream_pcie_tok_per_sec(n_stream);
    char buf[448];
    if (n_stream == 0) {
        std::snprintf(buf, sizeof(buf),
                      "PCIE B/tok=0 n_stream=0 gbs=%.1f. 20 tok/s is CUDA Q4 compute "
                      "(all FFN resident), not a PCIe bound. ggml_rebind_q4=0 so we "
                      "do not H2D-rebind mid-graph. 5080 stub tok/s=%.2f splits=%u "
                      "is not the speed path.",
                      static_cast<double>(kPcie5PracticalGBs),
                      static_cast<double>(kMeasured5080TokPerSecCbOff),
                      kMeasured5080GraphSplits);
    } else {
        const double ms = tok > 0.0 ? 1000.0 / tok : 0.0;
        std::snprintf(buf, sizeof(buf),
                      "PCIE B/tok=%llu n_stream=%u layer_MiB=%.1f gbs=%.1f "
                      "pcie_ms/tok=%.2f pcie_tok/s=%.1f (host ggml of those layers "
                      "is not the 20+ path; ggml_rebind_q4=0)",
                      static_cast<unsigned long long>(bpt), n_stream,
                      static_cast<double>(kQ4FfnLayerBytesMeasured5080) / (1024.0 * 1024.0),
                      static_cast<double>(kPcie5PracticalGBs), ms, tok);
    }
    return buf;
}

}  // namespace micro_llm
