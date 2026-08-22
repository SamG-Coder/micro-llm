#pragma once

// End-of-run decode telemetry. Numbers come from instrumentation, not guesses.
// The closed-form model is printed alongside so a VM without the 17GB GGUF
// can still show why 3.5 tok/s happened and what the 5080 hour should hit.

#include "micro_llm/residency.hpp"

#include <cstdint>
#include <string>

namespace micro_llm {

struct PerfSnapshot {
    uint32_t generated = 0;
    uint32_t prefill_tokens = 0;
    double decode_s = 0;
    double prefill_s = 0;
    double tok_s = 0;
    double mean_token_ms = 0;

    uint64_t h2d_bytes = 0;
    uint64_t d2h_bytes = 0;
    uint64_t sync_count = 0;
    uint64_t ffn_bind_count = 0;
    uint64_t ffn_prefetch_count = 0;
    uint64_t ffn_evict_count = 0;
    // host_ffn_binds: run total of FFN gate/up activations that arrived on host
    // (ggml CPU), not CUDA / CUDA_Host. Swap gate wants 0.
    uint64_t host_ffn_binds = 0;
    // missing_hooks: RUN count of FFN layers whose hook never ran (unwired).
    // n_fired stayed 0 because the tap did not fire — not a prune.
    // A streamed FFN with n_fired=0 is a missing hook; keep all 17408.
    // missing_hooks_token: same count for the last token only.
    uint32_t missing_hooks = 0;
    uint32_t missing_hooks_token = 0;
    uint64_t trace_encode_ns = 0;
    uint64_t ui_push_ns = 0;
    uint64_t sample_ns = 0;
    uint64_t decode_ns = 0;
    uint64_t hook_ns = 0;

    uint32_t n_parked_ffn = 0;
    uint32_t n_streamed_ffn = 0;
    uint64_t vram_weight_bytes = 0;
    uint64_t cuda_host_bytes = 0;
    uint64_t cpu_bytes = 0;
    uint64_t kv_bytes = 0;
    uint64_t scratch_bytes = 0;
    uint64_t card_stack_bytes = 0;
    uint64_t pcie_bytes_per_token = 0;

    bool flash_attn = false;
    bool quant_kv = false;
    bool deltanet_cuda = true;
    bool host_pages_pinned = false;
    int32_t n_gpu_layers = 0;
    uint32_t n_ctx = 0;

    bool measured = false;  // true after a real llama_decode loop
};

class PerfTelemetry {
public:
    void reset();
    void apply_plan(const ResidencyPlan& plan);

    void add_h2d(uint64_t bytes) { snap_.h2d_bytes += bytes; }
    void add_d2h(uint64_t bytes) { snap_.d2h_bytes += bytes; }
    void add_sync(uint64_t n = 1) { snap_.sync_count += n; }
    void add_bind() { snap_.ffn_bind_count += 1; }
    void add_prefetch() { snap_.ffn_prefetch_count += 1; }
    void add_evict() { snap_.ffn_evict_count += 1; }
    void add_ns(uint64_t* field, uint64_t ns) { if (field) *field += ns; }

    void set_generated(uint32_t n, double decode_s);
    void set_prefill(uint32_t n, double prefill_s);
    void set_host_pages_pinned(bool v) { snap_.host_pages_pinned = v; }
    void set_hook_counters(uint64_t host_ffn_binds, uint32_t missing_hooks_run,
                           uint32_t missing_hooks_token) {
        snap_.host_ffn_binds = host_ffn_binds;
        snap_.missing_hooks = missing_hooks_run;
        snap_.missing_hooks_token = missing_hooks_token;
    }
    void mark_measured() { snap_.measured = true; }

    const PerfSnapshot& snap() const { return snap_; }
    PerfSnapshot& snap() { return snap_; }

    std::string format_report() const;

    static PerfTelemetry& thread_local_instance();

private:
    PerfSnapshot snap_;
    ResidencyPlan plan_{};
};

// Steady-clock helper. Cheap enough for the token path if used sparingly.
uint64_t now_ns();

std::string format_tokens_per_sec_line(double tps, uint32_t n, double elapsed_s);
std::string format_ffn_cuda_park_line(uint32_t n_park, uint64_t bytes);
std::string format_ffn_cuda_bind_line(uint32_t layer, bool parked);

// PERFORMANCE one-liner. missing_hooks is the RUN count (see PerfSnapshot).
std::string format_performance_line(const PerfSnapshot& s);
// 5080 swap: tok/s>3.5 AND host_ffn_binds=0 AND missing_hooks=0.
std::string format_swap_gate_line(const PerfSnapshot& s);
bool swap_gate_ok(double tok_s, uint64_t host_ffn_binds, uint32_t missing_hooks);

}  // namespace micro_llm
