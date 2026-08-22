#include "micro_llm/perf_telemetry.hpp"

#include <chrono>
#include <cstdio>
#include <sstream>

namespace micro_llm {

uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

void PerfTelemetry::reset() { snap_ = PerfSnapshot{}; }

void PerfTelemetry::apply_plan(const ResidencyPlan& plan) {
    plan_ = plan;
    snap_.n_parked_ffn = plan.n_parked_ffn;
    snap_.n_streamed_ffn = plan.n_streamed_ffn;
    snap_.vram_weight_bytes = plan.vram_weight_bytes;
    snap_.cuda_host_bytes = plan.cuda_host_bytes;
    snap_.cpu_bytes = plan.cpu_bytes;
    snap_.kv_bytes = plan.kv_bytes;
    snap_.scratch_bytes = plan.scratch_bytes;
    snap_.card_stack_bytes = plan.card_stack_bytes;
    snap_.pcie_bytes_per_token = plan.pcie_bytes_per_token;
    snap_.flash_attn = plan.enable_flash_attn;
    snap_.quant_kv = plan.use_quant_kv;
    snap_.deltanet_cuda = plan.pin_deltanet;
    snap_.n_gpu_layers = plan.n_gpu_layers;
    snap_.n_ctx = plan.n_ctx;
}

void PerfTelemetry::set_generated(uint32_t n, double decode_s) {
    snap_.generated = n;
    snap_.decode_s = decode_s;
    snap_.tok_s = decode_s > 0.0 ? static_cast<double>(n) / decode_s : 0.0;
    snap_.mean_token_ms = n > 0 && decode_s > 0.0 ? (decode_s * 1000.0) / static_cast<double>(n) : 0.0;
}

void PerfTelemetry::set_prefill(uint32_t n, double prefill_s) {
    snap_.prefill_tokens = n;
    snap_.prefill_s = prefill_s;
}

PerfTelemetry& PerfTelemetry::thread_local_instance() {
    static thread_local PerfTelemetry t;
    return t;
}

namespace {

const char* bottleneck_line(const PerfSnapshot& s, const DecodeBudget& b) {
    if (!s.deltanet_cuda) {
        return "1) host DeltaNet (48*3.2ms floor)  2) host FFN ggml  3) per-layer D2H sync";
    }
    if (s.n_streamed_ffn > 16) {
        return "1) PCIe H2D of streamed FFN  2) GPU FFN GEMM  3) leftover attention";
    }
    if (s.tok_s > 0.0 && s.tok_s < 20.0 && s.measured) {
        return "1) streamed FFN PCIe  2) GPU compute  3) sample/trace (see ns fields)";
    }
    (void)b;
    return "1) GPU Q4_K FFN  2) streamed FFN PCIe  3) GA/DeltaNet";
}

}  // namespace

std::string PerfTelemetry::format_report() const {
    const DecodeBudget b = decode_budget_model(plan_);
    const double h2d_per_tok =
        snap_.generated > 0 ? static_cast<double>(snap_.h2d_bytes) / snap_.generated : 0.0;
    const double d2h_per_tok =
        snap_.generated > 0 ? static_cast<double>(snap_.d2h_bytes) / snap_.generated : 0.0;
    const double sync_per_tok =
        snap_.generated > 0 ? static_cast<double>(snap_.sync_count) / snap_.generated : 0.0;
    const double trace_us =
        snap_.generated > 0 ? static_cast<double>(snap_.trace_encode_ns) / snap_.generated / 1000.0
                            : 0.0;
    const double ui_us =
        snap_.generated > 0 ? static_cast<double>(snap_.ui_push_ns) / snap_.generated / 1000.0 : 0.0;
    const double sample_us =
        snap_.generated > 0 ? static_cast<double>(snap_.sample_ns) / snap_.generated / 1000.0 : 0.0;
    const double hook_us =
        snap_.generated > 0 ? static_cast<double>(snap_.hook_ns) / snap_.generated / 1000.0 : 0.0;

    char buf[2048];
    std::snprintf(
        buf, sizeof(buf),
        "==== micro-llm decode telemetry ====\n"
        "measured=%d generated=%u decode_s=%.3f tok/s=%.2f token_ms=%.2f\n"
        "prefill_tokens=%u prefill_s=%.3f ctx=%u ngl=%d flash=%d quant_kv=%d pinned_host=%d\n"
        "parked_ffn=%u streamed_ffn=%u deltanet=CUDA:%d\n"
        "vram_weights=%.2fMiB cuda_host=%.2fMiB cpu=%.2fMiB kv=%.2fMiB scratch=%.2fMiB "
        "stack=%.2fMiB / 15564.8MiB\n"
        "pcie_model_B/token=%llu  h2d_B/token=%.0f  d2h_B/token=%.0f  sync/token=%.2f\n"
        "trace_us/token=%.1f  ui_push_us/token=%.1f  sample_us/token=%.1f  hook_us/token=%.1f\n"
        "binds=%llu prefetch=%llu evict=%llu\n"
        "-- model (host hour 3.5 tok/s, this VM did not load the 17GB GGUF) --\n"
        "host_all_cpu_ms=%.1f (%.2f tok/s)\n"
        "park41_ffn_vram_delta_host_ms=%.1f (%.2f tok/s)  << observed: tok/s stayed ~3.5\n"
        "all_ffn_cuda_delta_host_ms=%.1f (%.2f tok/s)  << 20 tok/s IMPOSSIBLE if DeltaNet host\n"
        "planned_pcie5_ms=%.1f (%.2f tok/s)  planned_pcie4_ms=%.1f (%.2f tok/s)\n"
        "20tok/s possible: pcie5=%d pcie4=%d  (DeltaNet must be CUDA; ngl is not the pin)\n"
        "top bottlenecks: %s\n"
        "====================================\n",
        snap_.measured ? 1 : 0, snap_.generated, snap_.decode_s, snap_.tok_s, snap_.mean_token_ms,
        snap_.prefill_tokens, snap_.prefill_s, snap_.n_ctx, snap_.n_gpu_layers,
        snap_.flash_attn ? 1 : 0, snap_.quant_kv ? 1 : 0, snap_.host_pages_pinned ? 1 : 0,
        snap_.n_parked_ffn, snap_.n_streamed_ffn, snap_.deltanet_cuda ? 1 : 0,
        snap_.vram_weight_bytes / (1024.0 * 1024.0), snap_.cuda_host_bytes / (1024.0 * 1024.0),
        snap_.cpu_bytes / (1024.0 * 1024.0), snap_.kv_bytes / (1024.0 * 1024.0),
        snap_.scratch_bytes / (1024.0 * 1024.0), snap_.card_stack_bytes / (1024.0 * 1024.0),
        static_cast<unsigned long long>(snap_.pcie_bytes_per_token), h2d_per_tok, d2h_per_tok,
        sync_per_tok, trace_us, ui_us, sample_us, hook_us,
        static_cast<unsigned long long>(snap_.ffn_bind_count),
        static_cast<unsigned long long>(snap_.ffn_prefetch_count),
        static_cast<unsigned long long>(snap_.ffn_evict_count), b.host_ms, 1000.0 / b.host_ms,
        b.park_only_ms, 1000.0 / b.park_only_ms, b.ffn_cuda_delta_host_ms,
        1000.0 / b.ffn_cuda_delta_host_ms, b.planned_ms_pcie5, 1000.0 / b.planned_ms_pcie5,
        b.planned_ms_pcie4, 1000.0 / b.planned_ms_pcie4, plan_.twenty_tok_s_possible_pcie5 ? 1 : 0,
        plan_.twenty_tok_s_possible_pcie4 ? 1 : 0, bottleneck_line(snap_, b));
    return buf;
}

std::string format_tokens_per_sec_line(double tps, uint32_t n, double elapsed_s) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "tokens/s=%.2f generated=%u elapsed_s=%.2f", tps, n, elapsed_s);
    return buf;
}

std::string format_ffn_cuda_park_line(uint32_t n_park, uint64_t bytes) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "ffn_cuda_park n=%u bytes=%llu stream_slot=%u never_64=1 ngl=0", n_park,
                  static_cast<unsigned long long>(bytes), kQ4FfnLayerBytes);
    return buf;
}

std::string format_ffn_cuda_bind_line(uint32_t layer, bool parked) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "ffn_cuda_bind layer=%u parked=%d stream=%d", layer,
                  parked ? 1 : 0, parked ? 0 : 1);
    return buf;
}

}  // namespace micro_llm
