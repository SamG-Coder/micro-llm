#pragma once

// micro-llm-trace argument resolve. --ui is not exclusive of --model.

#include "micro_llm/live_forward.hpp"

#include <string>
#include <vector>

namespace micro_llm {

inline constexpr uint32_t kCliTestNPredict = 64;
inline constexpr uint32_t kHourNPredict = 20000;
inline constexpr uint32_t kHourCheckpointEvery = 2000;
// ngl is NOT the pin. 0 = CPU default. Tensor overrides pin the 16 GA
// blocks. 16 parks the wrong 16 layers. 99 parks the file.
inline constexpr int32_t kHybridNGpuLayers = 0;

enum class TraceCliMode : uint8_t {
    SampleUi = 0,     // --ui / no args: sample window, no GGUF
    UiCheck = 1,      // --ui-check
    CheckServe = 2,   // --check-serve
    HourHeadless = 3, // --stub without --ui (tests)
    HourLive = 4,     // --model (default UI) or --ui --model / --stub --ui
};

struct TraceCliArgs {
    LiveForwardConfig cfg;
    bool use_stub = false;
    bool want_ui = false;
    bool ui_check = false;
    bool n_predict_set = false;
    bool help = false;
    bool parse_error = false;
    std::string check_serve;
};

// When --model is set without --n-predict, use 20000. CLI/struct default
// stays 64 so --stub tests stay short.
inline uint32_t resolve_n_predict(bool model_set, bool n_predict_set, uint32_t value) {
    if (model_set && !n_predict_set) {
        return kHourNPredict;
    }
    return value;
}

// --model always opens the live window. --stub stays headless unless --ui.
inline bool hour_should_open_ui(bool want_ui, bool has_model, bool stub) {
    if (has_model) {
        return true;
    }
    return want_ui && stub;
}

inline TraceCliMode resolve_trace_mode(const TraceCliArgs& a) {
    if (a.ui_check) {
        return TraceCliMode::UiCheck;
    }
    if (!a.check_serve.empty()) {
        return TraceCliMode::CheckServe;
    }
    const bool hour = a.use_stub || !a.cfg.model_path.empty();
    if (!hour) {
        return TraceCliMode::SampleUi;
    }
    if (hour_should_open_ui(a.want_ui, !a.cfg.model_path.empty(), a.use_stub)) {
        return TraceCliMode::HourLive;
    }
    return TraceCliMode::HourHeadless;
}

TraceCliArgs parse_trace_cli(int argc, char** argv);

// Hybrid pin is tensor overrides, not ngl. ngl=0 (CPU default).
// Do NOT use ngl=16 (wrong 16) or ngl=99 (parks the file).
int32_t hybrid_n_gpu_layers();
int32_t clamp_hybrid_n_gpu_layers(int32_t n);
std::vector<std::string> hybrid_cpu_tensor_regexes();  // FFN + DeltaNet + MTP
// GA QKVO + the first n_park FFN layers (weights on CUDA). Unparked FFN
// match the CPU catch-all listed after these (first regex wins).
std::vector<std::string> hybrid_gpu_tensor_regexes(
    uint32_t n_park = ffn_park_layers_that_fit());
uint64_t pinned_ga_weight_bytes();
uint64_t pinned_ga_kv_bytes();
uint64_t streamed_ffn_workspace_bytes();
uint32_t hybrid_ffn_park_layers();
std::string format_ffn_cuda_park_line(uint32_t n_park, uint64_t bytes);
std::string format_ffn_cuda_bind_line(uint32_t layer, bool parked);
std::string format_tokens_per_sec_line(double tps, uint32_t n, double elapsed_s);

}  // namespace micro_llm
