#pragma once

// micro-llm-trace argument resolve. --ui is not exclusive of --model.

#include "micro_llm/live_forward.hpp"

#include <string>
#include <vector>

namespace micro_llm {

inline constexpr uint32_t kCliTestNPredict = 64;
inline constexpr uint32_t kHourNPredict = 20000;
inline constexpr uint32_t kHourCheckpointEvery = 2000;
inline constexpr int32_t kHybridNGpuLayers = 99;

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

// Hybrid pin: n_gpu_layers=99, then pull FFN + DeltaNet weights to CPU.
// Do NOT use ngl=16 (first 16 layers, the wrong 16).
int32_t hybrid_n_gpu_layers();
std::vector<std::string> hybrid_cpu_tensor_regexes();
uint64_t pinned_ga_weight_bytes();

}  // namespace micro_llm
