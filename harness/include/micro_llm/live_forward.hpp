#pragma once

// Live Qwen 27B (3.6/3.8 hybrid) forward. Not a from-scratch engine.
//
// LlamaCppLiveForwardBackend attaches to llama.cpp graph eval (cb_eval)
// when MICRO_LLM_HAS_LLAMA is set. Without llama.cpp it still compiles,
// probes GGUF metadata, and refuses to fake an hour.
//
// StubLiveForwardBackend is tests-only (no 17GB GGUF in CI).

#include "micro_llm/streamer.hpp"
#include "micro_llm/trace_hooks.hpp"

#include <memory>
#include <string>

namespace micro_llm {

struct LiveForwardConfig {
    std::string model_path;
    std::string prompt;
    std::string out_path = "prune_table.bin";
    uint32_t n_predict = 64;
    uint32_t n_ctx = 8192;
    uint32_t top_k = 40;
    bool load_vision = false;  // job is text coding assistant
    bool load_mtp = false;
    int32_t n_gpu_layers = -1;
};

struct LiveForwardStatus {
    bool ok = false;
    bool engine_linked = false;
    bool architecture_ok = false;
    bool ran_tokens = false;
    std::string backend;
    std::string message;
    uint64_t n_tokens = 0;
};

class LiveForwardBackend {
public:
    virtual ~LiveForwardBackend() = default;
    virtual const char* name() const = 0;
    virtual bool is_stub() const { return false; }
    virtual bool engine_linked() const { return false; }
    virtual LiveForwardStatus probe(const LiveForwardConfig& cfg) = 0;
    virtual LiveForwardStatus run(TraceHooks& hooks, TraceStreamer& streamer,
                                  const LiveForwardConfig& cfg) = 0;
};

class StubLiveForwardBackend : public LiveForwardBackend {
public:
    const char* name() const override { return "stub"; }
    bool is_stub() const override { return true; }
    LiveForwardStatus probe(const LiveForwardConfig& cfg) override;
    LiveForwardStatus run(TraceHooks& hooks, TraceStreamer& streamer,
                          const LiveForwardConfig& cfg) override;
};

class LlamaCppLiveForwardBackend : public LiveForwardBackend {
public:
    const char* name() const override { return "llama.cpp"; }
    bool engine_linked() const override;
    LiveForwardStatus probe(const LiveForwardConfig& cfg) override;
    LiveForwardStatus run(TraceHooks& hooks, TraceStreamer& streamer,
                          const LiveForwardConfig& cfg) override;
};

// kind: "stub" | "llama" | "auto" (llama if model_path set, else stub)
std::unique_ptr<LiveForwardBackend> make_live_forward(const std::string& kind);

const char* default_coding_assistant_prompt();

bool llama_cpp_linked();

}  // namespace micro_llm
