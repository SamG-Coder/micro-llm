#pragma once

// Live Qwen 27B (3.6/3.8 hybrid) forward. Not a from-scratch engine.
//
// LlamaCppLiveForwardBackend attaches to llama.cpp graph eval (cb_eval)
// when MICRO_LLM_HAS_LLAMA is set. Without llama.cpp it still compiles,
// probes GGUF metadata, and refuses to fake an hour.
//
// StubLiveForwardBackend is tests-only (no 17GB GGUF in CI).

#include "micro_llm/perf.hpp"
#include "micro_llm/streamer.hpp"
#include "micro_llm/trace_hooks.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace micro_llm {

struct LiveForwardConfig {
    std::string model_path;
    std::string prompt;
    std::string out_path = "prune_table.bin";
    uint32_t n_predict = 64;  // tests. Real hour without --n-predict uses 20000.
    uint32_t n_ctx = 8192;
    uint32_t top_k = 40;
    uint32_t n_batch = 512;
    uint32_t n_ubatch = 32;
    uint32_t checkpoint_every = 2000;
    bool load_vision = false;  // job is text coding assistant
    bool load_mtp = false;     // MTP extra GGUF block stays off
    bool continue_after_eos = true;
    bool disable_flash_attn = true;  // FA + CPU FFN split AVed; keep FA off
    // op_offload is how streamed (CPU-resident) FFN weights still run on
    // ggml CUDA after H2D. Not the park mechanism (that's tensor overrides).
    bool disable_op_offload = false;
    bool pack_checkpoint = false;  // scores only; never pack the ~18MB MLPT
    // Milestone 1: hooks off. --trace-off / --no-trace => cb_eval=nullptr.
    bool trace_hooks = false;
    // ngl is not the pin. 0 = CPU default. Not 16, not 99.
    int32_t n_gpu_layers = 0;
    uint32_t n_parked_ffn = 0;  // 0 = leftover after slots A/B + KV 20k
    std::function<void(const uint8_t* rec, size_t nbytes)> on_htr1;
    std::function<void(const PerfSnapshot&)> on_stats;
    std::atomic<bool>* abort = nullptr;
};

struct LiveForwardStatus {
    bool ok = false;
    bool engine_linked = false;
    bool architecture_ok = false;
    bool ran_tokens = false;
    std::string backend;
    std::string message;
    uint64_t n_tokens = 0;
    PerfSnapshot perf;
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
