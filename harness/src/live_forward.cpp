#include "micro_llm/live_forward.hpp"

#include "micro_llm/gguf_meta.hpp"
#include "micro_llm/graph_hooks.hpp"
#include "micro_llm/hook_ring.hpp"
#include "micro_llm/perf_telemetry.hpp"
#include "micro_llm/residency.hpp"
#include "micro_llm/trace_cli.hpp"

#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace micro_llm {

const char* default_coding_assistant_prompt() {
    return "You are a local coding assistant. Work in the repository. "
           "Answer with concrete patches, tests, and shell commands. "
           "Do not load vision. Job: implement, debug, and review C++ and Python.";
}

#if defined(MICRO_LLM_HAS_LLAMA)
bool llama_cpp_linked() { return true; }
#else
bool llama_cpp_linked() { return false; }
#endif

std::unique_ptr<LiveForwardBackend> make_live_forward(const std::string& kind) {
    if (kind == "stub") {
        return std::make_unique<StubLiveForwardBackend>();
    }
    if (kind == "llama" || kind == "llama.cpp" || kind == "auto") {
        return std::make_unique<LlamaCppLiveForwardBackend>();
    }
    return std::make_unique<LlamaCppLiveForwardBackend>();
}

LiveForwardStatus StubLiveForwardBackend::probe(const LiveForwardConfig&) {
    LiveForwardStatus s;
    s.ok = true;
    s.backend = name();
    s.message = "stub backend (tests only; no weights)";
    return s;
}

LiveForwardStatus StubLiveForwardBackend::run(TraceHooks& hooks, TraceStreamer& streamer,
                                              const LiveForwardConfig& cfg) {
    LiveForwardStatus s;
    s.backend = name();
    s.ok = true;
    s.ran_tokens = true;
    const uint32_t n = cfg.n_predict == 0 ? 2u : cfg.n_predict;
    hooks.mark_reserved_core(256);
    for (uint32_t id : {1u, 17u, 99u}) {
        hooks.on_vocab_id(id);
    }

    std::vector<float> gate(kFfnIntermediate, 0.f);
    std::vector<float> up(kFfnIntermediate, 0.f);
    std::vector<float> hin(kHiddenDim, 0.1f);
    std::vector<float> hout(kHiddenDim, 0.1f);

    const uint32_t n_park =
        cfg.n_parked_ffn != 0 ? cfg.n_parked_ffn : ffn_park_layers_that_fit();
    streamer.set_n_parked_ffn(n_park);
    auto& tel = PerfTelemetry::thread_local_instance();
    tel.reset();
    tel.apply_plan(plan_residency(default_qwen27b_q4km_catalog(), cfg.n_ctx,
                                  cfg.force_host_deltanet));
    std::fprintf(stderr, "%s\n",
                 format_ffn_cuda_park_line(n_park, static_cast<uint64_t>(n_park) * kQ4FfnLayerBytes)
                     .c_str());
    const auto t0 = std::chrono::steady_clock::now();
    streamer.begin_session();
    for (uint32_t t = 0; t < n; ++t) {
        hooks.begin_token(t);
        for (uint32_t layer = 0; layer < kNLayers; ++layer) {
            if (layer + 1 < kNLayers) {
                streamer.prefetch_ffn(layer + 1);
            }
            streamer.bind_ffn(layer);
            std::fill(gate.begin(), gate.end(), 0.f);
            std::fill(up.begin(), up.end(), 0.f);
            const uint32_t ch = (layer * 17u + t) % kFfnIntermediate;
            gate[ch] = 1.5f;
            up[ch] = 1.0f;
            hooks.on_ffn_activations(layer, gate.data(), up.data());
            streamer.evict_ffn(layer);
            if (is_delta_net_layer(layer)) {
                hout[0] = hin[0] + (layer % 7 == 0 ? 0.5f : 0.f);
                hooks.on_delta_hidden(pack_id_from_delta_layer(layer), hin.data(),
                                      hout.data(), 32);
            }
        }
        const uint32_t sampled = 2000 + t;
        const uint32_t topk[] = {3000 + t, 5u};
        streamer.enter_logits();
        hooks.on_vocab_id(sampled);
        hooks.on_topk_ids(topk, 2);
        if (cfg.on_htr1) {
            emit_htr1(hooks, sampled, topk, 2, t == 0, cfg.on_htr1);
        }
        hooks.after_logits(t, t == 0);
        streamer.leave_logits();
        if (cfg.checkpoint_every != 0 && !cfg.out_path.empty() && !cfg.pack_checkpoint &&
            hooks.table().n_tokens > 0 &&
            (hooks.table().n_tokens % cfg.checkpoint_every) == 0) {
            std::string err;
            checkpoint_prune_table(hooks.table(), cfg.out_path, &err);
        }
        if (cfg.abort && cfg.abort->load()) {
            break;
        }
    }
    streamer.end_session();
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double tps = elapsed > 0.0 ? static_cast<double>(n) / elapsed : 0.0;
    tel.set_generated(n, elapsed);
    std::fprintf(stderr, "%s\n", format_tokens_per_sec_line(tps, n, elapsed).c_str());
    std::fprintf(stderr, "%s", tel.format_report().c_str());
    if (cfg.on_stats) {
        cfg.on_stats(tps, n);
    }
    s.n_tokens = hooks.table().n_tokens;
    s.message = "stub hour (synthetic activations; not a 27B pass)";
    return s;
}

LiveForwardStatus LlamaCppLiveForwardBackend::probe(const LiveForwardConfig& cfg) {
    LiveForwardStatus s;
    s.backend = name();
    s.engine_linked = llama_cpp_linked();
    if (cfg.model_path.empty()) {
        s.message = "no --model path; cannot probe weights";
        return s;
    }
    GgufKv meta;
    std::string err;
    if (!read_gguf_meta(cfg.model_path, meta, &err)) {
        s.message = err.empty() ? "failed to read GGUF metadata" : err;
        return s;
    }
    s.architecture_ok = gguf_looks_like_qwen27b_hybrid(meta);
    if (!s.architecture_ok) {
        s.message = "GGUF is not Qwen 27B 3.6/3.8 hybrid (want arch qwen35, 64 layers, "
                    "d=5120, ffn=17408); got arch='" +
                    meta.architecture + "' n_layer=" + std::to_string(meta.n_layers) +
                    " n_embd=" + std::to_string(meta.n_embd) +
                    " n_ff=" + std::to_string(meta.n_ff);
        return s;
    }
    if (meta.has_vision_tensors && !cfg.load_vision) {
        s.message = "text-only job: vision tensors present in GGUF will be ignored "
                    "(do not pass an mmproj; do not set load_vision)";
    } else {
        s.message = "Qwen 27B hybrid metadata ok (arch=" + meta.architecture + ")";
    }
    s.ok = true;
    return s;
}

#if !defined(MICRO_LLM_HAS_LLAMA)
bool LlamaCppLiveForwardBackend::engine_linked() const { return false; }

LiveForwardStatus LlamaCppLiveForwardBackend::run(TraceHooks& /*hooks*/,
                                                  TraceStreamer& /*streamer*/,
                                                  const LiveForwardConfig& cfg) {
    LiveForwardStatus s = probe(cfg);
    s.engine_linked = false;
    s.ran_tokens = false;
    s.ok = false;
    if (cfg.model_path.empty()) {
        s.message = "llama.cpp backend: --model path.gguf is required and was not given";
        return s;
    }
    if (!s.architecture_ok && s.message.find("not Qwen") != std::string::npos) {
        s.message +=
            "; refusing to fake a 27B hour. llama.cpp is also not linked in this build.";
        return s;
    }
    s.message =
        "Qwen 27B hybrid GGUF recognized, but llama.cpp is not linked. "
        "Rebuild: cmake -S harness -B harness/build -DMICRO_LLM_LLAMA=ON "
        "-DMICRO_LLM_LLAMA_DIR=/path/to/llama.cpp "
        "(llama.cpp provides LLM_ARCH_QWEN35 / qwen35.cpp + cb_eval). "
        "Refusing to write a fake MLPT hour.";
    return s;
}
#endif

}  // namespace micro_llm
