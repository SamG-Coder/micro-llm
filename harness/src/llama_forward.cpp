#include "micro_llm/live_forward.hpp"

#if defined(MICRO_LLM_HAS_LLAMA)

#include "micro_llm/ffn_reduce.hpp"
#include "micro_llm/gguf_meta.hpp"
#include "micro_llm/graph_hooks.hpp"
#include "micro_llm/perf_telemetry.hpp"
#include "micro_llm/residency.hpp"
#include "micro_llm/trace_cli.hpp"

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace micro_llm {
namespace {

struct LlamaHookUser {
    GraphHookSession* session = nullptr;
    std::vector<float> host;
};

bool tensor_is_f32(const ggml_tensor* t) { return t && t->type == GGML_TYPE_F32; }

bool tensor_on_host(const ggml_tensor* t) {
    return t && t->buffer && ggml_backend_buffer_is_host(t->buffer);
}

// Live view. Device F32 stays on the card for the fire tap. Do not D2H
// the activation (and never dequant the FFN weights) just to score it.
const float* tensor_f32_view(ggml_tensor* t, LlamaHookUser* /*u*/, bool* on_device) {
    *on_device = false;
    if (!t || !tensor_is_f32(t) || !t->data) {
        return nullptr;
    }
    if (tensor_on_host(t)) {
        return static_cast<const float*>(t->data);
    }
    *on_device = true;
    return static_cast<const float*>(t->data);
}

ggml_backend_buffer_type_t buft_from_reg(const char* name) {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name(name);
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
        return nullptr;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    return dev ? ggml_backend_dev_buffer_type(dev) : nullptr;
}

ggml_backend_buffer_type_t hybrid_gpu_buft() {
    if (ggml_backend_buffer_type_t t = buft_from_reg("CUDA")) {
        return t;
    }
    if (ggml_backend_buffer_type_t t = buft_from_reg("cuda")) {
        return t;
    }
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            return ggml_backend_dev_buffer_type(dev);
        }
    }
    return nullptr;
}

ggml_backend_buffer_type_t hybrid_cuda_host_buft() {
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        if (ggml_backend_buffer_type_t t = ggml_backend_dev_host_buffer_type(dev)) {
            return t;
        }
    }
    return nullptr;
}

bool llama_cb_eval(struct ggml_tensor* t, bool ask, void* user_data) {
    auto* u = static_cast<LlamaHookUser*>(user_data);
    if (!u || !u->session || !t) {
        return false;
    }
    GraphTensorView v;
    v.name = t->name;
    v.ne0 = t->ne[0] > 0 ? static_cast<uint32_t>(t->ne[0]) : 0;
    v.ne1 = t->ne[1] > 0 ? static_cast<uint32_t>(t->ne[1]) : 1;
    if (ask) {
        return u->session->on_tensor(v, true);
    }
    bool on_device = false;
    v.data = tensor_f32_view(t, u, &on_device);
    v.on_device = on_device;
    return u->session->on_tensor(v, false);
}

void topk_ids(const float* logits, int32_t n_vocab, uint32_t k, uint32_t* out) {
    k = std::min(k, static_cast<uint32_t>(n_vocab));
    // Linear keep-k. Avoids partial_sort of 248k ids every token.
    std::vector<int32_t> best(k, 0);
    for (uint32_t i = 0; i < k; ++i) {
        best[i] = static_cast<int32_t>(i);
    }
    auto worse = [&](int32_t a, int32_t b) { return logits[a] < logits[b]; };
    for (int32_t i = static_cast<int32_t>(k); i < n_vocab; ++i) {
        uint32_t lo = 0;
        for (uint32_t j = 1; j < k; ++j) {
            if (worse(best[j], best[lo])) {
                lo = j;
            }
        }
        if (logits[i] > logits[best[lo]]) {
            best[lo] = i;
        }
    }
    std::sort(best.begin(), best.end(), [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    for (uint32_t i = 0; i < k; ++i) {
        out[i] = static_cast<uint32_t>(best[i]);
    }
}

}  // namespace

bool LlamaCppLiveForwardBackend::engine_linked() const { return true; }

LiveForwardStatus LlamaCppLiveForwardBackend::run(TraceHooks& hooks, TraceStreamer& streamer,
                                                  const LiveForwardConfig& cfg) {
    LiveForwardStatus st = probe(cfg);
    st.engine_linked = true;
    if (cfg.model_path.empty()) {
        st.ok = false;
        st.message = "llama.cpp backend: --model path.gguf is required";
        return st;
    }
    if (!st.architecture_ok) {
        st.ok = false;
        st.ran_tokens = false;
        return st;
    }
    if (cfg.load_vision) {
        st.ok = false;
        st.message = "refusing to load vision (coding-assistant job is text-only)";
        return st;
    }

    WeightCatalog cat = default_qwen27b_q4km_catalog();
    std::string cat_err;
    if (!read_gguf_weight_catalog(cfg.model_path, cat, &cat_err)) {
        cat = default_qwen27b_q4km_catalog();
        cat.source = "default-after-header-fail";
    }
    ResidencyPlan plan = plan_residency(cat, cfg.n_ctx, cfg.force_host_deltanet);
    if (cfg.n_parked_ffn != 0) {
        plan.n_parked_ffn = cfg.n_parked_ffn < kNLayers ? cfg.n_parked_ffn : (kNLayers - 1u);
        for (uint32_t i = 0; i < kNLayers; ++i) {
            plan.ffn_parked[i] = i < plan.n_parked_ffn;
        }
        plan.n_streamed_ffn = kNLayers - plan.n_parked_ffn;
    }
    if (cfg.disable_flash_attn) {
        plan.enable_flash_attn = false;
    }
    if (!cfg.use_quant_kv) {
        plan.use_quant_kv = false;
        plan.kv_bytes = static_cast<uint64_t>(plan.n_ctx) * kKvBytesPerTokenFp16;
    }

    auto& tel = PerfTelemetry::thread_local_instance();
    tel.reset();
    tel.apply_plan(plan);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = clamp_hybrid_n_gpu_layers(cfg.n_gpu_layers);
    mparams.load_mtp = false;
    mparams.use_mlock = true;

    const uint32_t n_park = plan.n_parked_ffn;
    streamer.set_n_parked_ffn(n_park);
    streamer.set_log_cuda_ffn(true);
    const uint64_t park_bytes = streamer.parked_ffn_bytes();
    std::fprintf(stderr, "%s\n", format_ffn_cuda_park_line(n_park, park_bytes).c_str());
    std::fprintf(stderr, "%s\n", format_residency_plan(plan).c_str());
    std::fprintf(stderr,
                 "ffn_cuda_workspace expected_ffn_mib=%.2f stream_slot_mib=%.2f "
                 "(not ngl=99, not op_offload, DeltaNet=CUDA)\n",
                 static_cast<double>(park_bytes) / (1024.0 * 1024.0),
                 static_cast<double>(kQ4FfnLayerBytes) / (1024.0 * 1024.0));

    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_type_t gpu_buft = hybrid_gpu_buft();
    ggml_backend_buffer_type_t host_buft = hybrid_cuda_host_buft();
    const std::vector<std::string> gpu_pats = residency_vram_regexes(plan);
    const std::vector<std::string> host_pats = residency_cuda_host_regexes(plan);
    const std::vector<std::string> cpu_pats = residency_cpu_regexes(plan);
    std::vector<llama_model_tensor_buft_override> buft_ovs;
    buft_ovs.reserve(gpu_pats.size() + host_pats.size() + cpu_pats.size() + 1);
    auto push_ovs = [&](const std::vector<std::string>& pats, ggml_backend_buffer_type_t buft) {
        if (!buft) {
            return;
        }
        for (const std::string& pat : pats) {
            llama_model_tensor_buft_override ov{};
            ov.pattern = pat.c_str();
            ov.buft = buft;
            buft_ovs.push_back(ov);
        }
    };
    push_ovs(gpu_pats, gpu_buft);
    push_ovs(host_pats, host_buft ? host_buft : cpu_buft);
    push_ovs(cpu_pats, cpu_buft);
    llama_model_tensor_buft_override ov_end{};
    ov_end.pattern = nullptr;
    ov_end.buft = nullptr;
    buft_ovs.push_back(ov_end);
    mparams.tensor_buft_overrides = buft_ovs.data();

    llama_model* model = llama_model_load_from_file(cfg.model_path.c_str(), mparams);
    if (!model) {
        llama_backend_free();
        st.ok = false;
        st.message = "llama.cpp failed to load GGUF (weights missing or architecture "
                     "unsupported by this llama.cpp build)";
        return st;
    }

    const int32_t n_layer = llama_model_n_layer(model);
    const int32_t n_embd = llama_model_n_embd(model);
    const uint32_t hook_layers = hook_layer_count_from_blocks(static_cast<uint32_t>(n_layer));
    if (hook_layers != kNLayers || n_embd != static_cast<int32_t>(kHiddenDim)) {
        llama_model_free(model);
        llama_backend_free();
        st.ok = false;
        st.architecture_ok = false;
        st.message = "loaded model is not 64 x 5120 Qwen 27B hybrid "
                     "(MTP extra block is ok; hook ring stays 64)";
        return st;
    }

    GraphHookSession session(hooks, streamer);
    if (cfg.on_htr1) {
        session.set_on_htr1(cfg.on_htr1);
    }
    LlamaHookUser user;
    user.session = &session;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = cfg.n_ctx;
    cparams.n_batch = cfg.n_batch ? cfg.n_batch : 512;
    cparams.n_ubatch = cfg.n_ubatch ? cfg.n_ubatch : 32;
    // FA is safe only when FFN+DeltaNet compute is CUDA. FA + CPU FFN AVed.
    if (!plan.enable_flash_attn) {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    cparams.op_offload = false;
    cparams.offload_kqv = true;
    cparams.cb_eval = llama_cb_eval;
    cparams.cb_eval_user_data = &user;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        llama_backend_free();
        st.ok = false;
        st.message = "llama.cpp failed to create context (OOM or missing DeltaNet kernels)";
        return st;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const std::string& prompt =
        cfg.prompt.empty() ? std::string(default_coding_assistant_prompt()) : cfg.prompt;

    std::vector<llama_token> tokens(prompt.size() + 32);
    int32_t n_tok = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                                   tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
    if (n_tok < 0) {
        tokens.resize(static_cast<size_t>(-n_tok));
        n_tok = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                               tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
    }
    if (n_tok <= 0) {
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        st.ok = false;
        st.message = "tokenize failed";
        return st;
    }
    tokens.resize(static_cast<size_t>(n_tok));

    streamer.begin_session();
    tel.set_host_pages_pinned(streamer.host_pages_pinned());
    hooks.mark_reserved_core(256);
    for (llama_token id : tokens) {
        if (id >= 0 && static_cast<uint32_t>(id) < kVocabSize) {
            hooks.on_vocab_id(static_cast<uint32_t>(id));
        }
    }

    const auto pre_t0 = std::chrono::steady_clock::now();
    const int32_t chunk = 512;
    int32_t consumed = 0;
    while (consumed < n_tok) {
        const int32_t n = std::min(chunk, n_tok - consumed);
        llama_batch batch = llama_batch_get_one(tokens.data() + consumed, n);
        session.begin_token(static_cast<uint32_t>(consumed));
        const int rc = llama_decode(ctx, batch);
        if (rc != 0) {
            streamer.end_session();
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            st.ok = false;
            st.message = "llama_decode failed during prefill (rc=" + std::to_string(rc) +
                         "). This llama.cpp build cannot run the Qwen 27B hybrid graph.";
            return st;
        }
        consumed += n;
        if (cfg.abort && cfg.abort->load()) {
            streamer.end_session();
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            st.ok = false;
            st.message = "hour aborted";
            return st;
        }
    }
    tel.set_prefill(static_cast<uint32_t>(n_tok),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - pre_t0).count());

    uint32_t generated = 0;
    const uint32_t want = cfg.n_predict;
    llama_token last = tokens.back();
    const auto gen_t0 = std::chrono::steady_clock::now();
    auto report_tps = [&]() {
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - gen_t0).count();
        const double tps = elapsed > 0.0 ? static_cast<double>(generated) / elapsed : 0.0;
        tel.set_generated(generated, elapsed);
        std::fprintf(stderr, "%s\n", format_tokens_per_sec_line(tps, generated, elapsed).c_str());
        if (cfg.on_stats) {
            cfg.on_stats(tps, generated);
        }
    };
    while (generated < want) {
        const uint64_t sample_t0 = now_ns();
        float* logits = llama_get_logits_ith(ctx, -1);
        if (!logits) {
            st.message = "llama_get_logits_ith returned null";
            break;
        }
        std::vector<uint32_t> top(cfg.top_k ? cfg.top_k : 1u);
        topk_ids(logits, n_vocab, static_cast<uint32_t>(top.size()), top.data());
        last = static_cast<llama_token>(top[0]);
        tel.snap().sample_ns += now_ns() - sample_t0;
        const bool special =
            llama_vocab_is_eog(vocab, last) || llama_vocab_is_control(vocab, last);
        const uint64_t tr0 = now_ns();
        session.finish_token(static_cast<uint32_t>(last), top.data(),
                             static_cast<uint32_t>(top.size()), special);
        tel.snap().trace_encode_ns += now_ns() - tr0;
        if (static_cast<uint32_t>(last) < kVocabSize) {
            hooks.on_vocab_id(static_cast<uint32_t>(last));
        }
        if (cfg.checkpoint_every != 0 && !cfg.out_path.empty() && !cfg.pack_checkpoint &&
            hooks.table().n_tokens > 0 && (hooks.table().n_tokens % cfg.checkpoint_every) == 0) {
            std::string err;
            checkpoint_prune_table(hooks.table(), cfg.out_path, &err);
        }
        if (generated != 0 && (generated % 32u) == 0) {
            report_tps();
        }
        if (cfg.abort && cfg.abort->load()) {
            break;
        }
        if (!cfg.continue_after_eos && llama_vocab_is_eog(vocab, last)) {
            break;
        }
        session.begin_token(static_cast<uint32_t>(n_tok + generated));
        llama_batch batch = llama_batch_get_one(&last, 1);
        const uint64_t d0 = now_ns();
        const int rc = llama_decode(ctx, batch);
        tel.snap().decode_ns += now_ns() - d0;
        if (rc != 0) {
            st.message = "llama_decode failed during generate (rc=" + std::to_string(rc) + ")";
            break;
        }
        ++generated;
    }
    report_tps();
    tel.mark_measured();
    tel.add_d2h(persistent_cuda_reduce().d2h_bytes());
    tel.add_sync(persistent_cuda_reduce().sync_count());
    std::fprintf(stderr, "%s", tel.format_report().c_str());

    streamer.end_session();
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    st.ok = hooks.table().n_tokens > 0;
    st.ran_tokens = st.ok;
    st.n_tokens = hooks.table().n_tokens;
    if (st.ok) {
        st.message = "llama.cpp qwen35 graph eval completed; MLPT is a real hour trace "
                     "(GA+DeltaNet+parked FFN on CUDA, streamed FFN CUDA_Host, device fire tap)";
    } else if (st.message.empty()) {
        st.message = "llama.cpp ran but after_logits never fired; refusing to fake n_fired";
    }
    return st;
}

}  // namespace micro_llm

#endif  // MICRO_LLM_HAS_LLAMA
