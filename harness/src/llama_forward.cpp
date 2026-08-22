#include "micro_llm/live_forward.hpp"

#if defined(MICRO_LLM_HAS_LLAMA)

#include "micro_llm/gguf_meta.hpp"
#include "micro_llm/graph_hooks.hpp"
#include "micro_llm/trace_cli.hpp"

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
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

const float* tensor_f32_host(ggml_tensor* t, LlamaHookUser* u, bool* on_device) {
    *on_device = false;
    if (!t) {
        return nullptr;
    }
    const size_t n = static_cast<size_t>(ggml_nelements(t));
    if (tensor_is_f32(t) && tensor_on_host(t) && t->data) {
        return static_cast<const float*>(t->data);
    }
    if (!tensor_is_f32(t)) {
        // Quantized / non-F32 activations are unexpected; refuse the tap.
        return nullptr;
    }
    // CUDA F32: D2H. Never treat a device pointer as host (prior 0xC0000005).
    u->host.resize(n);
    ggml_backend_tensor_get(t, u->host.data(), 0, ggml_nbytes(t));
    return u->host.data();
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
    v.data = tensor_f32_host(t, u, &on_device);
    v.on_device = on_device;
    return u->session->on_tensor(v, false);
}

void topk_ids(const float* logits, int32_t n_vocab, uint32_t k, uint32_t* out) {
    k = std::min(k, static_cast<uint32_t>(n_vocab));
    std::vector<int32_t> idx(static_cast<size_t>(n_vocab));
    for (int32_t i = 0; i < n_vocab; ++i) {
        idx[static_cast<size_t>(i)] = i;
    }
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int32_t a, int32_t b) {
        return logits[a] > logits[b];
    });
    for (uint32_t i = 0; i < k; ++i) {
        out[i] = static_cast<uint32_t>(idx[i]);
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

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    // Hybrid pin: ngl=99 (CUDA by default). NOT ngl=16 — that is the first
    // 16 layers (12 DeltaNet + 4 GA), the wrong 16, and would park ~16.5GB
    // on the card if someone meant "whole file".
    mparams.n_gpu_layers = cfg.n_gpu_layers;
    if (mparams.n_gpu_layers == 16) {
        mparams.n_gpu_layers = hybrid_n_gpu_layers();
    }
    mparams.load_mtp = cfg.load_mtp;

    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    const std::vector<std::string> cpu_pats = hybrid_cpu_tensor_regexes();
    std::vector<llama_model_tensor_buft_override> buft_ovs;
    buft_ovs.reserve(cpu_pats.size() + 1);
    for (const std::string& pat : cpu_pats) {
        llama_model_tensor_buft_override ov{};
        ov.pattern = pat.c_str();
        ov.buft = cpu_buft;
        buft_ovs.push_back(ov);
    }
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
    if (n_layer != static_cast<int32_t>(kNLayers) ||
        n_embd != static_cast<int32_t>(kHiddenDim)) {
        llama_model_free(model);
        llama_backend_free();
        st.ok = false;
        st.architecture_ok = false;
        st.message = "loaded model is not 64 x 5120 Qwen 27B hybrid";
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
    // flash-attn + CPU FFN split has AVed; disable FA and op_offload.
    if (cfg.disable_flash_attn) {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    if (cfg.disable_op_offload) {
        cparams.op_offload = false;
    }
    cparams.offload_kqv = true;  // KV stays CUDA with the 16 GA blocks
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
                                   tokens.data(), static_cast<int32_t>(tokens.size()), true,
                                   true);
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
    hooks.mark_reserved_core(256);
    for (llama_token id : tokens) {
        if (id >= 0 && static_cast<uint32_t>(id) < kVocabSize) {
            hooks.on_vocab_id(static_cast<uint32_t>(id));
        }
    }

    // Prefill in chunks so activations stay in budget.
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

    uint32_t generated = 0;
    const uint32_t want = cfg.n_predict;
    llama_token last = tokens.back();
    while (generated < want) {
        float* logits = llama_get_logits_ith(ctx, -1);
        if (!logits) {
            st.message = "llama_get_logits_ith returned null";
            break;
        }
        std::vector<uint32_t> top(cfg.top_k ? cfg.top_k : 1u);
        topk_ids(logits, n_vocab, static_cast<uint32_t>(top.size()), top.data());
        last = static_cast<llama_token>(top[0]);
        const bool special = llama_vocab_is_eog(vocab, last) ||
                             llama_vocab_is_control(vocab, last);
        session.finish_token(static_cast<uint32_t>(last), top.data(),
                             static_cast<uint32_t>(top.size()), special);
        // Continue after EOS. Do not stop the dump at the first EOG.
        if (static_cast<uint32_t>(last) < kVocabSize) {
            hooks.on_vocab_id(static_cast<uint32_t>(last));
        }
        if (cfg.checkpoint_every != 0 && !cfg.out_path.empty() &&
            hooks.table().n_tokens > 0 &&
            (hooks.table().n_tokens % cfg.checkpoint_every) == 0) {
            std::string err;
            checkpoint_prune_table(hooks.table(), cfg.out_path, &err);
        }
        if (cfg.abort && cfg.abort->load()) {
            break;
        }
        if (!cfg.continue_after_eos && llama_vocab_is_eog(vocab, last)) {
            break;
        }
        session.begin_token(static_cast<uint32_t>(n_tok + generated));
        llama_batch batch = llama_batch_get_one(&last, 1);
        const int rc = llama_decode(ctx, batch);
        if (rc != 0) {
            st.message = "llama_decode failed during generate (rc=" + std::to_string(rc) + ")";
            break;
        }
        ++generated;
    }

    streamer.end_session();
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    st.ok = hooks.table().n_tokens > 0;
    st.ran_tokens = st.ok;
    st.n_tokens = hooks.table().n_tokens;
    if (st.ok) {
        st.message = "llama.cpp qwen35 graph eval completed; MLPT is a real hour trace";
    } else if (st.message.empty()) {
        st.message = "llama.cpp ran but after_logits never fired; refusing to fake n_fired";
    }
    return st;
}

}  // namespace micro_llm

#endif  // MICRO_LLM_HAS_LLAMA
