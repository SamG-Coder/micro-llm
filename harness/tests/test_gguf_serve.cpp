#include "test_common.hpp"

#include "micro_llm/micro_llm.hpp"

#include <cstdio>
#include <string>
#include <vector>

void test_gguf_serve_gate(TestContext& ctx) {
    using namespace micro_llm;

    const std::string ok_path = "test_serve_ok_true.gguf";
    const std::string false_path = "test_serve_ok_false.gguf";
    const std::string missing_path = "test_serve_ok_missing.gguf";

    CHECK(ctx, write_gguf_kv_stub(ok_path, "qwen35", true, true));
    CHECK(ctx, write_gguf_kv_stub(false_path, "qwen35", true, false));
    CHECK(ctx, write_gguf_kv_stub(missing_path, "qwen35", false, false));

    GgufKv meta;
    CHECK(ctx, read_gguf_meta(ok_path, meta));
    CHECK(ctx, meta.architecture == "qwen35");
    CHECK(ctx, meta.n_layers == kNLayers);
    CHECK(ctx, meta.n_embd == kHiddenDim);
    CHECK(ctx, meta.n_ff == kFfnIntermediate);
    CHECK(ctx, gguf_looks_like_qwen27b_hybrid(meta));
    CHECK(ctx, gguf_hook_layer_count(meta) == kNLayers);

    const std::string mtp_path = "test_serve_mtp65.gguf";
    CHECK(ctx, write_gguf_kv_stub(mtp_path, "qwen35", true, true, kNLayers + 1, kHiddenDim,
                                 kFfnIntermediate, nullptr, nullptr, 1));
    GgufKv mtp;
    CHECK(ctx, read_gguf_meta(mtp_path, mtp));
    CHECK(ctx, mtp.n_layers == kNLayers + 1);
    CHECK(ctx, mtp.n_nextn == 1);
    CHECK(ctx, gguf_looks_like_qwen27b_hybrid(mtp));
    CHECK(ctx, gguf_hook_layer_count(mtp) == kNLayers);
    CHECK(ctx, hook_layer_count_from_blocks(mtp.n_layers) == kNLayers);
    CHECK(ctx, file_exists(mtp_path));
    CHECK(ctx, !file_exists("this_gguf_does_not_exist_15gib.gguf"));
    std::remove(mtp_path.c_str());
    CHECK(ctx, !file_exists(mtp_path));
    CHECK(ctx, meta.serve_ok_present);
    CHECK(ctx, meta.serve_ok);

    CHECK(ctx, remnant_may_serve_file(ok_path));
    CHECK(ctx, !remnant_may_serve_file(false_path));
    CHECK(ctx, !remnant_may_serve_file(missing_path));

    const std::string w13056 = "test_serve_ffn_13056.gguf";
    const std::string w10496 = "test_serve_ffn_10496.gguf";
    const std::string w10445 = "test_serve_ffn_10445.gguf";
    const std::vector<uint32_t> keep_13056(kNLayers, kWeakKeepMin27B);
    const std::vector<uint32_t> keep_10496(kNLayers, kWeakKeepMinRecover27B);
    const std::vector<uint32_t> keep_10445(kNLayers, 10445u);
    CHECK(ctx, write_gguf_kv_stub(w13056, "qwen35", true, true, kNLayers, kHiddenDim,
                                 kFfnIntermediate, nullptr, &keep_13056));
    CHECK(ctx, write_gguf_kv_stub(w10496, "qwen35", true, true, kNLayers, kHiddenDim,
                                 kFfnIntermediate, nullptr, &keep_10496));
    CHECK(ctx, write_gguf_kv_stub(w10445, "qwen35", true, true, kNLayers, kHiddenDim,
                                 kFfnIntermediate, nullptr, &keep_10445));
    CHECK(ctx, remnant_may_serve_file(w13056));
    CHECK(ctx, remnant_may_serve_file(w10496));
    CHECK(ctx, !remnant_may_serve_file(w10445));
    const std::string nff13056 = "test_serve_nff_13056.gguf";
    const std::string nff10445 = "test_serve_nff_10445.gguf";
    CHECK(ctx, write_gguf_kv_stub(nff13056, "qwen35", true, true, kNLayers, kHiddenDim,
                                 13056u));
    CHECK(ctx, write_gguf_kv_stub(nff10445, "qwen35", true, true, kNLayers, kHiddenDim,
                                 10445u));
    CHECK(ctx, remnant_may_serve_file(nff13056));
    CHECK(ctx, !remnant_may_serve_file(nff10445));
    std::remove(nff13056.c_str());
    std::remove(nff10445.c_str());
    const ServeGate g10445 = read_serve_gate(w10445);
    CHECK(ctx, g10445.key_present && g10445.serve_ok);
    CHECK(ctx, !g10445.ffn_width_ok);
    CHECK(ctx, !g10445.may_serve);
    CHECK(ctx, g10445.reason.find("256") != std::string::npos);
    std::remove(w13056.c_str());
    std::remove(w10496.c_str());
    std::remove(w10445.c_str());

    const ServeGate gfalse = read_serve_gate(false_path);
    CHECK(ctx, gfalse.key_present);
    CHECK(ctx, !gfalse.serve_ok);
    CHECK(ctx, !gfalse.may_serve);
    CHECK(ctx, gfalse.reason.find("F16") != std::string::npos);

    const ServeGate gmiss = read_serve_gate(missing_path);
    CHECK(ctx, !gmiss.key_present);
    CHECK(ctx, !gmiss.may_serve);

    auto llama = make_live_forward("llama");
    LiveForwardConfig cfg;
    cfg.model_path = ok_path;
    const LiveForwardStatus st = llama->probe(cfg);
    CHECK(ctx, st.architecture_ok);
    // KV-only stub has no weights. run() must not fake an hour.
    StreamerConfig scfg;
    scfg.ffn_scratch_bytes = 4096;
    TraceStreamer streamer(scfg);
    TraceHooks hooks;
    const LiveForwardStatus run = llama->run(hooks, streamer, cfg);
    if (!llama_cpp_linked()) {
        CHECK(ctx, !run.ok);
        CHECK(ctx, !run.ran_tokens);
        CHECK(ctx, hooks.table().n_tokens == 0);
        CHECK(ctx, run.message.find("not linked") != std::string::npos);
    }

    std::remove(ok_path.c_str());
    std::remove(false_path.c_str());
    std::remove(missing_path.c_str());
}
