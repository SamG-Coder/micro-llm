#include "test_common.hpp"

#include "micro_llm/micro_llm.hpp"

#include <cstdio>
#include <string>

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
    CHECK(ctx, meta.serve_ok_present);
    CHECK(ctx, meta.serve_ok);

    CHECK(ctx, remnant_may_serve_file(ok_path));
    CHECK(ctx, !remnant_may_serve_file(false_path));
    CHECK(ctx, !remnant_may_serve_file(missing_path));

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
