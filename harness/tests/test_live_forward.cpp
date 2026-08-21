#include "test_common.hpp"

#include "micro_llm/micro_llm.hpp"

#include <cstdio>
#include <string>

void test_live_forward_backends(TestContext& ctx) {
    using namespace micro_llm;

    auto stub = make_live_forward("stub");
    CHECK(ctx, stub != nullptr);
    CHECK(ctx, stub->is_stub());
    CHECK(ctx, std::string(stub->name()) == "stub");

    auto llama = make_live_forward("llama");
    CHECK(ctx, llama != nullptr);
    CHECK(ctx, !llama->is_stub());
    CHECK(ctx, std::string(llama->name()) == "llama.cpp");
    CHECK(ctx, llama->engine_linked() == llama_cpp_linked());

    StreamerConfig scfg;
    scfg.ffn_scratch_bytes = 4096;
    TraceStreamer streamer(scfg);
    TraceHooks hooks;
    LiveForwardConfig cfg;
    cfg.n_predict = 2;
    const LiveForwardStatus stub_st = stub->run(hooks, streamer, cfg);
    CHECK(ctx, stub_st.ok);
    CHECK(ctx, stub_st.ran_tokens);
    CHECK(ctx, hooks.table().n_tokens == 2);
    CHECK(ctx, hooks.table().layer_was_hooked(0));
    CHECK(ctx, hooks.table().layer_was_hooked(63));

    LiveForwardConfig missing;
    missing.model_path = "";
    TraceHooks h2;
    TraceStreamer s2(scfg);
    const LiveForwardStatus no_model = llama->run(h2, s2, missing);
    CHECK(ctx, !no_model.ok);
    CHECK(ctx, !no_model.ran_tokens);
    CHECK(ctx, h2.table().n_tokens == 0);

    LiveForwardConfig bad;
    bad.model_path = "definitely-missing-27b.gguf";
    const LiveForwardStatus probed = llama->probe(bad);
    CHECK(ctx, !probed.ok);
    CHECK(ctx, !probed.architecture_ok);
}
