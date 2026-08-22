#include "test_common.hpp"

#include "micro_llm/micro_llm.hpp"

#include <string>
#include <vector>

void test_graph_hook_names(TestContext& ctx) {
    using namespace micro_llm;

    int layer = -2;
    CHECK(ctx, classify_graph_tensor("ffn_gate-3", &layer) == GraphHookSite::FfnGate);
    CHECK(ctx, layer == 3);
    CHECK(ctx, classify_graph_tensor("ffn_up-63", &layer) == GraphHookSite::FfnUp);
    CHECK(ctx, layer == 63);
    CHECK(ctx, classify_graph_tensor("ffn_gate_par-3", &layer) == GraphHookSite::FfnGatePar);
    CHECK(ctx, classify_graph_tensor("attn_residual-4", &layer) == GraphHookSite::AttnResidual);
    CHECK(ctx, layer == 4);
    CHECK(ctx, classify_graph_tensor("l_out-0", &layer) == GraphHookSite::LayerOut);
    CHECK(ctx, classify_graph_tensor("model.input_embed", &layer) == GraphHookSite::InputEmbed);
    CHECK(ctx, classify_graph_tensor("result_output", &layer) == GraphHookSite::Logits);
    CHECK(ctx, classify_graph_tensor("attn_norm-1", &layer) == GraphHookSite::None);

    StreamerConfig scfg;
    scfg.ffn_scratch_bytes = 4096;
    TraceStreamer streamer(scfg);
    streamer.begin_session();
    TraceHooks hooks;
    GraphHookSession sess(hooks, streamer);
    sess.begin_token(0);

    std::vector<float> gate(kFfnIntermediate, 0.f);
    std::vector<float> up(kFfnIntermediate, 0.f);
    gate[10] = 2.f;
    up[10] = 1.5f;

    GraphTensorView g;
    g.name = "ffn_gate-2";
    g.data = gate.data();
    g.ne0 = kFfnIntermediate;
    g.ne1 = 1;
    CHECK(ctx, sess.on_tensor(g, true));
    CHECK(ctx, streamer.resident_ffn_layers() == 1);
    CHECK(ctx, streamer.compute_layer() == 2);
    CHECK(ctx, streamer.ffn_vram_bytes() == scfg.ffn_scratch_bytes);
    CHECK(ctx, sess.on_tensor(g, false));

    GraphTensorView u;
    u.name = "ffn_up-2";
    u.data = up.data();
    u.ne0 = kFfnIntermediate;
    u.ne1 = 1;
    CHECK(ctx, sess.on_tensor(u, true));
    CHECK(ctx, sess.on_tensor(u, false));
    CHECK(ctx, streamer.resident_ffn_layers() == 0);
    CHECK(ctx, streamer.ffn_vram_bytes() == 0);
    CHECK(ctx, hooks.table().layer_was_hooked(2));
    CHECK(ctx, hooks.table().channel(2, 10).n_fired == 1);
    CHECK(ctx, sess.ffn_gate_hits() == 1);
    CHECK(ctx, sess.ffn_up_hits() == 1);
    CHECK(ctx, sess.host_ffn_binds() == 2);  // gate + up on host
    CHECK(ctx, sess.missing_hooks_this_token() == 63);

    std::vector<float> hin(kHiddenDim, 1.f);
    std::vector<float> hout(kHiddenDim, 1.f);
    hout[0] = 4.f;
    GraphTensorView lin;
    lin.name = "l_out-3";  // GA layer out becomes next residual in
    lin.data = hin.data();
    lin.ne0 = kHiddenDim;
    sess.on_tensor(lin, false);

    GraphTensorView res;
    res.name = "attn_residual-4";  // DeltaNet layer 4
    res.data = hout.data();
    res.ne0 = kHiddenDim;
    CHECK(ctx, sess.on_tensor(res, true));
    CHECK(ctx, sess.on_tensor(res, false));
    CHECK(ctx, sess.delta_hits() == 1);
    CHECK(ctx, !hooks.table().pack_is_dead(pack_id_from_delta_layer(4)));

    streamer.end_session();
}
