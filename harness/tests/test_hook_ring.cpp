#include "test_common.hpp"

#include "micro_llm/micro_llm.hpp"

#include <cstring>
#include <string>
#include <vector>

void test_hook_ring_encode(TestContext& ctx) {
    using namespace micro_llm;

    CHECK(ctx, kHtr1RecordBytes == 139744u);
    CHECK(ctx, kHtr1RingDepth == 64u);
    CHECK(ctx, kHtr1OffFfnFired == 280u);

    TraceHooks hooks;
    hooks.begin_token(9);
    std::vector<float> gate(kFfnIntermediate, 0.f);
    std::vector<float> up(kFfnIntermediate, 0.f);
    gate[0] = 2.f;
    up[0] = 1.5f;
    CHECK(ctx, hooks.on_ffn_activations(0, gate.data(), up.data()));
    gate.assign(kFfnIntermediate, 0.f);
    up.assign(kFfnIntermediate, 0.f);
    gate[17407] = 2.f;
    up[17407] = 1.5f;
    CHECK(ctx, hooks.on_ffn_activations(7, gate.data(), up.data()));
    CHECK(ctx, hooks.on_delta_residual(3, 0.05f));
    CHECK(ctx, hooks.on_delta_residual(4, 0.01f));
    CHECK(ctx, hooks.token_fired(0, 0));
    CHECK(ctx, hooks.token_fired(7, 17407));

    Htr1TokenMeta meta;
    meta.token_index = 9;
    meta.sampled_id = 21;
    meta.flags = kHtr1FlagSpecialOrHighLoss;
    meta.n_topk = 3;
    meta.topk[0] = 21;
    meta.topk[1] = 20;
    meta.topk[2] = 0;

    std::vector<uint8_t> rec(kHtr1RecordBytes, 0);
    CHECK(ctx, encode_htr1_record(hooks, meta, rec.data()));

    Htr1TokenMeta got;
    std::vector<uint8_t> fired(kHtr1FfnBitsetBytes, 0);
    float pack_rel[kNDeltaNetPacks] = {};
    uint64_t spike = 0;
    CHECK(ctx, decode_htr1_record(rec.data(), &got, fired.data(), pack_rel, &spike));
    CHECK(ctx, got.token_index == 9);
    CHECK(ctx, got.sampled_id == 21);
    CHECK(ctx, (got.flags & kHtr1FlagSpecialOrHighLoss) != 0);
    CHECK(ctx, got.n_topk == 3);
    CHECK(ctx, got.topk[0] == 21);
    CHECK(ctx, bit_test(fired.data(), channel_bit_index(0, 0)));
    CHECK(ctx, bit_test(fired.data(), channel_bit_index(7, 17407)));
    CHECK(ctx, !bit_test(fired.data(), channel_bit_index(1, 0)));
    CHECK(ctx, pack_rel[3] > 0.04f);
    CHECK(ctx, pack_rel[4] < 0.02f);
    CHECK(ctx, (spike & (uint64_t{1} << 3)) != 0);
    CHECK(ctx, (spike & (uint64_t{1} << 4)) == 0);

    bool pushed = false;
    CHECK(ctx, emit_htr1(hooks, 21, meta.topk, 3, true,
                         [&](const uint8_t* p, size_t n) {
                             pushed = n == kHtr1RecordBytes && p != nullptr;
                         }));
    CHECK(ctx, pushed);

    HookRing ring;
    for (uint32_t i = 0; i < 70; ++i) {
        rec[0] = static_cast<uint8_t>(i);
        ring.push(rec.data());
    }
    CHECK(ctx, ring.count() == kHtr1RingDepth);
    CHECK(ctx, ring.latest() != nullptr);

    // Checkpoint: scores only, atomic replace. No pack.
    hooks.after_logits(9, true);
    const std::string out = "hour_ckpt_test.mlpt";
    std::string err;
    CHECK(ctx, checkpoint_prune_table(hooks.table(), out, &err));
    CHECK(ctx, err.empty());
    PruneTable loaded;
    CHECK(ctx, load_prune_table(loaded, out, &err));
    CHECK(ctx, loaded.n_tokens == 1);
    CHECK(ctx, loaded.channel(0, 0).n_fired == 1);
    std::remove(out.c_str());
    std::remove((out + ".tmp").c_str());
}
