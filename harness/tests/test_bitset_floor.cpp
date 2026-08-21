#include "test_common.hpp"

#include "micro_llm/micro_llm.hpp"

#include <vector>

void test_bitset_floor_or_after_logits(TestContext& ctx) {
    using namespace micro_llm;

    CHECK(ctx, kFloorBitsetBytes == (64u * 17408u) / 8u);
    CHECK(ctx, kFloorBitsetBytes == 139264u);

    TraceHooks hooks;
    std::vector<float> abs_act(kFfnIntermediate, 0.f);
    abs_act[7] = 1.25f;
    abs_act[100] = 0.5f;

    hooks.begin_token(0);
    CHECK(ctx, hooks.on_ffn_abs(5, abs_act.data()));
    CHECK(ctx, hooks.token_fired(5, 7));
    CHECK(ctx, hooks.token_fired(5, 100));
    // Floor cannot be decided at the FFN hook.
    CHECK(ctx, !hooks.table().floor_keep(5, 7));
    CHECK(ctx, !hooks.table().floor_keep(5, 100));
    CHECK(ctx, hooks.table().channel(5, 7).n_fired == 1);

    hooks.after_logits(0, false);  // ordinary token: do not OR into floor
    CHECK(ctx, !hooks.table().floor_keep(5, 7));
    CHECK(ctx, !hooks.table().floor_keep(5, 100));
    CHECK(ctx, !hooks.token_fired(5, 7));  // per-token bitset cleared
    CHECK(ctx, hooks.table().n_tokens == 1);

    hooks.begin_token(1);
    CHECK(ctx, hooks.on_ffn_abs(5, abs_act.data()));
    hooks.after_logits(1, true);  // high-loss: OR into floor
    CHECK(ctx, hooks.table().floor_keep(5, 7));
    CHECK(ctx, hooks.table().floor_keep(5, 100));
    CHECK(ctx, !hooks.table().floor_keep(5, 0));
    CHECK(ctx, hooks.table().channel(5, 7).n_fired == 2);

    // Special token on another layer.
    std::vector<float> abs2(kFfnIntermediate, 0.f);
    abs2[0] = 3.0f;
    hooks.begin_token(2);
    CHECK(ctx, hooks.on_ffn_abs(2, abs2.data()));
    CHECK(ctx, !hooks.table().floor_keep(2, 0));
    hooks.after_logits(2, true);
    CHECK(ctx, hooks.table().floor_keep(2, 0));
    CHECK(ctx, hooks.table().floor_keep(5, 7));  // previous floor bits stay
}
