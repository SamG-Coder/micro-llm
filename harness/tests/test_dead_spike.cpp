#include "test_common.hpp"

#include "micro_llm/micro_llm.hpp"

#include <vector>

void test_dead_vs_spike_pack(TestContext& ctx) {
    using namespace micro_llm;

    TraceHooks hooks(1e-6f, 1e-4f);

    // Pack 0: never spikes ? dead.
    hooks.on_delta_residual(0, 0.f);
    hooks.on_delta_residual(0, 1e-8f);
    CHECK(ctx, hooks.table().pack_is_dead(0));
    CHECK(ctx, pack_is_dead(hooks.table().pack(0)));
    CHECK(ctx, hooks.table().pack(0).n_spike == 0);
    CHECK(ctx, hooks.table().pack(0).pack == 0);
    CHECK(ctx, hooks.table().pack(0).layer == 0);

    // Pack 1: residuals stay below eps ? still dead (identity-like hour).
    hooks.on_delta_residual(1, 1e-6f);
    CHECK(ctx, hooks.table().pack_is_dead(1));

    // Pack 2: one spike. Not dead, even if later tokens look like identity.
    hooks.on_delta_residual(2, 0.5f);
    for (int i = 0; i < 100; ++i) {
        hooks.on_delta_residual(2, 0.f);
    }
    CHECK(ctx, !hooks.table().pack_is_dead(2));
    CHECK(ctx, hooks.table().pack(2).n_spike == 1);
    CHECK(ctx, hooks.table().pack(2).sumsq_residual > 0.0);
    CHECK(ctx, hooks.table().pack(2).pack == 2);
    CHECK(ctx, hooks.table().pack(2).layer == 2);

    // Pack 3 via hidden vectors.
    std::vector<float> hin(8, 1.f);
    std::vector<float> hout = hin;
    hout[0] = 4.f;
    CHECK(ctx, hooks.on_delta_hidden(3, hin.data(), hout.data(), 8));
    CHECK(ctx, !hooks.table().pack_is_dead(3));
    CHECK(ctx, hooks.table().pack(3).layer == delta_layer_from_pack_id(3));
    CHECK(ctx, hooks.table().pack(3).layer == 4);  // group 1 slot 0 ? layer 4

    // Pack 47 exists and starts dead.
    CHECK(ctx, hooks.table().pack(47).pack == 47);
    CHECK(ctx, hooks.table().pack(47).layer == 62);
    CHECK(ctx, hooks.table().pack_is_dead(47));
}
