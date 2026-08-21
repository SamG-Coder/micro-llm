#include "test_common.hpp"

#include "micro_llm/micro_llm.hpp"

#include <cmath>
#include <vector>

void test_relative_spike_identity(TestContext& ctx) {
    using namespace micro_llm;

    CHECK(ctx, kDefaultSpikeEps == 0.02f);

    std::vector<float> v(16, 3.f);
    const float r0 = relative_residual_l2(v.data(), v.data(), 16);
    CHECK(ctx, r0 == 0.f);

    TraceHooks hooks;  // default spike_eps = 0.02
    CHECK(ctx, hooks.on_delta_hidden(0, v.data(), v.data(), 16));
    CHECK(ctx, hooks.table().pack_is_dead(0));
    CHECK(ctx, hooks.table().pack(0).n_spike == 0);
    CHECK(ctx, hooks.table().pack(0).sumsq_residual == 0.0);

    std::vector<float> out = v;
    out[0] = 3.f + 0.001f;  // tiny absolute L2 would have spiked vs 1e-6
    const float r_small = relative_residual_l2(v.data(), out.data(), 16);
    CHECK(ctx, r_small < 0.02f);
    CHECK(ctx, hooks.on_delta_hidden(1, v.data(), out.data(), 16));
    CHECK(ctx, hooks.table().pack_is_dead(1));

    out[0] = 10.f;
    const float r_big = relative_residual_l2(v.data(), out.data(), 16);
    CHECK(ctx, r_big > 0.02f);
    CHECK(ctx, hooks.on_delta_hidden(2, v.data(), out.data(), 16));
    CHECK(ctx, !hooks.table().pack_is_dead(2));
    CHECK(ctx, std::fabs(r_big - 7.f / (std::sqrt(16.0 * 9.0) + 1e-12)) < 1e-5f);
}
