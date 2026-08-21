#include "test_common.hpp"

#include "micro_llm/types.hpp"

void test_serve_ok_refuses_unless_present_and_true(TestContext& ctx) {
    using namespace micro_llm;

    CHECK(ctx, remnant_serve_allowed(true, true));
    CHECK(ctx, !remnant_serve_allowed(true, false));   // --q4-k-to-f16 / not a Q4 remnant
    CHECK(ctx, !remnant_serve_allowed(false, true));   // key missing
    CHECK(ctx, !remnant_serve_allowed(false, false));
    CHECK(ctx, kKvServeOk[0] == 'm');
}
