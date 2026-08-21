#include "test_common.hpp"

#include "micro_llm/types.hpp"

#include <cstdint>

void test_serve_ok_refuses_unless_present_and_true(TestContext& ctx) {
    using namespace micro_llm;

    CHECK(ctx, remnant_serve_allowed(true, true));
    CHECK(ctx, !remnant_serve_allowed(true, false));   // --q4-k-to-f16 / not a Q4 remnant
    CHECK(ctx, !remnant_serve_allowed(false, true));   // key missing
    CHECK(ctx, !remnant_serve_allowed(false, false));
    CHECK(ctx, remnant_may_serve(true, true));
    CHECK(ctx, !remnant_may_serve(true, false));      // F16 host dump, refuse
    CHECK(ctx, !remnant_may_serve(false, true));
    CHECK(ctx, !remnant_may_serve(false, false));
    CHECK(ctx, remnant_may_serve(true, true) == remnant_serve_allowed(true, true));
    CHECK(ctx, kKvServeOk[0] == 'm');
}

void test_serve_stack_gate(TestContext& ctx) {
    using namespace micro_llm;

    const uint64_t w12 = 12ull * kGiB;
    const uint64_t w145 = (145ull * kGiB) / 10ull;
    const uint64_t ctx8k = 8192;
    const uint64_t ctx32k = 32768;

    // 12GB + 0.9 + 8k FP16 (0.5) = 13.4 < 15.2
    CHECK(ctx, remnant_serve_allowed(true, true, w12, ctx8k));
    // 14.5GB + 0.9 + 0.5 = 15.9 → refuse
    CHECK(ctx, !remnant_serve_allowed(true, true, w145, ctx8k));
    CHECK(ctx, !remnant_serve_allowed(true, false, w12, ctx8k));
    CHECK(ctx, !remnant_serve_allowed(false, true, w12, ctx8k));
    // 12GB + 32k FP16 (2) + 0.9 = 14.9 < 15.2
    CHECK(ctx, remnant_serve_allowed(true, true, w12, ctx32k));
    // same stack vs display 14.5 → refuse
    CHECK(ctx, !remnant_serve_allowed(true, true, w12, ctx32k, kCudaScratchBytes,
                                     kKvBytesPerTokenFp16, kServeUsableDisplayBytes));
    // Missing KV keys: defaults are the locked constants.
    CHECK(ctx, remnant_serve_allowed(true, true, w12, ctx8k) ==
                   remnant_serve_allowed(true, true, w12, ctx8k, kCudaScratchBytes,
                                         kKvBytesPerTokenFp16, kServeUsableBytes));
    CHECK(ctx, kCudaScratchBytes == (9ull * kGiB) / 10ull);
    CHECK(ctx, kKvBytesPerTokenFp16 == 65536ull);
    CHECK(ctx, kKvBytesPerTokenFp8 == 32768ull);
    CHECK(ctx, kServeUsableBytes == (152ull * kGiB) / 10ull);
}
