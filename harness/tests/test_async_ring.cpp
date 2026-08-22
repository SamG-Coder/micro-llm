#include "test_common.hpp"

#include "micro_llm/micro_llm.hpp"

#include <memory>
#include <vector>

void test_async_htr1_ring(TestContext& ctx) {
    using namespace micro_llm;

    CHECK(ctx, kUiDrainPeriodMs >= 1000u / kUiDrainHzMax);
    CHECK(ctx, kUiDrainPeriodMs <= 1000u / kUiDrainHzMin + 2);

    // Ring holds 64 * 140KB. Do not put it on the stack.
    auto ring = std::make_unique<AsyncHtr1Ring>();
    std::vector<uint8_t> rec(kHtr1RecordBytes, 0);
    rec[0] = 7;
    CHECK(ctx, ring->try_push(rec.data()));
    CHECK(ctx, ring->size() == 1);
    CHECK(ctx, ring->pushed() == 1);
    std::vector<uint8_t> out(kHtr1RecordBytes, 0);
    CHECK(ctx, ring->try_pop(out.data()));
    CHECK(ctx, out[0] == 7);
    CHECK(ctx, !ring->try_pop(out.data()));

    for (uint32_t i = 0; i < AsyncHtr1Ring::kDepth + 8; ++i) {
        rec[0] = static_cast<uint8_t>(i);
        ring->try_push(rec.data());
    }
    CHECK(ctx, ring->size() == AsyncHtr1Ring::kDepth);
    CHECK(ctx, ring->drops() > 0);

    std::vector<uint8_t> batch(3 * kHtr1RecordBytes, 0);
    const uint32_t n = ring->drain(batch.data(), 3);
    CHECK(ctx, n == 3);

    LiveStatsAtomics stats;
    stats.tok_s_milli.store(21500);
    CHECK(ctx, stats.tok_s_milli.load() == 21500);
}
