#include "test_common.hpp"

#include "micro_llm/micro_llm.hpp"

#include <cmath>
#include <vector>

void test_channel_alignment(TestContext& ctx) {
    using namespace micro_llm;

    CHECK(ctx, kFfnIntermediate == 17408);
    CHECK(ctx, kTensorAlign == 256);
    CHECK(ctx, ChannelStat::kPackedAlign == 256);
    CHECK(ctx, PackStat::kPackedAlign == 256);
    CHECK(ctx, ChannelIndex::kPackedAlign == 256);
    CHECK(ctx, PruneTable::kPackedAlign == 256);
    CHECK(ctx, TraceHooks::kPackedAlign == 256);
    CHECK(ctx, TraceStreamer::kPackedAlign == 256);
    CHECK(ctx, align_up(1) == 256);
    CHECK(ctx, align_up(256) == 256);
    CHECK(ctx, align_up(257) == 512);

    for (uint32_t c : {0u, 1u, 17u, 17407u}) {
        const ChannelTriplet t = channel_triplet(c);
        CHECK(ctx, t.gate == c && t.up == c && t.down == c);
        CHECK(ctx, same_channel_across_gate_up_down(t));
    }

    // Reduce uses the same index for gate and up; export will gather down[c].
    std::vector<float> gate(8, 0.f);
    std::vector<float> up(8, 0.f);
    std::vector<float> abs_out(8, 0.f);
    std::vector<uint8_t> bits(1, 0);
    gate[3] = 2.0f;
    up[3] = 3.0f;
    const uint32_t n = ffn_reduce_token_cpu(gate.data(), up.data(), abs_out.data(),
                                            bits.data(), 8, 1e-6f);
    CHECK(ctx, n == 1);
    const float expect = std::fabs(silu(2.0f) * 3.0f);
    CHECK(ctx, std::fabs(abs_out[3] - expect) < 1e-5f);
    CHECK(ctx, abs_out[0] == 0.f);
    CHECK(ctx, bit_test(bits.data(), 3));
    CHECK(ctx, !bit_test(bits.data(), 2));
    CHECK(ctx, !ffn_reduce_cuda_available() || ffn_reduce_backend() == ReduceBackend::Cuda);
    CHECK(ctx, ffn_reduce_backend() == ReduceBackend::Cpu ||
                   ffn_reduce_backend() == ReduceBackend::Cuda);
}
