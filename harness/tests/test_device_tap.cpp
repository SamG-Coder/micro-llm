#include "test_common.hpp"

#include "micro_llm/micro_llm.hpp"

#include <vector>

void test_device_tap_api(TestContext& ctx) {
    using namespace micro_llm;

    TraceHooks hooks;
    std::vector<float> gate(8, 1.f);
    std::vector<float> up(8, 1.f);

    // Host path still works and marks the layer hooked.
    CHECK(ctx, hooks.on_ffn_activations(4, gate.data(), up.data(), 8));
    CHECK(ctx, hooks.table().layer_was_hooked(4));

    // Device tap: without CUDA this must return false, not fake n_fired.
    const uint64_t fired_before = hooks.table().channel(5, 0).n_fired;
    const bool tapped = hooks.on_ffn_activations_device(5, gate.data(), up.data(), 8);
    if (!ffn_reduce_cuda_available()) {
        CHECK(ctx, !tapped);
        CHECK(ctx, hooks.table().channel(5, 0).n_fired == fired_before);
        CHECK(ctx, !hooks.table().layer_was_hooked(5));
    }

    // Alloc offset is not a device VA. Must not launch (5080 first-decode AV).
    const float* offset = reinterpret_cast<const float*>(static_cast<uintptr_t>(0x4000));
    CHECK(ctx, ptr_looks_like_integer_offset(offset));
    CHECK(ctx, !hooks.on_ffn_activations_device(6, offset, offset, 8));
    CHECK(ctx, !hooks.table().layer_was_hooked(6));

    GraphTensorView bad;
    bad.name = "ffn_gate-1";
    bad.data = offset;
    bad.on_device = true;
    bad.ptr_ok = false;
    bad.ne0 = kFfnIntermediate;
    StreamerConfig scfg;
    scfg.ffn_scratch_bytes = 4096;
    TraceStreamer streamer(scfg);
    streamer.begin_session();
    GraphHookSession sess(hooks, streamer);
    sess.begin_token(0);
    CHECK(ctx, !sess.on_tensor(bad, false));
    streamer.end_session();

    CudaReduceContext& ctxu = persistent_cuda_reduce();
    if (!ffn_reduce_cuda_available()) {
        CHECK(ctx, !ctxu.ensure(kFfnIntermediate));
        CHECK(ctx, ffn_reduce_token_cuda_device(gate.data(), up.data(), gate.data(),
                                                nullptr, 8, 1e-6f) == -1);
    }
}
