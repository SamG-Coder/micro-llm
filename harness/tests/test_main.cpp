#include "test_common.hpp"

#include <cstdio>

int main() {
    struct Row {
        const char* name;
        void (*fn)(TestContext&);
    } tests[] = {
        {"serialize_roundtrip", test_serialize_roundtrip},
        {"pack_id_0_47_mapping", test_pack_id_mapping},
        {"channel_alignment", test_channel_alignment},
        {"bitset_floor_or_after_logits", test_bitset_floor_or_after_logits},
        {"dead_vs_spike_pack", test_dead_vs_spike_pack},
        {"serve_ok_refuses_unless_present_and_true", test_serve_ok_refuses_unless_present_and_true},
        {"serve_stack_gate", test_serve_stack_gate},
        {"layer_hooked_trailer", test_layer_hooked_trailer},
        {"relative_spike_identity", test_relative_spike_identity},
        {"device_tap_api", test_device_tap_api},
        {"live_forward_backends", test_live_forward_backends},
        {"graph_hook_names", test_graph_hook_names},
        {"gguf_serve_gate", test_gguf_serve_gate},
    };

    int failed = 0;
    int passed = 0;
    const int n = static_cast<int>(sizeof(tests) / sizeof(tests[0]));
    for (int i = 0; i < n; ++i) {
        TestContext ctx;
        ctx.name = tests[i].name;
        std::printf("== %s\n", tests[i].name);
        tests[i].fn(ctx);
        std::printf("   %d passed, %d failed\n", ctx.passed, ctx.failed);
        passed += ctx.passed;
        failed += ctx.failed;
    }
    std::printf("total %d passed, %d failed, %d tests\n", passed, failed, n);
    return failed == 0 ? 0 : 1;
}
