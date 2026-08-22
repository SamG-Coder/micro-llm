#pragma once

#include <cstdio>
#include <string>

struct TestContext {
    int failed = 0;
    int passed = 0;
    std::string name;
};

inline void test_check(TestContext& ctx, bool cond, const char* expr, const char* file,
                       int line) {
    if (cond) {
        ++ctx.passed;
    } else {
        ++ctx.failed;
        std::fprintf(stderr, "  FAIL %s:%d: %s\n", file, line, expr);
    }
}

#define CHECK(ctx, cond) test_check((ctx), static_cast<bool>(cond), #cond, __FILE__, __LINE__)

void test_serialize_roundtrip(TestContext& ctx);
void test_pack_id_mapping(TestContext& ctx);
void test_channel_alignment(TestContext& ctx);
void test_bitset_floor_or_after_logits(TestContext& ctx);
void test_dead_vs_spike_pack(TestContext& ctx);
void test_serve_ok_refuses_unless_present_and_true(TestContext& ctx);
void test_serve_stack_gate(TestContext& ctx);
void test_layer_hooked_trailer(TestContext& ctx);
void test_relative_spike_identity(TestContext& ctx);
void test_device_tap_api(TestContext& ctx);
void test_live_forward_backends(TestContext& ctx);
void test_graph_hook_names(TestContext& ctx);
void test_gguf_serve_gate(TestContext& ctx);
void test_hotspot_ui_files(TestContext& ctx);
void test_hook_ring_encode(TestContext& ctx);
void test_hour_cli_resolve(TestContext& ctx);
void test_ffn_stream_budget(TestContext& ctx);
void test_residency_plan(TestContext& ctx);
void test_async_htr1_ring(TestContext& ctx);
