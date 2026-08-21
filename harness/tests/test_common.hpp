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
