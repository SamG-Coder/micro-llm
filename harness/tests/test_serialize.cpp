#include "test_common.hpp"

#include "micro_llm/micro_llm.hpp"

#include <cstdio>
#include <string>

void test_serialize_roundtrip(TestContext& ctx) {
    using namespace micro_llm;
    PruneTable a;
    a.fire_eps = 2.5e-5f;
    a.spike_eps = 3.5e-5f;
    a.n_tokens = 123;
    a.channel(0, 0).n_fired = 9;
    a.channel(0, 0).sumsq = 4.25f;
    a.channel(0, 0).maxabs = 1.5f;
    a.channel(63, 17407).n_fired = 2;
    a.channel(63, 17407).sumsq = 0.25f;
    a.channel(63, 17407).maxabs = 0.5f;
    a.channel(16, 100).n_fired = 1;
    a.channel(16, 100).sumsq = 8.0f;
    a.channel(16, 100).maxabs = 2.0f;
    a.pack(0).n_spike = 4;
    a.pack(0).sumsq_residual = 12.5;
    a.pack(47).n_spike = 1;
    a.pack(47).sumsq_residual = 0.01;
    a.set_floor_keep(3, 7, true);
    a.set_floor_keep(63, 17407, true);
    a.set_vocab_seen(0);
    a.set_vocab_seen(17);
    a.set_vocab_seen(248319);

    const std::string path = "test_prune_table.roundtrip.bin";
    std::string err;
    CHECK(ctx, save_prune_table(a, path, &err));
    if (!err.empty()) {
        std::fprintf(stderr, "  save err: %s\n", err.c_str());
    }

    PruneTable b;
    err.clear();
    CHECK(ctx, load_prune_table(b, path, &err));
    if (!err.empty()) {
        std::fprintf(stderr, "  load err: %s\n", err.c_str());
    }

    CHECK(ctx, a == b);
    CHECK(ctx, b.channel(0, 0).n_fired == 9);
    CHECK(ctx, b.channel(63, 17407).maxabs == 0.5f);
    CHECK(ctx, b.pack(0).n_spike == 4);
    CHECK(ctx, b.pack(47).pack == 47);
    CHECK(ctx, b.pack(47).layer == delta_layer_from_pack_id(47));
    CHECK(ctx, b.floor_keep(3, 7));
    CHECK(ctx, b.floor_keep(63, 17407));
    CHECK(ctx, !b.floor_keep(0, 1));
    CHECK(ctx, b.vocab_seen(0));
    CHECK(ctx, b.vocab_seen(17));
    CHECK(ctx, b.vocab_seen(248319));
    CHECK(ctx, !b.vocab_seen(18));
    CHECK(ctx, b.n_tokens == 123);
    CHECK(ctx, b.flags & kPruneTableFlagHasFloor);
    std::remove(path.c_str());
}
