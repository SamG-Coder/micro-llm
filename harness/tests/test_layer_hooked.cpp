#include "test_common.hpp"

#include "micro_llm/micro_llm.hpp"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

void test_layer_hooked_trailer(TestContext& ctx) {
    using namespace micro_llm;

    CHECK(ctx, kPruneTableFlagLayerHooked == (1u << 1));

    TraceHooks hooks;
    std::vector<float> abs_act(kFfnIntermediate, 0.f);
    abs_act[0] = 1.0f;

    hooks.begin_token(0);
    CHECK(ctx, hooks.on_ffn_abs(3, abs_act.data()));
    hooks.after_logits(0, false);

    CHECK(ctx, hooks.table().n_tokens == 1);
    CHECK(ctx, hooks.table().layer_was_hooked(3));
    CHECK(ctx, !hooks.table().layer_is_unwired(3));
    CHECK(ctx, !hooks.table().layer_is_dead(3));

    // Tokens ran, other layers never tapped — unwired, not dead. Do not fake n_fired.
    CHECK(ctx, hooks.table().layer_is_unwired(0));
    CHECK(ctx, !hooks.table().layer_was_hooked(0));
    CHECK(ctx, hooks.table().channel(0, 0).n_fired == 0);
    CHECK(ctx, !hooks.table().layer_is_dead(0));
    CHECK(ctx, hooks.table().layer_keep_full_width(0));
    CHECK(ctx, hooks.table().count_missing_hooks() == 63);  // all but layer 3

    // Hooked + all n_fired==0 => dead.
    hooks.begin_token(1);
    std::vector<float> zeros(kFfnIntermediate, 0.f);
    CHECK(ctx, hooks.on_ffn_abs(7, zeros.data()));
    hooks.after_logits(1, false);
    CHECK(ctx, hooks.table().layer_was_hooked(7));
    CHECK(ctx, hooks.table().layer_is_dead(7));
    CHECK(ctx, !hooks.table().layer_is_unwired(7));

    const std::string path = "test_layer_hooked.bin";
    std::string err;
    CHECK(ctx, save_prune_table(hooks.table(), path, &err));

    std::ifstream is(path, std::ios::binary | std::ios::ate);
    const auto sz = static_cast<size_t>(is.tellg());
    is.close();
    const size_t base = 80ull + 64ull * 17408ull * 16ull + 48ull * 24ull + 139264ull + 31040ull;
    CHECK(ctx, base == 17997328ull);
    CHECK(ctx, sz == base + 8);
    CHECK(ctx, sz < 20ull * 1024ull * 1024ull);  // ~18MB scores; do not pack a GGUF
    {
        std::ifstream magis(path, std::ios::binary);
        char mag[4] = {};
        magis.read(mag, 4);
        CHECK(ctx, mag[0] == 'M' && mag[1] == 'L' && mag[2] == 'P' && mag[3] == 'T');
        CHECK(ctx, !(mag[0] == 'G' && mag[1] == 'G' && mag[2] == 'U' && mag[3] == 'F'));
    }

    PruneTable loaded;
    CHECK(ctx, load_prune_table(loaded, path, &err));
    CHECK(ctx, loaded.flags & kPruneTableFlagLayerHooked);
    CHECK(ctx, loaded.layer_was_hooked(3));
    CHECK(ctx, loaded.layer_is_unwired(0));
    CHECK(ctx, loaded.layer_is_dead(7));
    CHECK(ctx, loaded.count_missing_hooks() == 62);  // 64 minus hooked 3 and 7
    CHECK(ctx, loaded.layer_keep_full_width(0));
    CHECK(ctx, !loaded.layer_keep_full_width(3));
    std::remove(path.c_str());
}
