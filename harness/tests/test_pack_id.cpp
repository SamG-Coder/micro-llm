#include "test_common.hpp"

#include "micro_llm/types.hpp"

#include <vector>

void test_pack_id_mapping(TestContext& ctx) {
    using namespace micro_llm;

    CHECK(ctx, kNDeltaNetPacks == 48);
    CHECK(ctx, pack_id_from_group_slot(0, 0) == 0);
    CHECK(ctx, pack_id_from_group_slot(0, 1) == 1);
    CHECK(ctx, pack_id_from_group_slot(0, 2) == 2);
    // Not 0..2 inside the next group:
    CHECK(ctx, pack_id_from_group_slot(1, 0) == 3);
    CHECK(ctx, pack_id_from_group_slot(1, 1) == 4);
    CHECK(ctx, pack_id_from_group_slot(1, 2) == 5);
    CHECK(ctx, pack_id_from_group_slot(15, 2) == 47);

    CHECK(ctx, layer_from_group_slot(0, 0) == 0);
    CHECK(ctx, layer_from_group_slot(0, 2) == 2);
    CHECK(ctx, layer_from_group_slot(1, 0) == 4);
    CHECK(ctx, layer_from_group_slot(15, 2) == 62);
    CHECK(ctx, layer_from_group_slot(3, 1) == 4 * 3 + 1);

    std::vector<int> seen(kNDeltaNetPacks, 0);
    uint32_t n_ga = 0;
    uint32_t n_dn = 0;
    for (uint32_t layer = 0; layer < kNLayers; ++layer) {
        if (is_gated_attention_layer(layer)) {
            ++n_ga;
            CHECK(ctx, (layer % 4) == 3);
            CHECK(ctx, !is_delta_net_layer(layer));
        } else {
            ++n_dn;
            const uint32_t p = pack_id_from_delta_layer(layer);
            CHECK(ctx, p < 48);
            CHECK(ctx, seen[p] == 0);
            seen[p] = 1;
            CHECK(ctx, delta_layer_from_pack_id(p) == layer);
            CHECK(ctx, layer_from_group_slot(group_from_pack_id(p), slot_from_pack_id(p)) ==
                           layer);
        }
    }
    CHECK(ctx, n_ga == 16);
    CHECK(ctx, n_dn == 48);
    for (uint32_t p = 0; p < 48; ++p) {
        CHECK(ctx, seen[p] == 1);
        CHECK(ctx, is_delta_net_layer(delta_layer_from_pack_id(p)));
        CHECK(ctx, !is_gated_attention_layer(delta_layer_from_pack_id(p)));
    }

    CHECK(ctx, is_layer_drop_floor(0));
    CHECK(ctx, is_layer_drop_floor(1));
    CHECK(ctx, is_layer_drop_floor(62));
    CHECK(ctx, is_layer_drop_floor(63));
    CHECK(ctx, !is_layer_drop_floor(2));
    CHECK(ctx, !is_layer_drop_floor(61));
    CHECK(ctx, ffn_is_scored(0) && ffn_is_scored(3) && ffn_is_scored(63));
    CHECK(ctx, !gated_attention_block_may_drop());
}
