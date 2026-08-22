#include "test_common.hpp"

#include "micro_llm/micro_llm.hpp"

#include <string>

void test_ffn_stream_budget(TestContext& ctx) {
    using namespace micro_llm;

    CHECK(ctx, kMaxResidentFfnLayers == 1);
    CHECK(ctx, kPinnedGaKvBytes == (69ull * kGiB) / 10ull);
    CHECK(ctx, kCudaScratchBytes == (9ull * kGiB) / 10ull);
    CHECK(ctx, kServeUsableBytes == (152ull * kGiB) / 10ull);
    CHECK(ctx, kHourCardSoftBytes == 14ull * kGiB);
    CHECK(ctx, streamed_hour_card_fits());
    CHECK(ctx, streamed_hour_card_bytes() < kServeUsableBytes);
    CHECK(ctx, streamed_hour_card_bytes() == kPinnedGaKvBytes + kCudaScratchBytes + kQ4FfnLayerBytes);
    CHECK(ctx, parked_all_ffn_exceeds_card());
    CHECK(ctx, parked_all_ffn_exceeds_card(kNLayers));
    CHECK(ctx, !parked_all_ffn_exceeds_card(1));
    const uint32_t n_park = ffn_park_layers_that_fit();
    CHECK(ctx, n_park > 0);
    CHECK(ctx, n_park < kNLayers);
    CHECK(ctx, n_park != 64);
    CHECK(ctx, hour_park_stream_fits(n_park));
    CHECK(ctx, hour_park_stream_card_bytes(n_park) <= kHourCardSoftBytes);
    CHECK(ctx, hour_park_stream_card_bytes(n_park) <= kServeUsableBytes);
    CHECK(ctx, !hour_park_stream_fits(kNLayers));

    CHECK(ctx, gguf_block_count_is_hybrid(64));
    CHECK(ctx, gguf_block_count_is_hybrid(65));
    CHECK(ctx, !gguf_block_count_is_hybrid(66));
    CHECK(ctx, hook_layer_count_from_blocks(65) == kNLayers);
    CHECK(ctx, hook_layer_count_from_blocks(64) == kNLayers);

    StreamerConfig scfg;
    scfg.ffn_scratch_bytes = 4096;
    TraceStreamer streamer(scfg);
    streamer.begin_session();
    CHECK(ctx, streamer.resident_ffn_layers() == 0);
    CHECK(ctx, streamer.peak_ffn_vram_bytes() == 4096);
    CHECK(ctx, streamer.peak_ffn_bytes() == 8192);
    CHECK(ctx, streamer.card_stack_fits());

    CHECK(ctx, streamer.prefetch_ffn(1));
    CHECK(ctx, streamer.resident_ffn_layers() == 0);  // prefetch is host
    CHECK(ctx, streamer.n_parked_ffn() == 0);
    CHECK(ctx, streamer.bind_ffn(0));
    CHECK(ctx, streamer.resident_ffn_layers() == 1);
    CHECK(ctx, streamer.compute_layer() == 0);
    CHECK(ctx, streamer.cuda_bind_count() == 1);
    CHECK(ctx, streamer.bind_ffn(1));
    CHECK(ctx, streamer.resident_ffn_layers() == 1);
    CHECK(ctx, streamer.compute_layer() == 1);
    CHECK(ctx, streamer.ffn_vram_bytes() == 4096);
    CHECK(ctx, streamer.evict_ffn(1));
    CHECK(ctx, streamer.resident_ffn_layers() == 0);
    CHECK(ctx, streamer.ffn_vram_bytes() == 0);

    StreamerConfig parked_cfg;
    parked_cfg.ffn_scratch_bytes = 4096;
    parked_cfg.n_parked_ffn = 8;
    TraceStreamer parked(parked_cfg);
    parked.begin_session();
    CHECK(ctx, parked.ffn_is_parked(7));
    CHECK(ctx, !parked.ffn_is_parked(8));
    CHECK(ctx, parked.bind_ffn(3));
    CHECK(ctx, parked.resident_ffn_layers() == 0);  // parked, not the stream slot
    CHECK(ctx, parked.bind_ffn(9));
    CHECK(ctx, parked.resident_ffn_layers() == 1);
    CHECK(ctx, parked.cuda_bind_count() == 2);
    parked.end_session();
    // MTP extra block is outside the 64-layer ring.
    CHECK(ctx, !streamer.bind_ffn(64));
    streamer.end_session();

    int layer = -2;
    CHECK(ctx, classify_graph_tensor("ffn_gate-64", &layer) == GraphHookSite::None);
    CHECK(ctx, classify_graph_tensor("ffn_up-64", &layer) == GraphHookSite::None);
    CHECK(ctx, classify_graph_tensor("ffn_gate-63", &layer) == GraphHookSite::FfnGate);
    CHECK(ctx, layer == 63);

    LiveForwardConfig cfg;
    CHECK(ctx, cfg.n_gpu_layers == 0);
    CHECK(ctx, cfg.n_gpu_layers != 16);
    CHECK(ctx, cfg.n_gpu_layers != 99);
    CHECK(ctx, !cfg.load_mtp);
    CHECK(ctx, !cfg.disable_flash_attn);  // FA on when FFN+DeltaNet are CUDA
    CHECK(ctx, cfg.disable_op_offload);
    CHECK(ctx, !cfg.pack_checkpoint);
}
