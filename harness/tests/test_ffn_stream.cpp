#include "test_common.hpp"

#include "micro_llm/micro_llm.hpp"

#include <string>

void test_ffn_stream_budget(TestContext& ctx) {
    using namespace micro_llm;

    CHECK(ctx, kNStreamSlots == 2);
    CHECK(ctx, kMinStreamedFfnLayers == 7);
    CHECK(ctx, kTargetMaxStreamedFfnLayers == 12);
    CHECK(ctx, kMaxParkedFfnLayers == 57);
    CHECK(ctx, kHourKvReserveTokens == 20000);
    CHECK(ctx, kHourKvReserveBytes == 20000ull * 65536ull);
    CHECK(ctx, kHourCardSoftBytes == 14ull * kGiB);
    CHECK(ctx, kPinnedGaWeightBytes == (6ull * kGiB) / 10ull);
    CHECK(ctx, hour_fixed_card_bytes() > kHourKvReserveBytes);
    CHECK(ctx, hour_fixed_card_bytes() < kHourCardSoftBytes);

    const uint32_t n_park = ffn_park_layers_that_fit();
    const uint32_t n_stream = ffn_stream_layers(n_park);
    CHECK(ctx, n_park > 0);
    CHECK(ctx, n_park < kNLayers);
    CHECK(ctx, n_park <= kMaxParkedFfnLayers);
    CHECK(ctx, n_park != 64);
    CHECK(ctx, n_stream >= kMinStreamedFfnLayers);
    CHECK(ctx, n_stream <= kTargetMaxStreamedFfnLayers);
    CHECK(ctx, streamed_hour_in_20_tok_band(n_stream));
    CHECK(ctx, hour_park_stream_fits(n_park));
    CHECK(ctx, hour_park_stream_card_bytes(n_park) <= kHourCardSoftBytes);
    CHECK(ctx, hour_park_stream_card_bytes(n_park) <= kServeUsableBytes);
    CHECK(ctx, !hour_park_stream_fits(kNLayers));
    CHECK(ctx, parked_all_ffn_exceeds_card());

    // 7780: 64 CPU FFN * 4.5ms = 3.47 tok/s. CUDA0 6760 MiB did not help.
    const double cpu64 = cpu_ffn_tok_per_sec();
    CHECK(ctx, cpu64 > 3.3 && cpu64 < 3.6);
    CHECK(ctx, kMeasured7780Cuda0MiB == 6760ull);
    CHECK(ctx, kMeasured7780TokPerSec192 > 3.4f && kMeasured7780TokPerSec192 < 3.7f);

    // 7-12 streamed on CUDA / PCIe 5 is the 20+ path. Host leftover is not.
    CHECK(ctx, stream_pcie_tok_per_sec(12) >= 20.0);
    CHECK(ctx, stream_pcie_tok_per_sec(7) >= 30.0);
    CHECK(ctx, cpu_ffn_tok_per_sec(4.5, 12) < 20.0);
    CHECK(ctx, cpu_ffn_tok_per_sec(4.5, 64) < 5.0);

    CHECK(ctx, gguf_block_count_is_hybrid(64));
    CHECK(ctx, gguf_block_count_is_hybrid(65));
    CHECK(ctx, !gguf_block_count_is_hybrid(66));
    CHECK(ctx, hook_layer_count_from_blocks(65) == kNLayers);
    CHECK(ctx, hook_layer_count_from_blocks(64) == kNLayers);

    CHECK(ctx, clamp_hybrid_n_gpu_layers(16) == 0);
    CHECK(ctx, clamp_hybrid_n_gpu_layers(99) == 0);
    CHECK(ctx, clamp_hybrid_n_gpu_layers(0) == 0);
    CHECK(ctx, hybrid_n_gpu_layers() == 0);
    CHECK(ctx, hybrid_n_gpu_layers() != 16);
    CHECK(ctx, hybrid_n_gpu_layers() != 99);

    const auto gpu = hybrid_gpu_tensor_regexes(n_park);
    bool has_parked_ffn = false;
    bool has_ga = false;
    bool has_embed = false;
    for (const auto& p : gpu) {
        if (p.find("ffn_gate") != std::string::npos) has_parked_ffn = true;
        if (p.find("attn_output") != std::string::npos) has_ga = true;
        if (p.find("token_embd") != std::string::npos) has_embed = true;
    }
    CHECK(ctx, has_parked_ffn && has_ga && has_embed);

    StreamerConfig scfg;
    scfg.ffn_scratch_bytes = 4096;
    TraceStreamer streamer(scfg);
    streamer.begin_session();
    CHECK(ctx, streamer.resident_ffn_layers() == 0);
    CHECK(ctx, streamer.peak_ffn_vram_bytes() == 8192);  // two slots
    CHECK(ctx, streamer.peak_ffn_bytes() == 8192);
    CHECK(ctx, streamer.card_stack_fits());

    CHECK(ctx, streamer.prefetch_ffn(1));
    CHECK(ctx, streamer.bind_ffn(0));
    CHECK(ctx, streamer.compute_layer() == 0);
    CHECK(ctx, streamer.last_bind_was_cuda());
    CHECK(ctx, streamer.cuda_bind_count() >= 1);
    CHECK(ctx, streamer.prefetch_outstanding());  // n+1 overlapped
    CHECK(ctx, streamer.resident_stream_slots() >= 1);
    CHECK(ctx, streamer.evict_ffn(0));
    streamer.end_session();

    StreamerConfig parked_cfg;
    parked_cfg.ffn_scratch_bytes = 4096;
    parked_cfg.n_parked_ffn = 8;
    TraceStreamer parked(parked_cfg);
    parked.begin_session();
    CHECK(ctx, parked.ffn_is_parked(7));
    CHECK(ctx, !parked.ffn_is_parked(8));
    CHECK(ctx, parked.bind_ffn(3));
    CHECK(ctx, parked.resident_ffn_layers() == 0);  // parked, not a stream slot
    CHECK(ctx, parked.last_bind_kind() == FfnComputeKind::ParkedCuda);
    CHECK(ctx, parked.bind_ffn(9));
    CHECK(ctx, parked.last_bind_kind() == FfnComputeKind::StreamCuda);
    CHECK(ctx, parked.resident_ffn_layers() >= 1);
    CHECK(ctx, parked.overlap_prefetch_count() >= 1 || parked.prefetch_outstanding());
    parked.end_session();

    CHECK(ctx, !streamer.bind_ffn(64));

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
    CHECK(ctx, cfg.disable_flash_attn);
    CHECK(ctx, !cfg.disable_op_offload);
    CHECK(ctx, !cfg.pack_checkpoint);

    const std::string park_line = format_ffn_cuda_park_line(n_park, n_park * kQ4FfnLayerBytes);
    CHECK(ctx, park_line.find("ngl=0") != std::string::npos);
    CHECK(ctx, park_line.find("never_64=1") != std::string::npos);
    CHECK(ctx, format_ffn_cuda_bind_line(9, false).find("stream=1") != std::string::npos);
}
