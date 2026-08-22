#include "test_common.hpp"

#include "micro_llm/micro_llm.hpp"

#include <cstring>
#include <regex>
#include <string>
#include <vector>

void test_ffn_stream_budget(TestContext& ctx) {
    using namespace micro_llm;

    CHECK(ctx, kNStreamSlots == 2);
    CHECK(ctx, kMinStreamedFfnLayers == 7);
    CHECK(ctx, kTargetMaxStreamedFfnLayers == 12);
    CHECK(ctx, kMaxParkedFfnLayers == kMeasured5080Park57);
    CHECK(ctx, kMaxParkedFfnLayers == 57);
    CHECK(ctx, kMaxParkedFfnLayers < kNLayers);
    CHECK(ctx, kHourKvReserveTokens == 20000);
    CHECK(ctx, kHourKvReserveBytes == 20000ull * 65536ull);
    CHECK(ctx, kHourCardSoftBytes == 14ull * kGiB);
    CHECK(ctx, kPinnedGaWeightBytes == (6ull * kGiB) / 10ull);
    CHECK(ctx, hour_fixed_card_bytes() > kHourKvReserveBytes);
    CHECK(ctx, hour_fixed_card_bytes() < kHourCardSoftBytes);

    CHECK(ctx, !ggml_can_rebind_q4_midgraph());
    CHECK(ctx, ggml_can_bind_q4_at_load());
    CHECK(ctx, ggml_slot_pack_ok(0, kStreamSlotBytes, kStreamSlotBytes));
    CHECK(ctx, !ggml_slot_pack_ok(1, kStreamSlotBytes, kStreamSlotBytes));
    CHECK(ctx, !ggml_slot_pack_ok(0, kStreamSlotBytes + 1, kStreamSlotBytes));
    CHECK(ctx, !ggml_slot_pack_ok(0, 0, kStreamSlotBytes));
    CHECK(ctx, ggml_stream_slot_kind(56, 57) == kStreamSlotParked);
    CHECK(ctx, ggml_stream_slot_kind(63, 57) == kStreamSlotA);
    CHECK(ctx, ggml_stream_slot_kind(62, 57) == kStreamSlotB);
    CHECK(ctx, ggml_stream_slot_kind(57, 57) == kStreamSlotCpu);
    CHECK(ctx, ggml_stream_slot_kind(59, 57) == kStreamSlotCpu);
    CHECK(ctx, ggml_stream_slot_kind(59, 57) != 2);  // extra park is illegal
    const std::string tbind =
        format_ggml_tensor_bind_line(57, kQ4FfnLayerBytesMeasured5080, true, true);
    CHECK(ctx, tbind.find("ggml_tensor_bind layer=57") == 0);
    CHECK(ctx, tbind.find("real_h2d=1") != std::string::npos);
    CHECK(ctx, tbind.find("ggml_used=1") != std::string::npos);
    CHECK(ctx, tbind.find("layer_MiB=160.0") != std::string::npos);
    CHECK(ctx, tbind.find("private_cudaMalloc=0") != std::string::npos);
    CHECK(ctx, format_ggml_tensor_bind_line(59, 0, false, false).find("ggml_used=0") !=
                   std::string::npos);
    CHECK(ctx, ffn_park_all_fits_5080_measured());
    CHECK(ctx, kQ4FfnLayerBytesMeasured5080 == 160ull * 1024ull * 1024ull);
    CHECK(ctx, kStreamSlotBytes == 160ull * 1024ull * 1024ull);
    CHECK(ctx, kStreamSlotPairBudgetBytes == 340ull * 1024ull * 1024ull);
    CHECK(ctx, kStreamWorkspaceBytes == kStreamSlotPairBudgetBytes);
    CHECK(ctx, kMeasured5080Cuda0MiB == 9851ull);
    CHECK(ctx, kMeasured5080FreeMiB == 2409ull);
    CHECK(ctx, kMeasured5080GraphSplits == 340);
    CHECK(ctx, kMeasured5080Park57 == 57);
    CHECK(ctx, kMeasured5080Stream7 == 7);

    const VramLedger led = vram_ledger_slots_first();
    CHECK(ctx, led.slot_a_bytes == kStreamSlotBytes);
    CHECK(ctx, led.slot_b_bytes == kStreamSlotBytes);
    CHECK(ctx, led.slot_a_bytes + led.slot_b_bytes + 20ull * 1024ull * 1024ull ==
                   kStreamSlotPairBudgetBytes);
    CHECK(ctx, led.kv20k_bytes == kHourKvReserveBytes);
    CHECK(ctx, led.scratch_bytes == kCudaScratchBytes);
    CHECK(ctx, led.graph_reserve_bytes == kHourGraphReserveBytes);
    CHECK(ctx, led.n_parked_ffn == kMeasured5080Park57);
    CHECK(ctx, led.n_streamed_ffn == kMeasured5080Stream7);
    CHECK(ctx, hour_never_park_64(led.n_parked_ffn));
    CHECK(ctx, vram_ledger_free_to_hard(led) > 0);
    const std::string vline = format_vram_ledger(led);
    CHECK(ctx, vline.find("VRAM_LEDGER") == 0);
    CHECK(ctx, vline.find("slot_A_MiB=160.0") != std::string::npos);
    CHECK(ctx, vline.find("slot_B_MiB=160.0") != std::string::npos);
    CHECK(ctx, vline.find("graph_reserve_MiB=") != std::string::npos);
    CHECK(ctx, vline.find("extra_park=0") != std::string::npos);
    CHECK(ctx, vline.find("kv20k_MiB=") != std::string::npos);
    CHECK(ctx, vline.find("free_to_15_2_MiB=") != std::string::npos);

    const SplitLedger sl = split_ledger_trace_off_cuda_ffn();
    CHECK(ctx, sl.trace_off);
    CHECK(ctx, sl.callback_hooks == 0);
    const std::string sline = format_split_ledger(sl);
    CHECK(ctx, sline.find("callback/hooks=0") != std::string::npos);
    CHECK(ctx, sline.find("CUDA_Host_to_CUDA=0") != std::string::npos);
    CHECK(ctx, sline.find("placement/buffer=") != std::string::npos);
    CHECK(ctx, classify_split_cause("blk.63.ffn_gate.weight", "CPU", true) ==
                   SplitCauseKind::Placement);
    CHECK(ctx, classify_split_cause("blk.63.ffn_down.weight", "CPU_Mapped", true) ==
                   SplitCauseKind::Placement);
    CHECK(ctx, classify_split_cause("blk.63.ffn_down.weight", "CPU_Mapped", true) !=
                   SplitCauseKind::UnsupportedOp);
    CHECK(ctx, classify_split_cause("blk.63.ffn_down.weight", "CPU_Mapped", true) !=
                   SplitCauseKind::CudaHostToCuda);
    CHECK(ctx, classify_split_cause("blk.63.ffn_up.weight", "CUDA_Host", true) ==
                   SplitCauseKind::CudaHostToCuda);
    CHECK(ctx, buft_is_cpu_mapped("CPU_Mapped"));
    CHECK(ctx, !buft_is_cpu_mapped("CUDA0"));
    CHECK(ctx, name_is_slot_q4_weight("blk.63.ffn_down.weight"));
    CHECK(ctx, name_is_slot_q4_weight("blk.63.ffn_out.weight"));
    CHECK(ctx, !name_is_slot_q4_weight("blk.63.ffn_down.scale"));
    CHECK(ctx, !name_is_slot_q4_weight("blk.63.ffn_norm.weight"));
    CHECK(ctx, classify_split_cause("output.weight", "CPU", true) ==
                   SplitCauseKind::BackendTransition);
    CHECK(ctx, classify_split_cause("cb_eval", "set", false) == SplitCauseKind::HookCallback);
    CHECK(ctx, name_is_ffn_mul_mat_src("blk.63.ffn_gate.weight"));
    const std::string causes =
        format_split_causes_block(sl, "blk.63.ffn_gate.weight", "CPU", 63);
    CHECK(ctx, causes.find("SPLIT_CAUSE kind=hook/callback") != std::string::npos);
    CHECK(ctx, causes.find("SPLIT_CAUSE kind=placement/buffer") != std::string::npos);
    CHECK(ctx, causes.find("SPLIT_CAUSE kind=CUDA_Host_to_CUDA") != std::string::npos);
    CHECK(ctx, causes.find("SPLIT_CAUSE kind=backend_transition") != std::string::npos);
    CHECK(ctx, causes.find("SPLIT_CAUSE kind=unsupported_op") != std::string::npos);
    CHECK(ctx, causes.find("SPLIT_CAUSE kind=other") != std::string::npos);
    const std::string why =
        format_split_why_line(kMeasured5080ReserveSplits, "blk.63.ffn_down.weight",
                             "CPU_Mapped");
    CHECK(ctx, why.find("SPLIT_WHY n=638") == 0);
    CHECK(ctx, why.find("last=blk.63.ffn_down.weight") != std::string::npos);
    CHECK(ctx, why.find("last_buft=CPU_Mapped") != std::string::npos);
    CHECK(ctx, kMeasured5080ReserveSplits532 == 532);
    CHECK(ctx, kMeasured5080Cuda0BindMiB == 13110ull);
    SplitLedger hooked = sl;
    hooked.trace_off = true;
    hooked.callback_hooks = 99;
    CHECK(ctx, format_split_ledger(hooked).find("callback/hooks=0") != std::string::npos);

    const FfnGemmCounts gemm = ffn_gemm_all_cuda();
    CHECK(ctx, gemm.cuda == 192);
    CHECK(ctx, gemm.cpu == 0);
    CHECK(ctx, format_ffn_gemm_line(gemm).find("cuda=192") != std::string::npos);

    const uint32_t n_park = led.n_parked_ffn;
    const uint32_t n_stream = led.n_streamed_ffn;
    CHECK(ctx, n_park == kMeasured5080Park57);
    CHECK(ctx, n_park == 57);
    CHECK(ctx, n_stream == 7);
    CHECK(ctx, n_park <= kMaxParkedFfnLayers);
    CHECK(ctx, streamed_hour_in_20_tok_band(n_stream));
    CHECK(ctx, !streamed_hour_in_20_tok_band(0));
    CHECK(ctx, hour_park_stream_fits(n_park));
    CHECK(ctx, !hour_park_stream_fits(kNLayers));
    CHECK(ctx, hour_park_stream_card_bytes(n_park) <= kHourCardSoftBytes);
    CHECK(ctx, hour_park_stream_card_bytes(n_park) <= kServeUsableBytes);

    // 7 leftover * 160 MiB = 1120 MiB. Those seven stream through 160 MiB A/B.
    CHECK(ctx, stream_pcie_bytes_per_tok(7) == 7ull * kQ4FfnLayerBytesMeasured5080);
    CHECK(ctx, stream_pcie_bytes_per_tok(0) == 0);
    CHECK(ctx, stream_pcie_tok_per_sec(7) >= 20.0);
    CHECK(ctx, stream_pcie_tok_per_sec(0) == 0.0);

    // 7780: 64 CPU FFN * 4.5ms = 3.47 tok/s. CUDA0 6760 MiB did not help.
    const double cpu64 = cpu_ffn_tok_per_sec();
    CHECK(ctx, cpu64 > 3.3 && cpu64 < 3.6);
    CHECK(ctx, kMeasured7780Cuda0MiB == 6760ull);
    CHECK(ctx, kMeasured7780TokPerSec192 > 3.4f && kMeasured7780TokPerSec192 < 3.7f);
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
    bool has_ssm = false;
    bool catch_all_ffn = false;
    for (const auto& p : gpu) {
        if (p.find("ffn_gate") != std::string::npos) has_parked_ffn = true;
        if (p.find("attn_output") != std::string::npos) has_ga = true;
        if (p.find("token_embd") != std::string::npos) has_embed = true;
        if (p.find("ssm_") != std::string::npos) has_ssm = true;
        if (p.find("[0-9]+") != std::string::npos &&
            (p.find("ffn_gate") != std::string::npos || p.find("ffn_up") != std::string::npos ||
             p.find("ffn_down") != std::string::npos)) {
            catch_all_ffn = true;
        }
    }
    CHECK(ctx, has_parked_ffn && has_ga && has_embed && has_ssm);
    CHECK(ctx, !catch_all_ffn);
    bool hit56 = false;
    bool hit57 = false;
    bool hit59 = false;
    bool hit61 = false;
    bool hit62 = false;
    bool hit63 = false;
    bool hit63_scale = false;
    for (const auto& p : gpu) {
        if (p.find("ffn_") == std::string::npos) {
            continue;
        }
        const std::regex re(p);
        if (std::regex_search(std::string("blk.56.ffn_gate.weight"), re)) hit56 = true;
        if (std::regex_search(std::string("blk.57.ffn_gate.weight"), re)) hit57 = true;
        if (std::regex_search(std::string("blk.59.ffn_up.weight"), re)) hit59 = true;
        if (std::regex_search(std::string("blk.61.ffn_down.weight"), re)) hit61 = true;
        if (std::regex_search(std::string("blk.62.ffn_gate.weight"), re)) hit62 = true;
        if (std::regex_search(std::string("blk.63.ffn_down.weight"), re)) hit63 = true;
        if (std::regex_search(std::string("blk.63.ffn_gate.scale"), re)) hit63_scale = true;
    }
    CHECK(ctx, hit56);
    CHECK(ctx, !hit57);
    CHECK(ctx, !hit59);
    CHECK(ctx, !hit61);
    CHECK(ctx, !hit62);
    CHECK(ctx, !hit63);
    CHECK(ctx, !hit63_scale);
    bool hit_post63 = false;
    for (const auto& p : gpu) {
        if (p.find("post_norm") == std::string::npos &&
            p.find("post_attention_norm") == std::string::npos) {
            continue;
        }
        const std::regex re(p);
        if (std::regex_search(std::string("blk.63.attn_post_norm.weight"), re) ||
            std::regex_search(std::string("blk.63.post_attention_norm.weight"), re)) {
            hit_post63 = true;
        }
    }
    CHECK(ctx, hit_post63);
    CHECK(ctx, classify_backend_buft_name("CUDA_Host") == BuftKind::CudaHost);
    CHECK(ctx, classify_backend_buft_name("CUDA0") == BuftKind::Cuda);
    CHECK(ctx, classify_backend_buft_name("CPU") == BuftKind::Cpu);
    CHECK(ctx, classify_backend_buft_name("CPU_Mapped") == BuftKind::Cpu);
    const std::string place = format_ffn_place_line(171, 21, 0, 0);
    CHECK(ctx, place.find("streamed_cuda_host=0") != std::string::npos);

    const auto cpu = hybrid_cpu_tensor_regexes();
    bool cpu_ffn = false;
    bool cpu_ssm = false;
    bool cpu_head = false;
    for (const auto& p : cpu) {
        if (p.find("ffn_") != std::string::npos) cpu_ffn = true;
        if (p.find("ssm_") != std::string::npos) cpu_ssm = true;
        if (p.find("output") != std::string::npos || p.find("nextn") != std::string::npos) {
            cpu_head = true;
        }
    }
    CHECK(ctx, !cpu_ffn && !cpu_ssm && cpu_head);

    StreamerConfig scfg;
    scfg.ffn_scratch_bytes = 4096;
    TraceStreamer streamer(scfg);
    streamer.set_n_parked_ffn(kNLayers);
    CHECK(ctx, streamer.n_parked_ffn() == kMaxParkedFfnLayers);
    CHECK(ctx, streamer.n_streamed_ffn() == kMinStreamedFfnLayers);
    CHECK(ctx, streamer.ensure_slots());
    CHECK(ctx, streamer.slot_allocated());
    CHECK(ctx, !streamer.slot_bound(0));
    CHECK(ctx, !streamer.slot_bound(1));
    CHECK(ctx, streamer.slot_bind_bytes() == 0);
    const uint8_t q4[64] = {1, 2, 3, 4};
    CHECK(ctx, streamer.bind_q4_into_slot(0, q4, sizeof(q4)));
    CHECK(ctx, streamer.slot_bound(0));
    CHECK(ctx, !streamer.slot_bound(1));
    CHECK(ctx, streamer.slot_bind_bytes() == sizeof(q4));
    CHECK(ctx, format_ffn_slot_bind_line(0, sizeof(q4), true).find("real_h2d=1") !=
                   std::string::npos);
    CHECK(ctx, format_ffn_slot_bind_line(0, kStreamSlotBytes, true).find("layer_MiB=160.0") !=
                   std::string::npos);
    CHECK(ctx, streamer.slot_real_h2d(0));
    StreamerConfig tiny;
    tiny.ffn_scratch_bytes = 80;
    TraceStreamer too_small(tiny);
    uint8_t layer160[160];
    std::memset(layer160, 1, sizeof(layer160));
    CHECK(ctx, !too_small.bind_q4_into_slot(0, layer160, sizeof(layer160)));
    CHECK(ctx, !too_small.slot_real_h2d(0));
    StreamerConfig pack;
    pack.ffn_scratch_bytes = 160;
    pack.n_parked_ffn = 57;
    TraceStreamer packed(pack);
    uint8_t gate[50], up[50], down[60];
    std::memset(gate, 2, sizeof(gate));
    std::memset(up, 3, sizeof(up));
    std::memset(down, 4, sizeof(down));
    packed.set_stream_host_part(57, 0, gate, sizeof(gate));
    packed.set_stream_host_part(57, 1, up, sizeof(up));
    packed.set_stream_host_part(57, 2, down, sizeof(down));
    packed.set_stream_host_part(58, 0, gate, sizeof(gate));
    packed.set_stream_host_part(58, 1, up, sizeof(up));
    packed.set_stream_host_part(58, 2, down, sizeof(down));
    CHECK(ctx, packed.bind_layer_into_slot(0, 57));
    CHECK(ctx, packed.slot_real_h2d(0));
    CHECK(ctx, packed.bind_layer_into_slot(1, 58));
    CHECK(ctx, packed.slot_real_h2d(1));
    streamer.begin_session();
    CHECK(ctx, streamer.n_parked_ffn() == kMaxParkedFfnLayers);
    CHECK(ctx, streamer.n_streamed_ffn() == 7);
    CHECK(ctx, streamer.bind_ffn(0));
    CHECK(ctx, streamer.last_bind_kind() == FfnComputeKind::ParkedCuda);
    CHECK(ctx, streamer.host_bind_count() == 0);
    CHECK(ctx, streamer.resident_ffn_layers() == 0);
    streamer.set_stream_host(57, q4, sizeof(q4));
    streamer.set_stream_host(58, q4, sizeof(q4));
    CHECK(ctx, streamer.h2d_overflow_q4() > 0);
    CHECK(ctx, streamer.slot_bound(0) || streamer.slot_bound(1));
    CHECK(ctx, streamer.host_bind_count() == 0);
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
    CHECK(ctx, parked.host_bind_count() == 0);
    CHECK(ctx, parked.resident_ffn_layers() >= 1);
    CHECK(ctx, parked.overlap_prefetch_count() >= 1 || parked.prefetch_outstanding());
    parked.end_session();

    CHECK(ctx, !streamer.bind_ffn(64));

    int layer = -2;
    CHECK(ctx, classify_graph_tensor("ffn_gate-64", &layer) == GraphHookSite::None);
    CHECK(ctx, classify_graph_tensor("ffn_up-64", &layer) == GraphHookSite::None);
    CHECK(ctx, classify_graph_tensor("ffn_gate-63", &layer) == GraphHookSite::FfnGate);
    CHECK(ctx, layer == 63);
    CHECK(ctx, classify_graph_tensor("ffn_gate.weight", &layer) == GraphHookSite::None);
    CHECK(ctx, classify_graph_tensor("ffn_up.weight", &layer) == GraphHookSite::None);

    // Buffer-base + offset. Offsets > 1 MiB are still offsets, not VAs.
    std::vector<char> buf(256, 0);
    bool ok = false;
    const float* p = resolve_f32_in_buffer(buf.data(), buf.size(), reinterpret_cast<void*>(64),
                                           0, 16, &ok);
    CHECK(ctx, ok);
    CHECK(ctx, p == reinterpret_cast<const float*>(buf.data() + 64));

    std::vector<char> big(3u * 1024u * 1024u, 0);
    const size_t off_1m5 = 1536u * 1024u;
    ok = false;
    p = resolve_f32_in_buffer(big.data(), big.size(),
                              reinterpret_cast<void*>(static_cast<uintptr_t>(off_1m5)), 0, 32,
                              &ok);
    CHECK(ctx, ok);
    CHECK(ctx, p == reinterpret_cast<const float*>(big.data() + off_1m5));
    CHECK(ctx, ptr_looks_like_integer_offset(reinterpret_cast<void*>(off_1m5), big.size()));
    CHECK(ctx, !ptr_looks_like_integer_offset(big.data() + off_1m5, big.size()));

    ok = false;
    p = resolve_f32_in_buffer(buf.data(), buf.size(), nullptr, 16, 8, &ok);
    CHECK(ctx, ok);
    CHECK(ctx, p == reinterpret_cast<const float*>(buf.data() + 16));

    ok = true;
    p = resolve_f32_in_buffer(buf.data(), buf.size(), reinterpret_cast<void*>(250), 0, 16, &ok);
    CHECK(ctx, !ok);
    CHECK(ctx, p == nullptr);

    LiveForwardConfig cfg;
    CHECK(ctx, cfg.n_gpu_layers == 0);
    CHECK(ctx, cfg.n_gpu_layers != 16);
    CHECK(ctx, cfg.n_gpu_layers != 99);
    CHECK(ctx, !cfg.load_mtp);
    CHECK(ctx, cfg.disable_flash_attn);
    CHECK(ctx, !cfg.disable_op_offload);
    CHECK(ctx, !cfg.pack_checkpoint);

    const std::string park_line =
        format_ffn_cuda_park_line(n_park, static_cast<uint64_t>(n_park) *
                                              kQ4FfnLayerBytesMeasured5080);
    CHECK(ctx, park_line.find("ngl=0") != std::string::npos);
    CHECK(ctx, park_line.find("ggml_rebind_q4=0") != std::string::npos);
    CHECK(ctx, park_line.find("never_64=1") != std::string::npos);
    CHECK(ctx, park_line.find("stream=7") != std::string::npos);
    CHECK(ctx, format_ffn_cuda_bind_line(9, false).find("stream=1") != std::string::npos);

    BenchLine onb;
    onb.trace_on = true;
    onb.tok_per_sec = 4.0;
    onb.decode_s = 8.0;
    onb.prefill_s = 2.0;
    onb.prefill_tok = 41;
    onb.host_ffn_binds = 0;
    onb.cuda0_model = 10000ull * 1024ull * 1024ull;
    onb.cuda0_compute = 1026ull * 1024ull * 1024ull;
    onb.nvidia_used = 13000ull * 1024ull * 1024ull;
    const std::string on_line = format_bench_line(onb);
    CHECK(ctx, on_line.find("BENCH TRACE=on") == 0);
    CHECK(ctx, on_line.find("swap_7780=1") != std::string::npos);
    CHECK(ctx, on_line.find("host_ffn_binds=0") != std::string::npos);

    const std::string pcie0 = format_pcie_bound_line(0);
    CHECK(ctx, pcie0.find("B/tok=0") != std::string::npos);
    CHECK(ctx, pcie0.find("ggml_rebind_q4=0") != std::string::npos);
    const std::string pcie7 = format_pcie_bound_line(7);
    CHECK(ctx, pcie7.find("B/tok=") != std::string::npos);
    CHECK(ctx, pcie7.find("n_stream=7") != std::string::npos);
}
