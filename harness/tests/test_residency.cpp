#include "test_common.hpp"

#include "micro_llm/micro_llm.hpp"

#include <cstdio>
#include <string>
#include <vector>

void test_residency_plan(TestContext& ctx) {
    using namespace micro_llm;

    const WeightCatalog cat = default_qwen27b_q4km_catalog();
    CHECK(ctx, !cat.from_gguf_header);
    CHECK(ctx, cat.n_ffn_layers_seen == kNLayers);
    CHECK(ctx, cat.ffn_layer_bytes[0] == kQ4FfnLayerBytes);
    CHECK(ctx, cat.deltanet_bytes > cat.ga_bytes);

    const ResidencyPlan p = plan_residency(cat, 8192, false);
    CHECK(ctx, p.n_gpu_layers == 0);
    CHECK(ctx, p.n_gpu_layers != 16);
    CHECK(ctx, p.n_gpu_layers != 99);
    CHECK(ctx, p.pin_deltanet);
    CHECK(ctx, p.pin_ga);
    CHECK(ctx, !p.pin_embed);
    CHECK(ctx, !p.pin_lm_head);
    CHECK(ctx, p.n_parked_ffn > 0);
    CHECK(ctx, p.n_parked_ffn < kNLayers);
    CHECK(ctx, p.n_parked_ffn + p.n_streamed_ffn == kNLayers);
    CHECK(ctx, p.ffn_parked[0]);
    CHECK(ctx, !p.ffn_parked[kNLayers - 1]);
    CHECK(ctx, p.kv_reserve_tokens == kHourKvReserveTokens);
    CHECK(ctx, p.kv_bytes == hour_kv_reserve_bytes(8192, true));
    CHECK(ctx, p.kv_bytes == 20000ull * kKvBytesPerTokenFp8);
    CHECK(ctx, p.card_stack_bytes <= kHourCardSoftBytes);
    CHECK(ctx, p.card_stack_bytes <= kServeUsableBytes);
    CHECK(ctx, p.pcie_bytes_per_token > 0);
    CHECK(ctx, p.pcie_bytes_per_token ==
                   static_cast<uint64_t>(p.n_streamed_ffn) * kQ4FfnLayerBytes);

    const DecodeBudget b = decode_budget_model(p);
    CHECK(ctx, b.twenty_impossible_if_deltanet_host);
    CHECK(ctx, b.host_ms > 250.0);  // ~3.5 tok/s
    CHECK(ctx, (1000.0 / b.host_ms) < 4.0);
    CHECK(ctx, (1000.0 / b.host_ms) > 3.0);
    CHECK(ctx, b.ffn_cuda_delta_host_ms > 50.0);  // 20 tok/s impossible
    CHECK(ctx, (1000.0 / b.ffn_cuda_delta_host_ms) < 20.0);

    const ResidencyPlan ctx32k = plan_residency(cat, 32768, false);
    CHECK(ctx, ctx32k.kv_reserve_tokens == 32768);
    CHECK(ctx, ctx32k.kv_bytes == 32768ull * kKvBytesPerTokenFp8);

    WeightCatalog huge = cat;
    huge.ga_bytes = 8ull * kGiB;
    huge.deltanet_bytes = 8ull * kGiB;
    const ResidencyPlan cut_park = plan_residency(huge, 8192, false);
    CHECK(ctx, cut_park.n_parked_ffn == 0);
    CHECK(ctx, cut_park.kv_reserve_tokens == kHourKvReserveTokens);
    CHECK(ctx, cut_park.kv_bytes == 20000ull * kKvBytesPerTokenFp8);

    const ResidencyPlan fp16kv = plan_residency(cat, 8192, false, false);
    CHECK(ctx, !fp16kv.use_quant_kv);
    CHECK(ctx, fp16kv.kv_reserve_tokens == kHourKvReserveTokens);
    CHECK(ctx, fp16kv.kv_bytes == 20000ull * kKvBytesPerTokenFp16);
    CHECK(ctx, fp16kv.n_parked_ffn < kNLayers);

    const ResidencyPlan host_dn = plan_residency(cat, 8192, true);
    CHECK(ctx, !host_dn.pin_deltanet);
    CHECK(ctx, !host_dn.twenty_tok_s_possible_pcie5);

    CHECK(ctx, classify_weight_tensor("blk.3.attn_q.weight") == TensorClass::GaQkvO);
    CHECK(ctx, classify_weight_tensor("blk.4.attn_qkv.weight") == TensorClass::DeltaNet);
    CHECK(ctx, classify_weight_tensor("blk.0.ffn_gate.weight") == TensorClass::Ffn);
    CHECK(ctx, classify_weight_tensor("token_embd.weight") == TensorClass::Embed);
    CHECK(ctx, classify_weight_tensor("output.weight") == TensorClass::LmHead);
    CHECK(ctx, classify_weight_tensor("blk.64.nextn.weight") == TensorClass::Mtp);
    int32_t layer = -2;
    CHECK(ctx, classify_weight_tensor("blk.12.ffn_up.weight", &layer) == TensorClass::Ffn);
    CHECK(ctx, layer == 12);

    const auto vram = residency_vram_regexes(p);
    bool has_ga = false, has_dn = false, has_ffn = false;
    for (const auto& r : vram) {
        if (r.find("3|7|11") != std::string::npos) has_ga = true;
        if (r.find("ssm_") != std::string::npos) has_dn = true;
        if (r.find("ffn_gate") != std::string::npos) has_ffn = true;
    }
    CHECK(ctx, has_ga && has_dn && has_ffn);

    const auto host = residency_cuda_host_regexes(p);
    bool has_emb = false;
    for (const auto& r : host) {
        if (r.find("token_embd") != std::string::npos) has_emb = true;
    }
    CHECK(ctx, has_emb);

    std::vector<GgufTensorDirRow> rows;
    rows.push_back({"blk.0.ffn_gate.weight", 50ull * 1024ull * 1024ull, 17408ull * 5120ull, 12});
    rows.push_back({"blk.0.ffn_up.weight", 50ull * 1024ull * 1024ull, 17408ull * 5120ull, 12});
    rows.push_back({"blk.0.ffn_down.weight", 50ull * 1024ull * 1024ull, 5120ull * 17408ull, 12});
    rows.push_back({"blk.3.attn_q.weight", 8ull * 1024ull * 1024ull, 5120ull * 5120ull, 12});
    rows.push_back({"token_embd.weight", 20ull * 1024ull * 1024ull, 248320ull * 5120ull, 12});
    const std::string path = "test_catalog.gguf";
    CHECK(ctx, write_gguf_tensor_dir_stub(path, "qwen35", rows));
    WeightCatalog loaded;
    std::string err;
    CHECK(ctx, read_gguf_weight_catalog(path, loaded, &err));
    CHECK(ctx, loaded.from_gguf_header);
    CHECK(ctx, loaded.ffn_layer_bytes[0] >= 100ull * 1024ull * 1024ull);
    CHECK(ctx, loaded.ga_bytes > 0);
    CHECK(ctx, loaded.embed_bytes > 0);
    std::remove(path.c_str());

    const std::string mtp_path = "test_mtp65.gguf";
    CHECK(ctx, write_gguf_kv_stub(mtp_path, "qwen35", true, true, kNLayers + 1, kHiddenDim,
                                 kFfnIntermediate, nullptr, nullptr, 1));
    GgufKv mtp;
    CHECK(ctx, read_gguf_meta(mtp_path, mtp));
    CHECK(ctx, mtp.n_layers == kNLayers + 1);
    CHECK(ctx, mtp.n_nextn == 1);
    CHECK(ctx, gguf_looks_like_qwen27b_hybrid(mtp));
    CHECK(ctx, gguf_hook_layer_count(mtp) == kNLayers);
    std::remove(mtp_path.c_str());

    PerfTelemetry tel;
    tel.reset();
    tel.apply_plan(p);
    tel.set_generated(256, 10.0);
    tel.set_hook_counters(0, 3, 2);
    tel.mark_measured();
    const std::string report = tel.format_report();
    CHECK(ctx, report.find("tok/s=") != std::string::npos);
    CHECK(ctx, report.find("pcie_model_B/token=") != std::string::npos);
    CHECK(ctx, report.find("IMPOSSIBLE if DeltaNet host") != std::string::npos);
    CHECK(ctx, report.find("missing_hooks=3") != std::string::npos);
    CHECK(ctx, report.find("missing_hooks_scope=run") != std::string::npos);
    CHECK(ctx, report.find("PERFORMANCE ") != std::string::npos);
    CHECK(ctx, report.find("swap_gate ") != std::string::npos);
    CHECK(ctx, format_tokens_per_sec_line(1.5, 8, 5.3).find("tokens/s=") == 0);
    CHECK(ctx, format_ffn_cuda_bind_line(3, true).find("ffn_cuda_bind") == 0);
    CHECK(ctx, format_ffn_cuda_park_line(8, 1000).find("ffn_cuda_park") == 0);

    CHECK(ctx, swap_gate_ok(3.51, 0, 0));
    CHECK(ctx, !swap_gate_ok(3.5, 0, 0));
    CHECK(ctx, !swap_gate_ok(20.0, 1, 0));
    CHECK(ctx, !swap_gate_ok(20.0, 0, 1));
    const std::string perf = format_performance_line(tel.snap());
    CHECK(ctx, perf.find("PERFORMANCE ") == 0);
    CHECK(ctx, perf.find("missing_hooks=3") != std::string::npos);
    CHECK(ctx, perf.find("host_ffn_binds=0") != std::string::npos);
    CHECK(ctx, format_swap_gate_line(tel.snap()).find("ok=0") != std::string::npos);
    tel.set_hook_counters(0, 0, 0);
    CHECK(ctx, format_swap_gate_line(tel.snap()).find("ok=1") != std::string::npos);
}
