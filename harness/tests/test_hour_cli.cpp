#include "test_common.hpp"

#include "micro_llm/micro_llm.hpp"

#include <cstring>
#include <string>
#include <vector>

void test_hour_cli_resolve(TestContext& ctx) {
    using namespace micro_llm;

    CHECK(ctx, resolve_n_predict(true, false, 64) == kHourNPredict);
    CHECK(ctx, resolve_n_predict(true, true, 64) == 64);
    CHECK(ctx, resolve_n_predict(false, false, 64) == 64);
    CHECK(ctx, kHourNPredict == 20000u);
    CHECK(ctx, kHourCheckpointEvery == 2000u);

    CHECK(ctx, hybrid_n_gpu_layers() == 0);
    CHECK(ctx, hybrid_n_gpu_layers() != 16);
    CHECK(ctx, hybrid_n_gpu_layers() != 99);
    CHECK(ctx, clamp_hybrid_n_gpu_layers(16) == 0);
    CHECK(ctx, clamp_hybrid_n_gpu_layers(99) == 0);
    CHECK(ctx, clamp_hybrid_n_gpu_layers(0) == 0);
    const auto pats = hybrid_cpu_tensor_regexes();
    bool has_gate = false, has_up = false, has_down = false, has_ssm = false;
    bool has_mtp = false, has_wrong_ngl = false;
    for (const auto& p : pats) {
        if (p.find("ffn_gate") != std::string::npos) has_gate = true;
        if (p.find("ffn_up") != std::string::npos) has_up = true;
        if (p.find("ffn_down") != std::string::npos) has_down = true;
        if (p.find("ssm_") != std::string::npos) has_ssm = true;
        if (p.find("nextn") != std::string::npos) has_mtp = true;
        if (p.find("ngl") != std::string::npos || p == "16" || p == "99") has_wrong_ngl = true;
    }
    CHECK(ctx, has_gate && has_up && has_down && has_ssm && has_mtp);
    CHECK(ctx, !has_wrong_ngl);
    const auto gpu = hybrid_gpu_tensor_regexes();
    bool has_ga = false, gpu_has_ffn = false, gpu_has_qkv = false;
    for (const auto& p : gpu) {
        if (p.find("3|7|11") != std::string::npos) has_ga = true;
        if (p.find("ffn_") != std::string::npos) gpu_has_ffn = true;
        if (p.find("attn_qkv") != std::string::npos) gpu_has_qkv = true;
    }
    CHECK(ctx, has_ga);
    CHECK(ctx, gpu_has_ffn);  // parked FFN on CUDA, not ngl=99 of the whole file
    CHECK(ctx, gpu_has_qkv);  // DeltaNet on CUDA — proven required for 20 tok/s
    const auto gpu_park = hybrid_gpu_tensor_regexes(hybrid_ffn_park_layers());
    bool gpu_parks_ffn = false;
    for (const auto& p : gpu_park) {
        if (p.find("ffn_gate") != std::string::npos) gpu_parks_ffn = true;
    }
    CHECK(ctx, gpu_parks_ffn);
    CHECK(ctx, hybrid_ffn_park_layers() > 0);
    CHECK(ctx, hybrid_ffn_park_layers() < kNLayers);
    CHECK(ctx, hour_park_stream_fits(hybrid_ffn_park_layers()));
    CHECK(ctx, !hour_park_stream_fits(kNLayers));
    CHECK(ctx, pinned_ga_weight_bytes() > 0);
    CHECK(ctx, pinned_ga_weight_bytes() < kGiB);  // card stack, not 15.3GB host GGUF
    CHECK(ctx, pinned_ga_kv_bytes() == kPinnedGaKvBytes);
    CHECK(ctx, streamed_hour_card_fits());
    CHECK(ctx, parked_all_ffn_exceeds_card());

    const char* prompt = default_coding_assistant_prompt();
    CHECK(ctx, prompt != nullptr);
    const std::string p(prompt);
    CHECK(ctx, p.find("merge_intervals") == std::string::npos);
    CHECK(ctx, p.find("binary_search") == std::string::npos);
    CHECK(ctx, p.find("slugify") == std::string::npos);
    CHECK(ctx, p.find("string_view") == std::string::npos);
    CHECK(ctx, p.find("BUF") == std::string::npos);

    char ui[] = "--ui";
    char* ui_only[] = {const_cast<char*>("micro-llm-trace"), ui, nullptr};
    const TraceCliArgs a1 = parse_trace_cli(2, ui_only);
    CHECK(ctx, resolve_trace_mode(a1) == TraceCliMode::SampleUi);
    CHECK(ctx, a1.cfg.n_predict == kCliTestNPredict);

    char model[] = "--model";
    char path[] = "qwen.gguf";
    char* model_only[] = {const_cast<char*>("micro-llm-trace"), model, path, nullptr};
    const TraceCliArgs a2 = parse_trace_cli(3, model_only);
    CHECK(ctx, resolve_trace_mode(a2) == TraceCliMode::HourLive);
    CHECK(ctx, a2.cfg.n_predict == kHourNPredict);
    CHECK(ctx, a2.cfg.n_gpu_layers == 0);
    CHECK(ctx, a2.cfg.n_gpu_layers != 16);
    CHECK(ctx, a2.cfg.n_gpu_layers != 99);
    CHECK(ctx, a2.cfg.n_batch == 512);
    CHECK(ctx, a2.cfg.n_ubatch == 32);
    CHECK(ctx, a2.cfg.checkpoint_every == 2000);
    CHECK(ctx, a2.cfg.continue_after_eos);
    CHECK(ctx, !a2.cfg.disable_flash_attn);  // FA on when FFN+DeltaNet are CUDA
    CHECK(ctx, a2.cfg.disable_op_offload);
    CHECK(ctx, !a2.cfg.load_mtp);
    CHECK(ctx, !a2.cfg.pack_checkpoint);
    CHECK(ctx, a2.cfg.n_parked_ffn > 0);
    CHECK(ctx, a2.cfg.n_parked_ffn < kNLayers);
    CHECK(ctx, format_tokens_per_sec_line(1.5, 8, 5.3).find("tokens/s=") == 0);
    CHECK(ctx, format_ffn_cuda_bind_line(3, true).find("ffn_cuda_bind") == 0);
    CHECK(ctx, format_ffn_cuda_park_line(8, 1000).find("ffn_cuda_park") == 0);

    char* both[] = {const_cast<char*>("micro-llm-trace"), ui, model, path, nullptr};
    const TraceCliArgs a3 = parse_trace_cli(4, both);
    CHECK(ctx, resolve_trace_mode(a3) == TraceCliMode::HourLive);
    CHECK(ctx, a3.want_ui);
    CHECK(ctx, !a3.cfg.model_path.empty());
    CHECK(ctx, a3.cfg.n_predict == kHourNPredict);

    char stub[] = "--stub";
    char np[] = "--n-predict";
    char two[] = "2";
    char* stub_args[] = {const_cast<char*>("micro-llm-trace"), stub, np, two, nullptr};
    const TraceCliArgs a4 = parse_trace_cli(4, stub_args);
    CHECK(ctx, resolve_trace_mode(a4) == TraceCliMode::HourHeadless);
    CHECK(ctx, a4.cfg.n_predict == 2);

    char* stub_ui[] = {const_cast<char*>("micro-llm-trace"), stub, ui, np, two, nullptr};
    const TraceCliArgs a5 = parse_trace_cli(5, stub_ui);
    CHECK(ctx, resolve_trace_mode(a5) == TraceCliMode::HourLive);

    // Stub hour still emits HTR1 and can checkpoint without a GGUF.
    LiveForwardConfig cfg;
    cfg.n_predict = 2;
    cfg.checkpoint_every = 1;
    cfg.out_path = "stub_hour_ckpt.mlpt";
    std::vector<uint32_t> tokens;
    cfg.on_htr1 = [&](const uint8_t* rec, size_t n) {
        CHECK(ctx, rec != nullptr);
        CHECK(ctx, n == kHtr1RecordBytes);
        Htr1TokenMeta meta;
        decode_htr1_record(rec, &meta, nullptr, nullptr, nullptr);
        tokens.push_back(meta.sampled_id);
    };
    StreamerConfig scfg;
    scfg.ffn_scratch_bytes = 4096;
    TraceStreamer streamer(scfg);
    TraceHooks hooks;
    auto stub_be = make_live_forward("stub");
    const LiveForwardStatus st = stub_be->run(hooks, streamer, cfg);
    CHECK(ctx, st.ok);
    CHECK(ctx, tokens.size() == 2);
    PruneTable loaded;
    std::string err;
    CHECK(ctx, load_prune_table(loaded, cfg.out_path, &err));
    CHECK(ctx, loaded.n_tokens == 2);
    std::remove(cfg.out_path.c_str());
    std::remove((cfg.out_path + ".tmp").c_str());
}
