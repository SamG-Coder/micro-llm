#include "micro_llm/gguf_meta.hpp"
#include "micro_llm/hotspot_ui.hpp"
#include "micro_llm/hook_ring.hpp"
#include "micro_llm/live_forward.hpp"
#include "micro_llm/micro_llm.hpp"
#include "micro_llm/serve.hpp"
#include "micro_llm/trace_cli.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef _WIN32
#ifndef S_ISREG
#define S_ISREG(m) (((m) & _S_IFMT) == _S_IFREG)
#endif
#endif
#include <sys/stat.h>

namespace {

bool file_exists(const std::string& path) {
    struct stat st {};
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s [--ui] [--ui-check]\n"
                 "          --model PATH.gguf [--prompt TEXT] [--out prune_table.bin]\n"
                 "          [--n-predict N] [--top-k K] [--ctx N] [--stub]\n"
                 "\n"
                 "--ui              Sample hotspot window. No GGUF. Does not start the hour.\n"
                 "--ui-check        Locate committed UI files and exit. No window.\n"
                 "\n"
                 "--model PATH      Start the hour. Opens the live window by default.\n"
                 "                  --ui --model is the same: window AND decode.\n"
                 "                  --n-predict defaults to 20000 when --model is set.\n"
                 "  --prompt        Job prompt. Default is the coding-assistant prompt.\n"
                 "  --out           MLPT path (default prune_table.bin). Checkpoint every 2000.\n"
                 "  --n-predict     Tokens to generate (default 64 in tests; 20000 with --model).\n"
                 "  --stub          Tests only. Synthetic hooks, no GGUF.\n"
                 "  --check-serve PATH  Print remnant_may_serve for a remnant GGUF and exit.\n",
                 argv0);
}

int run_hour(micro_llm::LiveForwardConfig cfg, bool use_stub, bool push_ui) {
    using namespace micro_llm;
    cfg.abort = &hotspot_live_abort();
    if (push_ui) {
        const uint64_t ga = pinned_ga_weight_bytes();
        const uint64_t ffn_ws = streamed_ffn_workspace_bytes();
        const uint32_t n_park = cfg.n_parked_ffn ? cfg.n_parked_ffn : hybrid_ffn_park_layers();
        const std::string attach =
            std::string("{\"type\":\"live-attach\",\"title\":\"live hour\",") +
            "\"weightBytes\":" + std::to_string(ga) +
            ",\"cudaBytes\":" + std::to_string(kCudaScratchBytes) +
            ",\"ffnWorkspaceBytes\":" + std::to_string(ffn_ws) +
            ",\"ffnParkedBytes\":" +
            std::to_string(static_cast<uint64_t>(n_park) * kQ4FfnLayerBytes) +
            ",\"nParkedFfn\":" + std::to_string(n_park) +
            ",\"pinnedGaKvBytes\":" + std::to_string(pinned_ga_kv_bytes()) +
            ",\"kvBytesPerToken\":" + std::to_string(kKvBytesPerTokenFp16) +
            ",\"ctx\":" + std::to_string(cfg.n_ctx) +
            ",\"nPredict\":" + std::to_string(cfg.n_predict) +
            ",\"streamFfn\":true,\"nGpuLayers\":0,\"loadMtp\":false}";
        hotspot_live_push_json(attach);
        cfg.on_htr1 = [](const uint8_t* rec, size_t n) { hotspot_live_push_htr1(rec, n); };
        cfg.on_stats = [](double tps, uint32_t ntok) {
            char buf[192];
            std::snprintf(buf, sizeof(buf),
                          "{\"type\":\"live-stats\",\"tokensPerSec\":%.3f,\"nTokens\":%u}", tps,
                          ntok);
            hotspot_live_push_json(buf);
        };
    }

    StreamerConfig scfg;
    if (use_stub) {
        scfg.ffn_scratch_bytes = 4096;
    }
    scfg.n_parked_ffn = cfg.n_parked_ffn ? cfg.n_parked_ffn : hybrid_ffn_park_layers();
    scfg.log_cuda_ffn = !use_stub;
    TraceStreamer streamer(scfg);
    TraceHooks hooks;
    auto backend = make_live_forward(use_stub ? "stub" : "llama");

    const LiveForwardStatus st = backend->run(hooks, streamer, cfg);
    if (!st.ok) {
        std::fprintf(stderr, "error: %s\n", st.message.c_str());
        std::fprintf(stderr, "backend=%s engine_linked=%d architecture_ok=%d\n",
                     backend->name(), backend->engine_linked() ? 1 : 0,
                     st.architecture_ok ? 1 : 0);
        return use_stub ? 1 : 3;
    }

    std::string err;
    if (!save_prune_table(hooks.table(), cfg.out_path, &err)) {
        std::fprintf(stderr, "save failed: %s\n", err.c_str());
        return 1;
    }
    std::printf("wrote %s\n", cfg.out_path.c_str());
    std::printf("backend=%s stub=%d engine_linked=%d tokens=%llu\n", backend->name(),
                backend->is_stub() ? 1 : 0, backend->engine_linked() ? 1 : 0,
                static_cast<unsigned long long>(hooks.table().n_tokens));
    std::printf("%s\n", st.message.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace micro_llm;

    const TraceCliArgs parsed = parse_trace_cli(argc, argv);
    if (parsed.help || parsed.parse_error) {
        usage(argv[0]);
        return parsed.help ? 0 : 2;
    }

    if (resolve_trace_mode(parsed) == TraceCliMode::CheckServe) {
        if (!file_exists(parsed.check_serve)) {
            std::fprintf(stderr, "error: remnant GGUF not found: %s\n", parsed.check_serve.c_str());
            return 2;
        }
        std::string err;
        const bool ok = remnant_may_serve_file(parsed.check_serve, &err);
        std::printf("remnant_may_serve=%s\n", ok ? "true" : "false");
        if (!ok) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        return 0;
    }

    if (resolve_trace_mode(parsed) == TraceCliMode::UiCheck) {
        HotspotUiOptions opt;
        opt.check_only = true;
        return run_hotspot_ui(opt);
    }

    if (resolve_trace_mode(parsed) == TraceCliMode::SampleUi) {
        HotspotUiOptions opt;
        opt.live = false;
        opt.title = "micro-llm hotspot — sample ring";
        return run_hotspot_ui(opt);
    }

    LiveForwardConfig cfg = parsed.cfg;
    const bool use_stub = parsed.use_stub;

    if (!use_stub) {
        if (cfg.model_path.empty()) {
            std::fprintf(stderr,
                         "error: --model PATH.gguf is required for a real 27B hour\n"
                         "(use --stub only in tests; that does not produce a real MLPT)\n");
            return 2;
        }
        if (!file_exists(cfg.model_path)) {
            std::fprintf(stderr, "error: no weights at %s\n", cfg.model_path.c_str());
            return 2;
        }
    }

    const bool live_ui = resolve_trace_mode(parsed) == TraceCliMode::HourLive;
    if (!live_ui) {
        return run_hour(cfg, use_stub, false);
    }

    HotspotUiOptions opt;
    opt.live = true;
    opt.title = "micro-llm hotspot — live hour";
    opt.run_hour = [cfg, use_stub]() { return run_hour(cfg, use_stub, true); };
    return run_hotspot_ui(opt);
}
