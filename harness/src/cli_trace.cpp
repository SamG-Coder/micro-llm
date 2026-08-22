#include "micro_llm/gguf_meta.hpp"
#include "micro_llm/hotspot_ui.hpp"
#include "micro_llm/live_forward.hpp"
#include "micro_llm/micro_llm.hpp"
#include "micro_llm/serve.hpp"

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
                 "No args / --ui  Open the sample hotspot window. Does not load a GGUF.\n"
                 "                Does not start the hour. Windows uses WebView2.\n"
                 "--ui-check      Locate committed UI files and exit. No window.\n"
                 "\n"
                 "Live Qwen 27B (3.6/3.8 hybrid) hour dump. Job: local coding assistant.\n"
                 "Does not load vision. Hour stays down unless --model or --stub.\n"
                 "\n"
                 "  --model     Qwen 27B GGUF (qwen35). Starts the hour. Not required for --ui.\n"
                 "  --prompt    Job prompt. Default is the coding-assistant prompt.\n"
                 "  --out       MLPT path (default prune_table.bin)\n"
                 "  --n-predict Tokens to generate after the prompt (default 64)\n"
                 "  --stub      Tests only. Synthetic hooks, no GGUF.\n"
                 "  --check-serve PATH  Print remnant_may_serve for a remnant GGUF and exit.\n",
                 argv0);
}

}  // namespace

int main(int argc, char** argv) {
    using namespace micro_llm;

    LiveForwardConfig cfg;
    cfg.prompt = default_coding_assistant_prompt();
    bool use_stub = false;
    bool want_ui = false;
    bool ui_check = false;
    std::string check_serve;

    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires an argument\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        if (std::strcmp(argv[i], "--model") == 0) {
            cfg.model_path = need("--model");
        } else if (std::strcmp(argv[i], "--prompt") == 0) {
            cfg.prompt = need("--prompt");
        } else if (std::strcmp(argv[i], "--out") == 0) {
            cfg.out_path = need("--out");
        } else if (std::strcmp(argv[i], "--n-predict") == 0) {
            cfg.n_predict = static_cast<uint32_t>(std::strtoul(need("--n-predict"), nullptr, 10));
        } else if (std::strcmp(argv[i], "--top-k") == 0) {
            cfg.top_k = static_cast<uint32_t>(std::strtoul(need("--top-k"), nullptr, 10));
        } else if (std::strcmp(argv[i], "--ctx") == 0) {
            cfg.n_ctx = static_cast<uint32_t>(std::strtoul(need("--ctx"), nullptr, 10));
        } else if (std::strcmp(argv[i], "--stub") == 0) {
            use_stub = true;
        } else if (std::strcmp(argv[i], "--ui") == 0) {
            want_ui = true;
        } else if (std::strcmp(argv[i], "--ui-check") == 0) {
            ui_check = true;
        } else if (std::strcmp(argv[i], "--check-serve") == 0) {
            check_serve = need("--check-serve");
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (!check_serve.empty()) {
        if (!file_exists(check_serve)) {
            std::fprintf(stderr, "error: remnant GGUF not found: %s\n", check_serve.c_str());
            return 2;
        }
        std::string err;
        const bool ok = remnant_may_serve_file(check_serve, &err);
        std::printf("remnant_may_serve=%s\n", ok ? "true" : "false");
        if (!ok) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        return 0;
    }

    const bool hour = use_stub || !cfg.model_path.empty();
    if (!hour || want_ui || ui_check) {
        if (hour && (want_ui || ui_check)) {
            std::fprintf(stderr,
                         "note: --ui does not load a GGUF; hour flags were ignored\n");
        }
        HotspotUiOptions opt;
        opt.check_only = ui_check;
        return run_hotspot_ui(opt);
    }

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

    StreamerConfig scfg;
    if (use_stub) {
        scfg.ffn_scratch_bytes = 4096;
    }
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
