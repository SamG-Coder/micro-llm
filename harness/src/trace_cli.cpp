#include "micro_llm/trace_cli.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace micro_llm {

int32_t hybrid_n_gpu_layers() { return kHybridNGpuLayers; }

std::vector<std::string> hybrid_cpu_tensor_regexes() {
    // n_gpu_layers=99 puts tensors on CUDA. These regexes pull FFN and
    // DeltaNet weights back to CPU. GA QKVO (attn_q/k/v/output on layers
    // 3,7,...,63) is not in this list and stays on the card with KV.
    return {
        "blk\\.[0-9]+\\.ffn_gate",
        "blk\\.[0-9]+\\.ffn_up",
        "blk\\.[0-9]+\\.ffn_down",
        "blk\\.[0-9]+\\.ssm_",
        "blk\\.[0-9]+\\.attn_qkv",
        "blk\\.[0-9]+\\.attn_gate",
    };
}

uint64_t pinned_ga_weight_bytes() {
    // 16 Gated Attention blocks at Q4, ~0.6 GiB. Card stack, not host GGUF.
    return (6ull * kGiB) / 10ull;
}

TraceCliArgs parse_trace_cli(int argc, char** argv) {
    TraceCliArgs out;
    out.cfg.prompt = default_coding_assistant_prompt();
    out.cfg.n_predict = kCliTestNPredict;
    out.cfg.n_gpu_layers = kHybridNGpuLayers;
    out.cfg.n_batch = 512;
    out.cfg.n_ubatch = 32;
    out.cfg.checkpoint_every = kHourCheckpointEvery;
    out.cfg.continue_after_eos = true;
    out.cfg.disable_flash_attn = true;
    out.cfg.disable_op_offload = true;

    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires an argument\n", flag);
                out.parse_error = true;
                return "";
            }
            return argv[++i];
        };
        if (out.parse_error) {
            break;
        }
        if (std::strcmp(argv[i], "--model") == 0) {
            out.cfg.model_path = need("--model");
        } else if (std::strcmp(argv[i], "--prompt") == 0) {
            out.cfg.prompt = need("--prompt");
        } else if (std::strcmp(argv[i], "--out") == 0) {
            out.cfg.out_path = need("--out");
        } else if (std::strcmp(argv[i], "--n-predict") == 0) {
            out.cfg.n_predict = static_cast<uint32_t>(std::strtoul(need("--n-predict"), nullptr, 10));
            out.n_predict_set = true;
        } else if (std::strcmp(argv[i], "--top-k") == 0) {
            out.cfg.top_k = static_cast<uint32_t>(std::strtoul(need("--top-k"), nullptr, 10));
        } else if (std::strcmp(argv[i], "--ctx") == 0) {
            out.cfg.n_ctx = static_cast<uint32_t>(std::strtoul(need("--ctx"), nullptr, 10));
        } else if (std::strcmp(argv[i], "--stub") == 0) {
            out.use_stub = true;
        } else if (std::strcmp(argv[i], "--ui") == 0) {
            out.want_ui = true;
        } else if (std::strcmp(argv[i], "--ui-check") == 0) {
            out.ui_check = true;
        } else if (std::strcmp(argv[i], "--check-serve") == 0) {
            out.check_serve = need("--check-serve");
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            out.help = true;
        } else {
            out.parse_error = true;
        }
    }

    out.cfg.n_predict =
        resolve_n_predict(!out.cfg.model_path.empty(), out.n_predict_set, out.cfg.n_predict);
    return out;
}

}  // namespace micro_llm
