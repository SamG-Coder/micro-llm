#include "micro_llm/trace_cli.hpp"
#include "micro_llm/residency.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace micro_llm {
namespace {

std::string layer_alt(uint32_t lo, uint32_t hi_exclusive) {
    std::ostringstream os;
    os << '(';
    for (uint32_t i = lo; i < hi_exclusive; ++i) {
        if (i != lo) {
            os << '|';
        }
        os << i;
    }
    os << ')';
    return os.str();
}

}  // namespace

int32_t hybrid_n_gpu_layers() { return kHybridNGpuLayers; }

int32_t clamp_hybrid_n_gpu_layers(int32_t n) {
    if (n == 16 || n == 99) {
        return kHybridNGpuLayers;
    }
    return n;
}

uint32_t hybrid_ffn_park_layers() { return ffn_park_layers_that_fit(); }

std::vector<std::string> hybrid_cpu_tensor_regexes() {
    return residency_cpu_regexes(plan_residency(default_qwen27b_q4km_catalog(), kDefaultServeCtx));
}

std::vector<std::string> hybrid_gpu_tensor_regexes(uint32_t n_park) {
    if (n_park > kNLayers - 1u) {
        n_park = kNLayers - 1u;
    }
    std::vector<std::string> out;
    if (n_park > 0) {
        const std::string alt = layer_alt(0, n_park);
        out.push_back("blk\\." + alt + "\\.ffn_gate");
        out.push_back("blk\\." + alt + "\\.ffn_up");
        out.push_back("blk\\." + alt + "\\.ffn_down");
    }
    out.push_back("blk\\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\\.attn_q[^k]");
    out.push_back("blk\\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\\.attn_k");
    out.push_back("blk\\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\\.attn_v");
    out.push_back("blk\\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\\.attn_output");
    out.push_back("blk\\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\\.attn_norm");
    out.push_back("blk\\.[0-9]+\\.ssm_");
    out.push_back("blk\\.[0-9]+\\.attn_qkv");
    out.push_back("blk\\.[0-9]+\\.attn_gate");
    return out;
}

uint64_t pinned_ga_weight_bytes() { return kPinnedGaWeightBytes; }

uint64_t pinned_ga_kv_bytes() { return kPinnedGaKvBytes; }

uint64_t streamed_ffn_workspace_bytes() { return kQ4FfnLayerBytes; }

TraceCliArgs parse_trace_cli(int argc, char** argv) {
    TraceCliArgs out;
    out.cfg.prompt = default_coding_assistant_prompt();
    out.cfg.n_predict = kCliTestNPredict;
    out.cfg.n_gpu_layers = kHybridNGpuLayers;
    out.cfg.n_batch = 512;
    out.cfg.n_ubatch = 32;
    out.cfg.checkpoint_every = kHourCheckpointEvery;
    out.cfg.continue_after_eos = true;
    out.cfg.disable_flash_attn = false;
    out.cfg.disable_op_offload = true;
    out.cfg.load_mtp = false;
    out.cfg.pack_checkpoint = false;
    out.cfg.n_parked_ffn = hybrid_ffn_park_layers();
    out.cfg.use_quant_kv = true;

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
        } else if (std::strcmp(argv[i], "--no-flash-attn") == 0) {
            out.cfg.disable_flash_attn = true;
        } else if (std::strcmp(argv[i], "--host-deltanet") == 0) {
            out.cfg.force_host_deltanet = true;
        } else if (std::strcmp(argv[i], "--no-quant-kv") == 0) {
            out.cfg.use_quant_kv = false;
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            out.help = true;
        } else {
            out.parse_error = true;
        }
    }

    out.cfg.n_predict =
        resolve_n_predict(!out.cfg.model_path.empty(), out.n_predict_set, out.cfg.n_predict);
    out.cfg.n_gpu_layers = clamp_hybrid_n_gpu_layers(out.cfg.n_gpu_layers);
    return out;
}

}  // namespace micro_llm
