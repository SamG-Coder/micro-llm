#include "micro_llm/trace_cli.hpp"
#include "micro_llm/ggml_ptr.hpp"
#include "micro_llm/vram_ledger.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace micro_llm {

int32_t hybrid_n_gpu_layers() { return kHybridNGpuLayers; }

int32_t clamp_hybrid_n_gpu_layers(int32_t n) {
    if (n == 16 || n == 99) {
        return kHybridNGpuLayers;
    }
    return n;
}

uint32_t hybrid_ffn_park_layers() { return vram_ledger_slots_first().n_parked_ffn; }

std::vector<std::string> hybrid_cpu_tensor_regexes() {
    // Only MTP + lm_head stay host. FFN + DeltaNet on CUDA collapses the
    // 340-split graph (5080: DeltaNet-CPU + 7 streamed FFN-CPU = 0.52 tok/s).
    return {
        "nextn",
        "shared_head",
        "output\\.weight",
    };
}

std::vector<std::string> hybrid_gpu_tensor_regexes(uint32_t n_park) {
    (void)n_park;
    std::vector<std::string> out;
    // ALL FFN on CUDA. Partial park (0..n) left streamed layers on host and
    // split the graph (5080: 0.96 tok/s). A static graph cannot share two
    // slots across N Q4 tensors; leftover after slots+KV parks all 64.
    out.push_back("blk\\.[0-9]+\\.ffn_gate");
    out.push_back("blk\\.[0-9]+\\.ffn_up");
    out.push_back("blk\\.[0-9]+\\.ffn_down");
    // DeltaNet on CUDA. Leaving it on host splits every layer against parked FFN.
    out.push_back("blk\\.[0-9]+\\.ssm_");
    out.push_back("blk\\.[0-9]+\\.attn_qkv");
    out.push_back("blk\\.[0-9]+\\.attn_gate");
    // 16 Gated Attention QKVO (layers 3,7,...,63).
    out.push_back("blk\\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\\.attn_q[^k]");
    out.push_back("blk\\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\\.attn_k");
    out.push_back("blk\\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\\.attn_v");
    out.push_back("blk\\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\\.attn_output");
    out.push_back("blk\\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\\.attn_norm");
    out.push_back("token_embd");
    return out;
}

uint64_t pinned_ga_weight_bytes() { return kPinnedGaWeightBytes; }

uint64_t pinned_ga_kv_bytes() { return kPinnedGaWeightBytes + kHourKvReserveBytes; }

uint64_t streamed_ffn_workspace_bytes() { return kStreamWorkspaceBytes; }

std::string format_ffn_cuda_park_line(uint32_t n_park, uint64_t bytes) {
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "ffn_cuda_park n=%u bytes=%llu stream_slots=%u stream_bytes=%u "
                  "ngl=0 kv20k_bytes=%llu ggml_rebind_q4=%d",
                  n_park, static_cast<unsigned long long>(bytes), kNStreamSlots,
                  kStreamWorkspaceBytes, static_cast<unsigned long long>(kHourKvReserveBytes),
                  ggml_can_rebind_q4_midgraph() ? 1 : 0);
    return buf;
}

std::string format_ffn_cuda_bind_line(uint32_t layer, bool parked) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "ffn_cuda_bind layer=%u parked=%d stream=%d workspace=cuda",
                  layer, parked ? 1 : 0, parked ? 0 : 1);
    return buf;
}

std::string format_ffn_slot_bind_line(int slot, uint64_t bytes, bool real_h2d) {
    char buf[160];
    std::snprintf(buf, sizeof(buf),
                  "ffn_slot_bind slot=%d bytes=%llu real_h2d=%d (cudaMalloc_alone=0)",
                  slot, static_cast<unsigned long long>(bytes), real_h2d ? 1 : 0);
    return buf;
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
    out.cfg.disable_op_offload = false;
    out.cfg.load_mtp = false;
    out.cfg.pack_checkpoint = false;
    out.cfg.trace_hooks = false;
    out.cfg.n_parked_ffn = hybrid_ffn_park_layers();

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
        } else if (std::strcmp(argv[i], "--trace-off") == 0 ||
                   std::strcmp(argv[i], "--no-trace") == 0) {
            out.cfg.trace_hooks = false;
            out.trace_off_set = true;
        } else if (std::strcmp(argv[i], "--trace-on") == 0) {
            out.cfg.trace_hooks = true;
            out.trace_on_set = true;
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
