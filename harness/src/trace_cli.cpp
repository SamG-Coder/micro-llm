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
    // Only MTP + lm_head stay host. output_norm is NOT here — a CPU
    // 180K output_norm after CUDA0 layer-norm AVs (974b0c3 #641).
    // Do NOT send 57–63 FFN to CUDA_Host (zero-copy GEMM, 340 splits).
    return {
        "nextn",
        "shared_head",
        "output\\.weight",
    };
}

std::vector<std::string> hybrid_gpu_tensor_regexes(uint32_t n_park) {
    if (n_park > kMaxParkedFfnLayers) {
        n_park = kMaxParkedFfnLayers;
    }
    std::vector<std::string> out;
    // Parked FFN only (layers 0..n_park-1). A catch-all blk.[0-9]+.ffn_*
    // parks 57–63 too (never-64, CUDA0 14360, real_h2d=0). One pattern
    // per parked layer so blk.5 does not swallow blk.57.
    for (uint32_t layer = 0; layer < n_park; ++layer) {
        const std::string n = std::to_string(layer);
        out.push_back("blk\\." + n + "\\.ffn_gate");
        out.push_back("blk\\." + n + "\\.ffn_up");
        out.push_back("blk\\." + n + "\\.ffn_down");
    }
    // Do NOT GPU-override 57–63 FFN (extra park / extra CUDA weight
    // buffers). 63 gate+up+down bind into slot A; 62 into B.
    // DeltaNet on CUDA. Leaving it on host splits every layer against parked FFN.
    out.push_back("blk\\.[0-9]+\\.ssm_");
    out.push_back("blk\\.[0-9]+\\.attn_qkv");
    out.push_back("blk\\.[0-9]+\\.attn_gate");
    // Tiny F32 norms on every layer. A CPU attn_post_norm feeds FFN and
    // pulls blk.63 ffn_gate/up MUL_MAT onto host (532 splits).
    out.push_back("blk\\.[0-9]+\\.attn_norm");
    out.push_back("blk\\.[0-9]+\\.attn_post_norm");
    out.push_back("blk\\.[0-9]+\\.post_attention_norm");
    out.push_back("blk\\.[0-9]+\\.attn_q_norm");
    out.push_back("blk\\.[0-9]+\\.attn_k_norm");
    out.push_back("blk\\.[0-9]+\\.ffn_norm");
    // Collapse residual→norm→output_norm onto CUDA0. CPU output_norm
    // after CUDA0 norm-63 was the #641 0xC0000005. Tiny F32. Do not
    // GPU-override output.weight (lm_head stays host).
    out.push_back("output_norm");
    // 16 Gated Attention QKVO (layers 3,7,...,63).
    out.push_back("blk\\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\\.attn_q[^k]");
    out.push_back("blk\\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\\.attn_k");
    out.push_back("blk\\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\\.attn_v");
    out.push_back("blk\\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\\.attn_output");
    out.push_back("token_embd");
    return out;
}

uint64_t pinned_ga_weight_bytes() { return kPinnedGaWeightBytes; }

uint64_t pinned_ga_kv_bytes() { return kPinnedGaWeightBytes + kHourKvReserveBytes; }

uint64_t streamed_ffn_workspace_bytes() { return kStreamWorkspaceBytes; }

std::string format_ffn_cuda_park_line(uint32_t n_park, uint64_t bytes) {
    if (n_park > kMaxParkedFfnLayers) {
        n_park = kMaxParkedFfnLayers;
    }
    const uint32_t n_stream = ffn_stream_layers(n_park);
    char buf[384];
    std::snprintf(buf, sizeof(buf),
                  "ffn_cuda_park n=%u bytes=%llu stream=%u stream_slots=%u stream_bytes=%llu "
                  "ngl=0 kv20k_bytes=%llu ggml_rebind_q4=%d never_64=1",
                  n_park, static_cast<unsigned long long>(bytes), n_stream, kNStreamSlots,
                  static_cast<unsigned long long>(kStreamWorkspaceBytes),
                  static_cast<unsigned long long>(kHourKvReserveBytes),
                  ggml_can_rebind_q4_midgraph() ? 1 : 0);
    return buf;
}

std::string format_ffn_cuda_bind_line(uint32_t layer, bool parked) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "ffn_cuda_bind layer=%u parked=%d stream=%d workspace=cuda",
                  layer, parked ? 1 : 0, parked ? 0 : 1);
    return buf;
}

std::string format_ggml_tensor_bind_line(uint32_t layer, uint64_t packed, bool real_h2d,
                                         bool ggml_used) {
    char buf[224];
    std::snprintf(buf, sizeof(buf),
                  "ggml_tensor_bind layer=%u packed=%llu layer_MiB=%.1f real_h2d=%d "
                  "ggml_used=%d (private_cudaMalloc=0)",
                  layer, static_cast<unsigned long long>(packed),
                  static_cast<double>(packed) / (1024.0 * 1024.0), real_h2d ? 1 : 0,
                  ggml_used ? 1 : 0);
    return buf;
}

std::string format_ffn_slot_bind_line(int slot, uint64_t bytes, bool real_h2d) {
    char buf[192];
    std::snprintf(buf, sizeof(buf),
                  "ffn_slot_bind slot=%d bytes=%llu layer_MiB=%.1f real_h2d=%d "
                  "(cudaMalloc_alone=0 need_full_layer=1)",
                  slot, static_cast<unsigned long long>(bytes),
                  static_cast<double>(bytes) / (1024.0 * 1024.0), real_h2d ? 1 : 0);
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
