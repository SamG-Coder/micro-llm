#include "micro_llm/residency.hpp"
#include "micro_llm/gguf_meta.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace micro_llm {
namespace {

int32_t parse_blk_layer(const std::string& name) {
    const auto pos = name.find("blk.");
    if (pos == std::string::npos) {
        return -1;
    }
    size_t i = pos + 4;
    if (i >= name.size() || name[i] < '0' || name[i] > '9') {
        return -1;
    }
    int32_t v = 0;
    for (; i < name.size() && name[i] >= '0' && name[i] <= '9'; ++i) {
        v = v * 10 + (name[i] - '0');
    }
    if (i >= name.size() || name[i] != '.') {
        return -1;
    }
    return v;
}

uint64_t type_block_bytes(uint32_t t) {
    // ggml type ids. Enough for Qwen UD-Q4_K_M + common neighbors.
    switch (t) {
        case 0:
            return 4;  // F32
        case 1:
            return 2;  // F16
        case 2:
            return 18;  // Q4_0  (32 vals / 18B)
        case 3:
            return 20;  // Q4_1
        case 6:
            return 22;  // Q5_0
        case 7:
            return 24;  // Q5_1
        case 8:
            return 34;  // Q8_0
        case 9:
            return 36;  // Q8_1
        case 10:
            return 84;  // Q2_K  (256 vals)
        case 11:
            return 110;  // Q3_K
        case 12:
            return 144;  // Q4_K
        case 13:
            return 176;  // Q5_K
        case 14:
            return 210;  // Q6_K
        case 15:
            return 256;  // Q8_K
        case 24:
            return 2;  // BF16
        case 30:
            return 1;  // F8
        default:
            return 2;
    }
}

uint64_t type_block_elems(uint32_t t) {
    switch (t) {
        case 0:
        case 1:
        case 24:
        case 30:
            return 1;
        case 2:
        case 3:
        case 6:
        case 7:
        case 8:
        case 9:
            return 32;
        default:
            return 256;  // K-quants
    }
}

}  // namespace

int32_t layer_from_tensor_name(const std::string& name) { return parse_blk_layer(name); }

uint64_t gguf_type_nbytes(uint32_t ggml_type, uint64_t n_elems) {
    const uint64_t be = type_block_elems(ggml_type);
    const uint64_t bb = type_block_bytes(ggml_type);
    if (be == 1) {
        return n_elems * bb;
    }
    const uint64_t blocks = (n_elems + be - 1) / be;
    return blocks * bb;
}

TensorClass classify_weight_tensor(const std::string& name, int32_t* layer_out) {
    const int32_t layer = parse_blk_layer(name);
    if (layer_out) {
        *layer_out = layer;
    }
    auto has = [&](const char* s) { return name.find(s) != std::string::npos; };
    if (name.rfind("v.", 0) == 0 || name.rfind("mm.", 0) == 0 || has("vision") || has("clip.")) {
        return TensorClass::Vision;
    }
    if (has("nextn") || has("shared_head")) {
        return TensorClass::Mtp;
    }
    if (name == "token_embd.weight" || name == "token_embd" || has("tok_embeddings")) {
        return TensorClass::Embed;
    }
    if (name == "output.weight" || name == "output_norm.weight" || has("lm_head")) {
        return name.find("norm") != std::string::npos ? TensorClass::Norm : TensorClass::LmHead;
    }
    if (layer >= 0 && (has(".ffn_gate") || has(".ffn_up") || has(".ffn_down"))) {
        return TensorClass::Ffn;
    }
    if (layer >= 0 && is_gated_attention_layer(static_cast<uint32_t>(layer)) &&
        (has(".attn_q") || has(".attn_k") || has(".attn_v") || has(".attn_output") ||
         has(".attn_norm")) &&
        !has("attn_qkv") && !has("attn_gate")) {
        return has("norm") ? TensorClass::Norm : TensorClass::GaQkvO;
    }
    if (layer >= 0 && (has(".ssm_") || has(".attn_qkv") || has(".attn_gate") || has(".attn_alpha") ||
                       has(".attn_beta") || has("delta"))) {
        return TensorClass::DeltaNet;
    }
    if (has("norm")) {
        return TensorClass::Norm;
    }
    return TensorClass::Other;
}

WeightCatalog default_qwen27b_q4km_catalog() {
    WeightCatalog c;
    c.from_gguf_header = false;
    c.source = "default-qwen27b-ud-q4_k_m";
    for (uint32_t i = 0; i < kNLayers; ++i) {
        c.ffn_layer_bytes[i] = kQ4FfnLayerBytes;
    }
    c.n_ffn_layers_seen = kNLayers;
    c.ga_bytes = kPinnedGaWeightBytes;
    c.deltanet_bytes = (41ull * kGiB) / 10ull;  // 4.1 GiB
    c.embed_bytes = (7ull * kGiB) / 10ull;      // 0.7 GiB Q4
    c.lm_head_bytes = (7ull * kGiB) / 10ull;
    c.norm_bytes = 8ull * 1024ull * 1024ull;
    c.total_bytes = 0;
    for (uint32_t i = 0; i < kNLayers; ++i) {
        c.total_bytes += c.ffn_layer_bytes[i];
    }
    c.total_bytes += c.ga_bytes + c.deltanet_bytes + c.embed_bytes + c.lm_head_bytes + c.norm_bytes;
    return c;
}

bool read_gguf_weight_catalog(const std::string& path, WeightCatalog& out, std::string* err) {
    out = WeightCatalog{};
    GgufKv meta;
    if (!read_gguf_meta(path, meta, err)) {
        return false;
    }
    std::vector<GgufTensorDirRow> rows;
    if (!read_gguf_tensor_dir(path, rows, err)) {
        return false;
    }
    out.from_gguf_header = true;
    out.source = path;
    for (const auto& raw : rows) {
        int32_t layer = -1;
        const TensorClass cls = classify_weight_tensor(raw.name, &layer);
        GgufTensorRow r;
        r.name = raw.name;
        r.nbytes = raw.nbytes;
        r.n_elems = static_cast<uint32_t>(raw.n_elems > UINT32_MAX ? UINT32_MAX : raw.n_elems);
        r.layer = layer;
        r.cls = cls;
        out.total_bytes += r.nbytes;
        switch (r.cls) {
            case TensorClass::Ffn:
                if (r.layer >= 0 && static_cast<uint32_t>(r.layer) < kNLayers) {
                    out.ffn_layer_bytes[static_cast<size_t>(r.layer)] += r.nbytes;
                }
                break;
            case TensorClass::GaQkvO:
                out.ga_bytes += r.nbytes;
                break;
            case TensorClass::DeltaNet:
                out.deltanet_bytes += r.nbytes;
                break;
            case TensorClass::Embed:
                out.embed_bytes += r.nbytes;
                break;
            case TensorClass::LmHead:
                out.lm_head_bytes += r.nbytes;
                break;
            case TensorClass::Norm:
                out.norm_bytes += r.nbytes;
                break;
            case TensorClass::Mtp:
                out.mtp_bytes += r.nbytes;
                break;
            case TensorClass::Vision:
                out.vision_bytes += r.nbytes;
                break;
            case TensorClass::Other:
                out.other_bytes += r.nbytes;
                break;
        }
    }
    for (uint32_t i = 0; i < kNLayers; ++i) {
        if (out.ffn_layer_bytes[i] != 0) {
            ++out.n_ffn_layers_seen;
        }
    }
    return true;
}

namespace {

double ms_for_bytes(uint64_t bytes, double gbps) {
    if (gbps <= 0.0) {
        return 0.0;
    }
    return (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) / gbps * 1000.0;
}

std::string parked_alt(const ResidencyPlan& plan) {
    std::ostringstream os;
    os << '(';
    bool first = true;
    for (uint32_t i = 0; i < kNLayers; ++i) {
        if (!plan.ffn_parked[i]) {
            continue;
        }
        if (!first) {
            os << '|';
        }
        first = false;
        os << i;
    }
    os << ')';
    return os.str();
}

std::string streamed_alt(const ResidencyPlan& plan) {
    std::ostringstream os;
    os << '(';
    bool first = true;
    for (uint32_t i = 0; i < kNLayers; ++i) {
        if (plan.ffn_parked[i]) {
            continue;
        }
        if (!first) {
            os << '|';
        }
        first = false;
        os << i;
    }
    os << ')';
    return os.str();
}

}  // namespace

ResidencyPlan plan_residency(const WeightCatalog& cat, uint32_t n_ctx, bool force_host_deltanet,
                             bool use_quant_kv) {
    ResidencyPlan p;
    p.catalog = cat;
    p.n_ctx = n_ctx == 0 ? kDefaultServeCtx : n_ctx;
    p.n_gpu_layers = 0;
    p.pin_ga = true;
    p.pin_deltanet = !force_host_deltanet;
    p.pin_embed = false;
    p.pin_lm_head = false;
    p.pin_norm = true;
    p.use_quant_kv = use_quant_kv;
    p.disable_op_offload = true;
    p.scratch_bytes = kCudaScratchBytes;
    p.stream_slot_bytes = kQ4FfnLayerBytes;
    // Reserve 20k KV (or n_ctx if larger) BEFORE parking under 14 GiB.
    // If nvidia + KV would break 14, cut park, never this reserve.
    p.kv_reserve_tokens = kv_reserve_tokens_for_ctx(p.n_ctx);
    p.kv_bytes = hour_kv_reserve_bytes(p.n_ctx, p.use_quant_kv);
    p.deltanet_must_be_cuda = true;

    uint64_t ffn_bytes = 0;
    for (uint32_t i = 0; i < kNLayers; ++i) {
        ffn_bytes += cat.ffn_layer_bytes[i] ? cat.ffn_layer_bytes[i] : kQ4FfnLayerBytes;
    }

    // Must-pin: GA + DeltaNet + norms + KV + scratch + one stream slot.
    // Embed and lm_head stay CUDA_Host / CPU. MTP off the card.
    uint64_t must = cat.ga_bytes + (p.pin_deltanet ? cat.deltanet_bytes : 0) + cat.norm_bytes +
                    p.kv_bytes + p.scratch_bytes + p.stream_slot_bytes;
    const uint64_t card = kHourCardSoftBytes;
    uint64_t room = must < card ? card - must : 0;

    // Park early FFN so the last layers stream: prefetch n+1 during parked compute.
    uint32_t n_park = 0;
    uint64_t parked_bytes = 0;
    for (uint32_t i = 0; i < kNLayers - 1u; ++i) {
        const uint64_t layer_b =
            cat.ffn_layer_bytes[i] ? cat.ffn_layer_bytes[i] : static_cast<uint64_t>(kQ4FfnLayerBytes);
        if (parked_bytes + layer_b > room) {
            break;
        }
        p.ffn_parked[i] = true;
        parked_bytes += layer_b;
        ++n_park;
    }
    p.n_parked_ffn = n_park;
    p.n_streamed_ffn = kNLayers - n_park;

    uint64_t streamed_bytes = 0;
    for (uint32_t i = 0; i < kNLayers; ++i) {
        const uint64_t layer_b =
            cat.ffn_layer_bytes[i] ? cat.ffn_layer_bytes[i] : static_cast<uint64_t>(kQ4FfnLayerBytes);
        if (!p.ffn_parked[i]) {
            streamed_bytes += layer_b;
        }
    }

    p.vram_weight_bytes = cat.ga_bytes + (p.pin_deltanet ? cat.deltanet_bytes : 0) + cat.norm_bytes +
                          parked_bytes;
    p.cuda_host_bytes = streamed_bytes + cat.embed_bytes;
    p.cpu_bytes = cat.lm_head_bytes + cat.mtp_bytes + (p.pin_deltanet ? 0 : cat.deltanet_bytes);
    p.card_stack_bytes =
        p.vram_weight_bytes + p.scratch_bytes + p.kv_bytes + p.stream_slot_bytes;
    p.pcie_bytes_per_token = streamed_bytes;

    p.enable_flash_attn = p.pin_deltanet && p.n_streamed_ffn < kNLayers;
    // FA + CPU FFN AVed. If any FFN is CUDA_Host the op still runs on CUDA,
    // so FA is allowed. Kill FA only when DeltaNet is forced host.
    if (force_host_deltanet) {
        p.enable_flash_attn = false;
    }

    p.est_h2d_ms_pcie5 = ms_for_bytes(p.pcie_bytes_per_token, kPcie5GBps);
    p.est_h2d_ms_pcie4 = ms_for_bytes(p.pcie_bytes_per_token, kPcie4GBps);

    const double parked_compute = static_cast<double>(p.n_parked_ffn) * kGpuQ4FfnMs;
    const double stream_compute = static_cast<double>(p.n_streamed_ffn) * kGpuQ4FfnMs;
    const double delta_ms =
        p.pin_deltanet ? 48.0 * kGpuDeltaNetMs : 48.0 * kCpuDeltaNetMs;
    const double ga_ms = 16.0 * kGpuGatedAttnMs;
    // Critical path: parked compute (prefetch hides 2 * layer H2D) then streamed H2D.
    const double hidden_h2d = std::min(p.est_h2d_ms_pcie5, 2.0 * kQ4FfnLayerBytes /
                                                               (kPcie5GBps * 1024.0 * 1024.0) *
                                                               1000.0);
    (void)hidden_h2d;
    p.est_compute_ms = parked_compute + stream_compute + delta_ms + ga_ms;
    // Streamed weight traffic is the PCIe tax; compute of those layers overlaps poorly
    // with their own H2D, so wall ~= parked_compute + max(stream H2D, stream compute)
    // + leftover attention not hidden in FFN.
    const double stream_wall5 = std::max(p.est_h2d_ms_pcie5, stream_compute);
    const double stream_wall4 = std::max(p.est_h2d_ms_pcie4, stream_compute);
    const double attn_left = delta_ms + ga_ms;  // in-layer with FFN; ~30% exposed
    p.est_tok_s_pcie5 =
        1000.0 / std::max(1.0, parked_compute + stream_wall5 + attn_left * 0.3);
    p.est_tok_s_pcie4 =
        1000.0 / std::max(1.0, parked_compute + stream_wall4 + attn_left * 0.3);
    p.twenty_tok_s_possible_pcie5 = p.est_tok_s_pcie5 >= 20.0 && p.pin_deltanet;
    p.twenty_tok_s_possible_pcie4 = p.est_tok_s_pcie4 >= 20.0 && p.pin_deltanet;

    std::ostringstream why;
    why << "pin GA+" << (p.pin_deltanet ? "DeltaNet" : "NO-DeltaNet") << " VRAM, park "
        << p.n_parked_ffn << " FFN, stream " << p.n_streamed_ffn
        << " FFN on CUDA_Host (not host ggml). ngl=0. never_64=1. "
        << "pcie_bytes/token=" << p.pcie_bytes_per_token
        << " card=" << p.card_stack_bytes;
    if (!p.pin_deltanet) {
        why << " HOST DELTANET: 20 tok/s impossible (48*3.2ms=154ms floor).";
    }
    p.why = why.str();
    (void)ffn_bytes;
    return p;
}

std::vector<std::string> residency_vram_regexes(const ResidencyPlan& plan) {
    std::vector<std::string> out;
    if (plan.n_parked_ffn > 0) {
        const std::string alt = parked_alt(plan);
        if (alt.size() > 2) {
            out.push_back("blk\\." + alt + "\\.ffn_gate");
            out.push_back("blk\\." + alt + "\\.ffn_up");
            out.push_back("blk\\." + alt + "\\.ffn_down");
        }
    }
    // 16 Gated Attention QKVO. Trailing class so attn_q does not eat attn_qkv.
    out.push_back("blk\\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\\.attn_q[^k]");
    out.push_back("blk\\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\\.attn_k");
    out.push_back("blk\\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\\.attn_v");
    out.push_back("blk\\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\\.attn_output");
    out.push_back("blk\\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\\.attn_norm");
    if (plan.pin_deltanet) {
        out.push_back("blk\\.[0-9]+\\.ssm_");
        out.push_back("blk\\.[0-9]+\\.attn_qkv");
        out.push_back("blk\\.[0-9]+\\.attn_gate");
        out.push_back("blk\\.[0-9]+\\.attn_alpha");
        out.push_back("blk\\.[0-9]+\\.attn_beta");
    }
    if (plan.pin_norm) {
        out.push_back("blk\\.[0-9]+\\.ffn_norm");
        out.push_back("blk\\.[0-9]+\\.attn_norm");
        out.push_back("output_norm");
    }
    return out;
}

std::vector<std::string> residency_cuda_host_regexes(const ResidencyPlan& plan) {
    std::vector<std::string> out;
    if (plan.n_streamed_ffn > 0) {
        const std::string alt = streamed_alt(plan);
        if (alt.size() > 2) {
            out.push_back("blk\\." + alt + "\\.ffn_gate");
            out.push_back("blk\\." + alt + "\\.ffn_up");
            out.push_back("blk\\." + alt + "\\.ffn_down");
        }
    }
    out.push_back("token_embd");
    return out;
}

std::vector<std::string> residency_cpu_regexes(const ResidencyPlan& /*plan*/) {
    // Catch-all after VRAM / CUDA_Host. MTP + lm_head stay host. First match wins
    // so parked/streamed FFN and DeltaNet must be listed first.
    return {
        "blk\\.[0-9]+\\.ffn_gate",
        "blk\\.[0-9]+\\.ffn_up",
        "blk\\.[0-9]+\\.ffn_down",
        "blk\\.[0-9]+\\.ssm_",
        "blk\\.[0-9]+\\.attn_qkv",
        "blk\\.[0-9]+\\.attn_gate",
        "nextn",
        "shared_head",
        "output\\.weight",
        "lm_head",
    };
}

std::string format_residency_plan(const ResidencyPlan& plan) {
    char buf[768];
    std::snprintf(buf, sizeof(buf),
                  "residency n_park=%u n_stream=%u ga=%llu dn=%llu embed_host=%llu "
                  "lm_head_cpu=%llu vram_w=%llu cuda_host=%llu kv=%llu kv_reserve=%u "
                  "stack=%llu pcie_B/tok=%llu flash=%d qkv=%d ngl=%d never_64=1 catalog=%s",
                  plan.n_parked_ffn, plan.n_streamed_ffn,
                  static_cast<unsigned long long>(plan.catalog.ga_bytes),
                  static_cast<unsigned long long>(plan.catalog.deltanet_bytes),
                  static_cast<unsigned long long>(plan.catalog.embed_bytes),
                  static_cast<unsigned long long>(plan.catalog.lm_head_bytes),
                  static_cast<unsigned long long>(plan.vram_weight_bytes),
                  static_cast<unsigned long long>(plan.cuda_host_bytes),
                  static_cast<unsigned long long>(plan.kv_bytes), plan.kv_reserve_tokens,
                  static_cast<unsigned long long>(plan.card_stack_bytes),
                  static_cast<unsigned long long>(plan.pcie_bytes_per_token),
                  plan.enable_flash_attn ? 1 : 0, plan.use_quant_kv ? 1 : 0, plan.n_gpu_layers,
                  plan.catalog.source.c_str());
    return buf;
}

DecodeBudget decode_budget_model(const ResidencyPlan& plan) {
    DecodeBudget b;
    // Measured 5080 hour: 3.5 tok/s with FFN+DeltaNet on host ggml (ngl=99 + CPU ov).
    b.host_ms = 1000.0 / kHostHourMeasuredTokS;
    // Park-41 grew CUDA0 to ~6.7 GiB; tok/s did not move. Same wall.
    b.park_only_ms = 1000.0 / kHostHourMeasuredTokS;
    b.ffn_cuda_delta_host_ms =
        64.0 * kGpuQ4FfnMs + 48.0 * kCpuDeltaNetMs + 16.0 * kGpuGatedAttnMs;
    b.pcie_bytes_per_token = static_cast<double>(plan.pcie_bytes_per_token);
    b.planned_ms_pcie5 = 1000.0 / std::max(0.1, plan.est_tok_s_pcie5);
    b.planned_ms_pcie4 = 1000.0 / std::max(0.1, plan.est_tok_s_pcie4);
    b.twenty_impossible_if_deltanet_host = (48.0 * kCpuDeltaNetMs) > 50.0;
    return b;
}

}  // namespace micro_llm
