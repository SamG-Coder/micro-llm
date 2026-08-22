#pragma once

// Intelligent TRACE residency. Header-only catalog of the GGUF (no weight load).
// Pin hot tensors in VRAM. Stream only the FFN layers that do not fit.
// DeltaNet is CUDA — host DeltaNet makes >=20 tok/s impossible (see model).

#include "micro_llm/types.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace micro_llm {

enum class TensorHome : uint8_t { Vram = 0, CudaHost = 1, Cpu = 2 };

enum class TensorClass : uint8_t {
    Ffn = 0,
    GaQkvO = 1,
    DeltaNet = 2,
    Embed = 3,
    LmHead = 4,
    Norm = 5,
    Mtp = 6,
    Vision = 7,
    Other = 8,
};

struct GgufTensorRow {
    std::string name;
    uint64_t nbytes = 0;
    uint32_t n_elems = 0;
    int32_t layer = -1;
    TensorClass cls = TensorClass::Other;
};

struct WeightCatalog {
    std::array<uint64_t, kNLayers> ffn_layer_bytes{};
    uint64_t ga_bytes = 0;
    uint64_t deltanet_bytes = 0;
    uint64_t embed_bytes = 0;
    uint64_t lm_head_bytes = 0;
    uint64_t norm_bytes = 0;
    uint64_t mtp_bytes = 0;
    uint64_t vision_bytes = 0;
    uint64_t other_bytes = 0;
    uint64_t total_bytes = 0;
    uint32_t n_ffn_layers_seen = 0;
    bool from_gguf_header = false;
    std::string source = "default-qwen27b-ud-q4_k_m";
};

struct ResidencyPlan {
    WeightCatalog catalog;
    std::array<bool, kNLayers> ffn_parked{};
    uint32_t n_parked_ffn = 0;
    uint32_t n_streamed_ffn = 0;
    bool pin_ga = true;
    bool pin_deltanet = true;  // proven: host DeltaNet caps the hour well below 20 tok/s
    bool pin_embed = false;    // CUDA_Host lookup
    bool pin_lm_head = false;  // logits only
    bool pin_norm = true;
    bool enable_flash_attn = true;  // safe only when FFN + DeltaNet are not CPU
    bool use_quant_kv = true;
    bool disable_op_offload = true;
    int32_t n_gpu_layers = 0;  // never 16, never 99
    uint64_t vram_weight_bytes = 0;
    uint64_t cuda_host_bytes = 0;
    uint64_t cpu_bytes = 0;
    uint64_t kv_bytes = 0;
    uint64_t scratch_bytes = kCudaScratchBytes;
    uint64_t stream_slot_bytes = kQ4FfnLayerBytes;
    uint64_t card_stack_bytes = 0;
    uint64_t pcie_bytes_per_token = 0;
    uint32_t n_ctx = kDefaultServeCtx;
    double est_h2d_ms_pcie5 = 0;
    double est_h2d_ms_pcie4 = 0;
    double est_compute_ms = 0;
    double est_tok_s_pcie5 = 0;
    double est_tok_s_pcie4 = 0;
    double host_hour_tok_s = kHostHourMeasuredTokS;
    bool twenty_tok_s_possible_pcie5 = false;
    bool twenty_tok_s_possible_pcie4 = false;
    bool deltanet_must_be_cuda = true;
    std::string why;
};

// Default Qwen 27B UD-Q4_K_M sizes. Used when this VM has no 17GB GGUF.
WeightCatalog default_qwen27b_q4km_catalog();

// Read tensor *directory* only (name, dims, type). Does not map weight blobs.
bool read_gguf_weight_catalog(const std::string& path, WeightCatalog& out, std::string* err = nullptr);

TensorClass classify_weight_tensor(const std::string& name, int32_t* layer_out = nullptr);
int32_t layer_from_tensor_name(const std::string& name);
uint64_t gguf_type_nbytes(uint32_t ggml_type, uint64_t n_elems);

ResidencyPlan plan_residency(const WeightCatalog& cat, uint32_t n_ctx,
                             bool force_host_deltanet = false);

// First-match-wins llama.cpp tensor_buft_overrides.
std::vector<std::string> residency_vram_regexes(const ResidencyPlan& plan);
std::vector<std::string> residency_cuda_host_regexes(const ResidencyPlan& plan);
std::vector<std::string> residency_cpu_regexes(const ResidencyPlan& plan);

std::string format_residency_plan(const ResidencyPlan& plan);

// Closed-form decode budget. Numbers are the host-hour fit + first principles.
// Used when the 17GB GGUF is not loaded (this VM) and as the PR proof.
struct DecodeBudget {
    double host_ms = 0;          // 3.5 tok/s hour (all FFN+DeltaNet host ggml)
    double park_only_ms = 0;     // 41 FFN VRAM, rest + DeltaNet still host
    double ffn_cuda_delta_host_ms = 0;
    double planned_ms_pcie5 = 0;
    double planned_ms_pcie4 = 0;
    double pcie_bytes_per_token = 0;
    bool twenty_impossible_if_deltanet_host = true;
};

DecodeBudget decode_budget_model(const ResidencyPlan& plan);

}  // namespace micro_llm
