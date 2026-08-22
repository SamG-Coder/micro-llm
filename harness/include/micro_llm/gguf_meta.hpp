#pragma once

// Minimal GGUF KV reader. Enough for serve_ok, architecture probe, and
// "do not load vision" detection. Not a tensor loader.

#include "micro_llm/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace micro_llm {

struct GgufKv {
    std::string architecture;
    uint32_t n_layers = 0;  // GGUF block_count; may be 65 (64 + MTP)
    uint32_t n_nextn = 0;   // nextn_predict_layers; 1 = extra MTP block
    uint32_t n_embd = 0;
    uint32_t n_ff = 0;
    uint32_t n_vocab = 0;
    uint64_t n_tensors = 0;
    uint64_t n_kv = 0;
    bool serve_ok_present = false;
    bool serve_ok = false;
    bool has_vision_tensors = false;
    bool has_mtp_tensors = false;
    std::vector<uint32_t> keep_channel_n;  // micro_llm.keep_channels.n
    std::string error;
    bool ok = false;
};

// Read KV + tensor names only. Does not load weights.
bool read_gguf_meta(const std::string& path, GgufKv& out, std::string* err = nullptr);

// Qwen 27B 3.6/3.8 hybrid: arch qwen35 (or qwen3next), 64 / 5120 / 17408.
// block_count=65 + nextn_predict_layers=1 is the same hybrid plus MTP.
// Hook ring and card stack use 64 layers; the extra block is not counted.
bool gguf_looks_like_qwen27b_hybrid(const GgufKv& m);
uint32_t gguf_hook_layer_count(const GgufKv& m);

// Write a KV-only GGUF (0 tensors) for tests.
bool write_gguf_kv_stub(const std::string& path, const std::string& architecture,
                        bool serve_ok_present, bool serve_ok,
                        uint32_t n_layers = kNLayers, uint32_t n_embd = kHiddenDim,
                        uint32_t n_ff = kFfnIntermediate, std::string* err = nullptr,
                        const std::vector<uint32_t>* keep_channel_n = nullptr,
                        uint32_t nextn_predict_layers = 0);

}  // namespace micro_llm
