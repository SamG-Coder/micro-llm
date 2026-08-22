#pragma once

// Hooks a future live forward calls. This is not an inference engine.
//
// FFN: tap |SiLU(gate)*up| while the layer is still on chip, then evict.
// High-loss floor cannot be decided here ? keep a per-token fired bitset
// (~140KB) and OR it into the session floor AFTER logits if the token was
// special or high-loss.
//
// DeltaNet: residual |hidden_out - hidden_in| into global pack 0..47.
// Vocab: original tokenizer IDs (prompt, sampled, top-k logits).

#include "micro_llm/prune_table.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace micro_llm {

class TraceHooks {
public:
    static constexpr uint32_t kPackedAlign = kTensorAlign;

    explicit TraceHooks(float fire_eps = kDefaultFireEps,
                        float spike_eps = kDefaultSpikeEps);

    // Start a token. Clears the per-token fired bitset. Does not touch floor.
    void begin_token(uint32_t token_index);

    // Live FFN activations for one token, one layer. gate/up are n_channels
    // (default 17408). Internally reduces, accumulates n_fired/sumsq/maxabs,
    // sets bits in the per-token fired bitset, then the caller may evict.
    // Does NOT set the high-loss floor.
    bool on_ffn_activations(uint32_t layer, const float* gate, const float* up,
                            uint32_t n_channels = kFfnIntermediate);

    // Live device pointers. GPU accumulators (n_fired/sumsq/maxabs) +
    // this-token bitset. NO D2H of 17408 activations. Returns false if
    // CUDA is not built or the pointers are unusable.
    bool on_ffn_activations_device(uint32_t layer, const float* d_gate,
                                   const float* d_up,
                                   uint32_t n_channels = kFfnIntermediate);

    // Async D2H of the ~140KB bitset into token_fired_. Not 17408 floats.
    bool pull_token_bitset_async();
    bool sync_gpu_trace();
    // 2k checkpoint / end: copy GPU accums into the host table.
    bool pull_gpu_accums();
    bool gpu_accums_active() const { return gpu_accums_; }

    // Already-reduced |SiLU(gate)*up| for one token (no [chunk x C] scratch).
    bool on_ffn_abs(uint32_t layer, const float* abs_act,
                    uint32_t n_channels = kFfnIntermediate);

    // DeltaNet pack score already computed as relative r
    // (||out-in||_2 / (||in||_2 + 1e-12)). pack_id is GLOBAL 0..47.
    bool on_delta_residual(uint32_t pack_id, float relative_r);

    // Vector form. Spike when r = ||out-in||_2 / (||in||_2 + 1e-12) > spike_eps.
    // Identity (out==in) must not spike. sumsq_residual accumulates ||out-in||^2.
    bool on_delta_hidden(uint32_t pack_id, const float* hidden_in,
                         const float* hidden_out, uint32_t hidden_dim);

    // Vocab: original tokenizer IDs.
    bool on_vocab_id(uint32_t token_id);
    bool on_topk_ids(const uint32_t* ids, uint32_t k);
    void mark_reserved_core(uint32_t n_ids = 256);

    // AFTER logits: if special or high-loss, OR the per-token fired bitset
    // into the session floor. Then clear the per-token bitset.
    void after_logits(uint32_t token_index, bool special_or_high_loss);

    const PruneTable& table() const { return table_; }
    PruneTable& table() { return table_; }

    const uint8_t* token_fired_bits() const { return token_fired_.data(); }
    const float* token_pack_rel() const { return token_pack_rel_.data(); }
    bool token_fired(uint32_t layer, uint32_t channel) const;
    uint32_t current_token() const { return token_index_; }
    bool token_open() const { return token_open_; }

private:
    PruneTable table_;
    std::vector<uint8_t> token_fired_;
    std::array<float, kNDeltaNetPacks> token_pack_rel_{};
    std::vector<float> abs_scratch_;  // one token, 17408 floats — not [chunk x C]
    uint32_t token_index_ = 0;
    bool token_open_ = false;
    bool gpu_accums_ = false;
};

}  // namespace micro_llm
