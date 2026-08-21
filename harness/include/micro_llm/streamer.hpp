#pragma once

// Trace-path streamer. Not a 27B engine.
//
// Pin: CUDA + 16 Gated Attention blocks + KV + DeltaNet state + embed.
// Stream FFNs with TWO scratch buffers + async prefetch of n+1.
// Host pages PINNED (required for prefetch to overlap).
// Double-buffer peak ~300MB (two Q4 FFN layers).
// Do NOT pin lm_head next to embed; lm_head only at logits.
// No mid-session shrink.

#include "micro_llm/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace micro_llm {

enum class PinTarget : uint8_t {
    CudaContext = 0,
    GatedAttentionBlocks = 1,  // all 16 QKVO + 4 KV heads
    KvCache = 2,
    DeltaNetState = 3,
    Embed = 4,
    LmHead = 5,                // only at logits, never with the resident set
};

struct StreamerConfig {
    static constexpr uint32_t kPackedAlign = kTensorAlign;
    size_t ffn_scratch_bytes = kQ4FfnLayerBytes;
    uint32_t n_layers = kNLayers;
    bool pin_host_pages = true;
};

class TraceStreamer {
public:
    static constexpr uint32_t kPackedAlign = kTensorAlign;
    static constexpr int kNScratch = 2;

    explicit TraceStreamer(StreamerConfig cfg = {});

    void begin_session();
    void end_session();
    bool session_open() const { return session_open_; }

    // Resident pin set. lm_head is refused here.
    bool pin(PinTarget t);
    bool is_pinned(PinTarget t) const;
    bool pin_resident();  // CUDA + 16 GA + KV + DeltaNet + embed

    // FFN double-buffer. prefetch n+1 into the idle scratch.
    bool prefetch_ffn(uint32_t layer);
    bool bind_ffn(uint32_t layer);
    bool evict_ffn(uint32_t layer);

    int compute_scratch() const { return compute_buf_; }
    int prefetch_scratch() const { return prefetch_buf_; }
    uint32_t compute_layer() const { return compute_layer_; }
    uint32_t prefetch_layer() const { return prefetch_layer_; }
    bool prefetch_outstanding() const { return prefetch_outstanding_; }
    bool host_pages_pinned() const { return host_pages_pinned_; }

    size_t scratch_bytes() const { return cfg_.ffn_scratch_bytes; }
    size_t peak_ffn_bytes() const { return 2u * cfg_.ffn_scratch_bytes; }
    const uint8_t* scratch(int i) const { return scratch_[i].data(); }

    // lm_head lives only around logits.
    bool enter_logits();
    bool leave_logits();
    bool lm_head_resident() const { return lm_head_resident_; }

    // v1: stream all, score, cut after.
    bool mid_session_shrink_allowed() const { return false; }
    bool try_mid_session_shrink();  // always fails, records the attempt

    bool shrink_attempted() const { return shrink_attempted_; }

    const StreamerConfig& config() const { return cfg_; }

private:
    bool ensure_scratch();
    void try_lock_host_pages();

    StreamerConfig cfg_;
    std::array<std::vector<uint8_t>, 2> scratch_;
    int compute_buf_ = 0;
    int prefetch_buf_ = 1;
    uint32_t compute_layer_ = ~0u;
    uint32_t prefetch_layer_ = ~0u;
    uint32_t resident_ = 0;
    bool session_open_ = false;
    bool prefetch_outstanding_ = false;
    bool host_pages_pinned_ = false;
    bool lm_head_resident_ = false;
    bool shrink_attempted_ = false;
};

inline constexpr uint32_t pin_bit(PinTarget t) {
    return 1u << static_cast<uint32_t>(t);
}

}  // namespace micro_llm
