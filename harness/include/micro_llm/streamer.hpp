#pragma once

// Trace-path streamer. Not a 27B engine.
//
// Pin: CUDA + 16 Gated Attention QKVO + KV (20k reserved) + embed.
// Park as many FFN layers as fit under 14GB AFTER KV+two stream slots.
// Stream UNPARKED FFN on CUDA: two VRAM slots, H2D of n+1 overlapped with
// compute of n, pinned host pages, then evict. Not host ggml.
// Never park all 64. ngl is not the pin (not 16, not 99). 15.2 is hard.
// DeltaNet stays CPU. MTP extra block is not in this stack.
// Do NOT pin lm_head next to embed; lm_head only at logits.
// No mid-session shrink.

#include "micro_llm/perf.hpp"
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
    LmHead = 5,  // only at logits, never with the resident set
};

enum class FfnComputeKind : uint8_t {
    ParkedCuda = 0,
    StreamCuda = 1,
    HostGgml = 2,  // 7780 trap — streamed FFN must not land here
};

struct StreamerConfig {
    static constexpr uint32_t kPackedAlign = kTensorAlign;
    size_t ffn_scratch_bytes = kQ4FfnLayerBytes;
    uint32_t n_layers = kNLayers;
    uint32_t n_parked_ffn = 0;  // layers [0, n) stay on CUDA. Hour sets this.
    uint32_t n_stream_slots = kNStreamSlots;
    bool pin_host_pages = true;
    bool log_cuda_ffn = false;
    bool stream_compute_cuda = true;  // streamed FFN must hit ggml CUDA kernels
};

class TraceStreamer {
public:
    static constexpr uint32_t kPackedAlign = kTensorAlign;
    static constexpr int kNScratch = 2;

    explicit TraceStreamer(StreamerConfig cfg = {});
    ~TraceStreamer();

    TraceStreamer(const TraceStreamer&) = delete;
    TraceStreamer& operator=(const TraceStreamer&) = delete;

    void begin_session();
    void end_session();
    bool session_open() const { return session_open_; }

    bool pin(PinTarget t);
    bool is_pinned(PinTarget t) const;
    bool pin_resident();  // CUDA + 16 GA + KV + DeltaNet state + embed

    // Prefetch n+1 into the idle VRAM slot (H2D). Overlap with compute of n.
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
    size_t peak_ffn_vram_bytes() const {
        return static_cast<size_t>(cfg_.n_stream_slots) * cfg_.ffn_scratch_bytes;
    }
    size_t ffn_vram_bytes() const {
        return static_cast<size_t>(resident_stream_slots()) * cfg_.ffn_scratch_bytes;
    }
    uint32_t resident_ffn_layers() const { return resident_stream_slots(); }
    uint32_t resident_stream_slots() const;
    uint32_t n_parked_ffn() const { return cfg_.n_parked_ffn; }
    uint32_t n_streamed_ffn() const { return ffn_stream_layers(cfg_.n_parked_ffn); }
    void set_n_parked_ffn(uint32_t n);
    bool ffn_is_parked(uint32_t layer) const { return layer < cfg_.n_parked_ffn; }
    uint32_t next_streamed_layer(uint32_t layer) const;
    uint64_t parked_ffn_bytes() const {
        return static_cast<uint64_t>(cfg_.n_parked_ffn) * kQ4FfnLayerBytes;
    }
    uint64_t cuda_bind_count() const { return cuda_bind_count_; }
    uint64_t host_bind_count() const { return host_bind_count_; }
    uint64_t overlap_prefetch_count() const { return overlap_prefetch_count_; }
    uint64_t h2d_bytes() const { return h2d_bytes_; }
    void set_log_cuda_ffn(bool on) { cfg_.log_cuda_ffn = on; }
    uint64_t card_stack_bytes() const { return hour_park_stream_card_bytes(cfg_.n_parked_ffn); }
    bool card_stack_fits() const { return hour_park_stream_fits(cfg_.n_parked_ffn); }
    const uint8_t* scratch(int i) const { return scratch_[static_cast<size_t>(i)].data(); }

    // Register host weight bytes for a streamed layer (pinned mmap / GGUF).
    void set_stream_host(uint32_t layer, const void* ptr, size_t bytes);
    FfnComputeKind last_bind_kind() const { return last_bind_kind_; }
    bool last_bind_was_cuda() const { return last_bind_kind_ != FfnComputeKind::HostGgml; }

    void set_perf(PerfClocks* perf) { perf_ = perf; }
    PerfClocks* perf() const { return perf_; }

    bool enter_logits();
    bool leave_logits();
    bool lm_head_resident() const { return lm_head_resident_; }

    bool mid_session_shrink_allowed() const { return false; }
    bool try_mid_session_shrink();
    bool shrink_attempted() const { return shrink_attempted_; }

    const StreamerConfig& config() const { return cfg_; }

private:
    bool ensure_scratch();
    void try_lock_host_pages();
    bool slot_copy_h2d(int slot, uint32_t layer);
    void release_cuda_slots();

    StreamerConfig cfg_;
    std::array<std::vector<uint8_t>, 2> scratch_;
    std::array<void*, 2> d_slot_{};
    std::array<uint32_t, 2> slot_layer_{{~0u, ~0u}};
    std::array<const void*, kNLayers> host_ptr_{};
    std::array<size_t, kNLayers> host_bytes_{};
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
    bool cuda_slots_ = false;
    uint64_t cuda_bind_count_ = 0;
    uint64_t host_bind_count_ = 0;
    uint64_t overlap_prefetch_count_ = 0;
    uint64_t h2d_bytes_ = 0;
    FfnComputeKind last_bind_kind_ = FfnComputeKind::HostGgml;
    PerfClocks* perf_ = nullptr;
};

inline constexpr uint32_t pin_bit(PinTarget t) {
    return 1u << static_cast<uint32_t>(t);
}

}  // namespace micro_llm
