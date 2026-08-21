#include "micro_llm/streamer.hpp"
#include "micro_llm/ffn_reduce.hpp"

#include <cstring>

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace micro_llm {

TraceStreamer::TraceStreamer(StreamerConfig cfg) : cfg_(cfg) {}

bool TraceStreamer::ensure_scratch() {
    for (int i = 0; i < kNScratch; ++i) {
        if (scratch_[i].size() != cfg_.ffn_scratch_bytes) {
            scratch_[i].assign(cfg_.ffn_scratch_bytes, 0);
        }
    }
    return true;
}

void TraceStreamer::try_lock_host_pages() {
    host_pages_pinned_ = false;
    if (!cfg_.pin_host_pages) {
        return;
    }
#if defined(__linux__)
    bool ok = true;
    for (int i = 0; i < kNScratch; ++i) {
        if (scratch_[i].empty()) {
            continue;
        }
        if (mlock(scratch_[i].data(), scratch_[i].size()) != 0) {
            ok = false;
        }
    }
    host_pages_pinned_ = ok;
#else
    // Non-Linux: treat the request as logical pin for the control plane.
    host_pages_pinned_ = true;
#endif
}

void TraceStreamer::begin_session() {
    session_open_ = true;
    shrink_attempted_ = false;
    lm_head_resident_ = false;
    compute_buf_ = 0;
    prefetch_buf_ = 1;
    compute_layer_ = ~0u;
    prefetch_layer_ = ~0u;
    prefetch_outstanding_ = false;
    resident_ = 0;
    ensure_scratch();
    try_lock_host_pages();
    pin_resident();
}

void TraceStreamer::end_session() {
    leave_logits();
    compute_layer_ = ~0u;
    prefetch_layer_ = ~0u;
    prefetch_outstanding_ = false;
    session_open_ = false;
#if defined(__linux__)
    for (int i = 0; i < kNScratch; ++i) {
        if (!scratch_[i].empty()) {
            munlock(scratch_[i].data(), scratch_[i].size());
        }
    }
#endif
}

bool TraceStreamer::pin(PinTarget t) {
    if (t == PinTarget::LmHead) {
        // Never pin lm_head next to embed / the resident set.
        return false;
    }
    if (t == PinTarget::CudaContext) {
        // Persistent reduce scratch for the hour. Device-tap reuses this.
        persistent_cuda_reduce().ensure(kFfnIntermediate);
    }
    resident_ |= pin_bit(t);
    return true;
}

bool TraceStreamer::is_pinned(PinTarget t) const {
    if (t == PinTarget::LmHead) {
        return lm_head_resident_;
    }
    return (resident_ & pin_bit(t)) != 0;
}

bool TraceStreamer::pin_resident() {
    bool ok = true;
    ok = pin(PinTarget::CudaContext) && ok;
    ok = pin(PinTarget::GatedAttentionBlocks) && ok;
    ok = pin(PinTarget::KvCache) && ok;
    ok = pin(PinTarget::DeltaNetState) && ok;
    ok = pin(PinTarget::Embed) && ok;
    return ok && !is_pinned(PinTarget::LmHead);
}

bool TraceStreamer::prefetch_ffn(uint32_t layer) {
    if (!session_open_ || layer >= cfg_.n_layers) {
        return false;
    }
    ensure_scratch();
    prefetch_buf_ = 1 - compute_buf_;
    prefetch_layer_ = layer;
    prefetch_outstanding_ = true;
    // Real path: cudaMemcpyAsync into scratch_[prefetch_buf_] from pinned host.
    // Overlap only if host_pages_pinned_.
    if (!scratch_[prefetch_buf_].empty()) {
        scratch_[prefetch_buf_][0] = static_cast<uint8_t>(layer & 0xffu);
    }
    return true;
}

bool TraceStreamer::bind_ffn(uint32_t layer) {
    if (!session_open_ || layer >= cfg_.n_layers) {
        return false;
    }
    if (prefetch_outstanding_ && prefetch_layer_ == layer) {
        compute_buf_ = prefetch_buf_;
        prefetch_outstanding_ = false;
    }
    compute_layer_ = layer;
    prefetch_buf_ = 1 - compute_buf_;
    return true;
}

bool TraceStreamer::evict_ffn(uint32_t layer) {
    if (compute_layer_ != layer) {
        return false;
    }
    compute_layer_ = ~0u;
    return true;
}

bool TraceStreamer::enter_logits() {
    if (!session_open_) {
        return false;
    }
    lm_head_resident_ = true;
    return true;
}

bool TraceStreamer::leave_logits() {
    lm_head_resident_ = false;
    return true;
}

bool TraceStreamer::try_mid_session_shrink() {
    shrink_attempted_ = true;
    return false;
}

}  // namespace micro_llm
