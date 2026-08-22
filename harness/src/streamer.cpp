#include "micro_llm/streamer.hpp"
#include "micro_llm/ffn_reduce.hpp"
#include "micro_llm/perf_telemetry.hpp"

#include <cstdio>
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
    cuda_bind_count_ = 0;
    if (cfg_.log_cuda_ffn) {
        std::fprintf(stderr,
                     "ffn_cuda_park n=%u bytes=%llu stream_slot=%zu card_stack=%llu "
                     "soft=14GiB hard=15.2GiB never_64=1 ngl=0\n",
                     cfg_.n_parked_ffn, static_cast<unsigned long long>(parked_ffn_bytes()),
                     cfg_.ffn_scratch_bytes,
                     static_cast<unsigned long long>(card_stack_bytes()));
    }
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
    if (ffn_is_parked(layer)) {
        return true;  // already on CUDA; no host scratch
    }
    PerfTelemetry::thread_local_instance().add_prefetch();
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
    ++cuda_bind_count_;
    PerfTelemetry::thread_local_instance().add_bind();
    const bool parked = ffn_is_parked(layer);
    if (cfg_.log_cuda_ffn && cuda_bind_count_ <= cfg_.n_layers) {
        std::fprintf(stderr, "ffn_cuda_bind layer=%u parked=%d stream=%d workspace=%zu\n", layer,
                     parked ? 1 : 0, parked ? 0 : 1, cfg_.ffn_scratch_bytes);
    }
    if (parked) {
        return true;
    }
    if (compute_layer_ != ~0u && compute_layer_ != layer) {
        evict_ffn(compute_layer_);
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
    PerfTelemetry::thread_local_instance().add_evict();
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
