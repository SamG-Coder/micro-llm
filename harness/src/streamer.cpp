#include "micro_llm/streamer.hpp"
#include "micro_llm/ffn_reduce.hpp"

#include <cstring>

#if defined(__linux__)
#include <sys/mman.h>
#endif

#if defined(MICRO_LLM_HAS_CUDA)
#include <cuda_runtime.h>
#endif

namespace micro_llm {

TraceStreamer::TraceStreamer(StreamerConfig cfg) : cfg_(cfg) {
    host_ptr_.fill(nullptr);
    host_bytes_.fill(0);
    if (cfg_.n_stream_slots < 1) {
        cfg_.n_stream_slots = 1;
    }
    if (cfg_.n_stream_slots > 2) {
        cfg_.n_stream_slots = 2;
    }
}

TraceStreamer::~TraceStreamer() { release_cuda_slots(); }

void TraceStreamer::set_n_parked_ffn(uint32_t n) {
    if (n > kMaxParkedFfnLayers) {
        n = kMaxParkedFfnLayers;
    }
    cfg_.n_parked_ffn = n;
}

uint32_t TraceStreamer::resident_stream_slots() const {
    uint32_t n = 0;
    if (compute_layer_ != ~0u && !ffn_is_parked(compute_layer_)) {
        ++n;
    }
    if (prefetch_outstanding_ && prefetch_layer_ != ~0u && prefetch_layer_ != compute_layer_ &&
        !ffn_is_parked(prefetch_layer_)) {
        ++n;
    }
    return n;
}

uint32_t TraceStreamer::next_streamed_layer(uint32_t layer) const {
    uint32_t n = layer + 1;
    while (n < cfg_.n_layers && ffn_is_parked(n)) {
        ++n;
    }
    return n < cfg_.n_layers ? n : ~0u;
}

void TraceStreamer::set_stream_host(uint32_t layer, const void* ptr, size_t bytes) {
    if (layer >= kNLayers) {
        return;
    }
    host_ptr_[layer] = ptr;
    host_bytes_[layer] = bytes;
}

bool TraceStreamer::ensure_scratch() {
    for (int i = 0; i < kNScratch; ++i) {
        if (scratch_[static_cast<size_t>(i)].size() != cfg_.ffn_scratch_bytes) {
            scratch_[static_cast<size_t>(i)].assign(cfg_.ffn_scratch_bytes, 0);
        }
    }
    ensure_slots();
    return true;
}

bool TraceStreamer::ensure_slots() {
    for (int i = 0; i < kNScratch; ++i) {
        if (scratch_[static_cast<size_t>(i)].size() != cfg_.ffn_scratch_bytes) {
            scratch_[static_cast<size_t>(i)].assign(cfg_.ffn_scratch_bytes, 0);
        }
    }
#if defined(MICRO_LLM_HAS_CUDA)
    if (!cuda_slots_ && cfg_.stream_compute_cuda) {
        bool ok = true;
        for (int i = 0; i < kNScratch; ++i) {
            if (d_slot_[static_cast<size_t>(i)]) {
                continue;
            }
            if (cudaMalloc(&d_slot_[static_cast<size_t>(i)], cfg_.ffn_scratch_bytes) !=
                cudaSuccess) {
                d_slot_[static_cast<size_t>(i)] = nullptr;
                ok = false;
            }
        }
        cuda_slots_ = ok;
        if (perf_) {
            perf_->set_cuda_events(ok);
        }
    }
#endif
    slots_allocated_ = true;
    return slots_allocated_;
}

bool TraceStreamer::bind_q4_into_slot(int slot, const void* host_q4, size_t bytes) {
    if (slot < 0 || slot >= kNScratch || !host_q4 || bytes == 0) {
        return false;
    }
    ensure_slots();
    const size_t copy_n = bytes < cfg_.ffn_scratch_bytes ? bytes : cfg_.ffn_scratch_bytes;
    if (perf_) {
        perf_->begin_span(PerfSpan::Pcie);
    }
#if defined(MICRO_LLM_HAS_CUDA)
    if (d_slot_[static_cast<size_t>(slot)]) {
        const cudaError_t rc =
            cudaMemcpy(d_slot_[static_cast<size_t>(slot)], host_q4, copy_n, cudaMemcpyHostToDevice);
        if (rc != cudaSuccess) {
            if (perf_) {
                perf_->end_span(PerfSpan::Pcie);
            }
            return false;
        }
    } else
#endif
    {
        auto& dst = scratch_[static_cast<size_t>(slot)];
        if (dst.size() < copy_n) {
            if (perf_) {
                perf_->end_span(PerfSpan::Pcie);
            }
            return false;
        }
        std::memcpy(dst.data(), host_q4, copy_n);
    }
    h2d_bytes_ += copy_n;
    slot_bind_bytes_ += copy_n;
    slot_bound_[static_cast<size_t>(slot)] = true;
    if (perf_) {
        perf_->add_h2d(copy_n);
        perf_->end_span(PerfSpan::Pcie);
        perf_->add_cuda_ffn_bind();
    }
    ++cuda_bind_count_;
    last_bind_kind_ = FfnComputeKind::StreamCuda;
    return true;
}

void TraceStreamer::release_cuda_slots() {
#if defined(MICRO_LLM_HAS_CUDA)
    for (int i = 0; i < kNScratch; ++i) {
        if (d_slot_[static_cast<size_t>(i)]) {
            cudaFree(d_slot_[static_cast<size_t>(i)]);
            d_slot_[static_cast<size_t>(i)] = nullptr;
        }
    }
#endif
    cuda_slots_ = false;
    slot_layer_[0] = ~0u;
    slot_layer_[1] = ~0u;
}

void TraceStreamer::try_lock_host_pages() {
    host_pages_pinned_ = false;
    if (!cfg_.pin_host_pages) {
        return;
    }
#if defined(__linux__)
    bool ok = true;
    for (int i = 0; i < kNScratch; ++i) {
        auto& buf = scratch_[static_cast<size_t>(i)];
        if (buf.empty()) {
            continue;
        }
        if (mlock(buf.data(), buf.size()) != 0) {
            ok = false;
        }
    }
    host_pages_pinned_ = ok;
#elif defined(MICRO_LLM_HAS_CUDA)
    bool ok = true;
    for (int i = 0; i < kNScratch; ++i) {
        auto& buf = scratch_[static_cast<size_t>(i)];
        if (buf.empty()) {
            continue;
        }
        if (cudaHostRegister(buf.data(), buf.size(), cudaHostRegisterDefault) != cudaSuccess) {
            ok = false;
        }
    }
    host_pages_pinned_ = ok;
#else
    host_pages_pinned_ = true;
#endif
}

bool TraceStreamer::slot_copy_h2d(int slot, uint32_t layer) {
    if (slot < 0 || slot >= kNScratch || layer >= cfg_.n_layers) {
        return false;
    }
    const size_t n = host_bytes_[layer] != 0 ? host_bytes_[layer] : cfg_.ffn_scratch_bytes;
    const size_t copy_n = n < cfg_.ffn_scratch_bytes ? n : cfg_.ffn_scratch_bytes;
    const void* src = host_ptr_[layer];
    if (!src) {
        src = scratch_[static_cast<size_t>(slot)].data();
        if (!scratch_[static_cast<size_t>(slot)].empty()) {
            scratch_[static_cast<size_t>(slot)][0] = static_cast<uint8_t>(layer & 0xffu);
        }
    }
    if (perf_) {
        perf_->begin_span(PerfSpan::Pcie);
    }
#if defined(MICRO_LLM_HAS_CUDA)
    if (d_slot_[static_cast<size_t>(slot)] && src && copy_n > 0) {
        // Async H2D. Overlap only if host pages are pinned.
        const cudaError_t rc =
            cudaMemcpyAsync(d_slot_[static_cast<size_t>(slot)], src, copy_n,
                            cudaMemcpyHostToDevice, static_cast<cudaStream_t>(0));
        if (rc != cudaSuccess) {
            cudaMemcpy(d_slot_[static_cast<size_t>(slot)], src, copy_n, cudaMemcpyHostToDevice);
        }
    }
#endif
    h2d_bytes_ += copy_n;
    if (perf_) {
        perf_->add_h2d(copy_n);
        perf_->end_span(PerfSpan::Pcie);
    }
    slot_layer_[static_cast<size_t>(slot)] = layer;
    return true;
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
    cuda_bind_count_ = 0;
    host_bind_count_ = 0;
    overlap_prefetch_count_ = 0;
    h2d_bytes_ = 0;
    slot_bind_bytes_ = 0;
    slot_bound_[0] = false;
    slot_bound_[1] = false;
    last_bind_kind_ = FfnComputeKind::HostGgml;
    slot_layer_[0] = ~0u;
    slot_layer_[1] = ~0u;
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
        auto& buf = scratch_[static_cast<size_t>(i)];
        if (!buf.empty()) {
            munlock(buf.data(), buf.size());
        }
    }
#endif
#if defined(MICRO_LLM_HAS_CUDA) && !defined(__linux__)
    for (int i = 0; i < kNScratch; ++i) {
        auto& buf = scratch_[static_cast<size_t>(i)];
        if (!buf.empty()) {
            cudaHostUnregister(buf.data());
        }
    }
#endif
}

bool TraceStreamer::pin(PinTarget t) {
    if (t == PinTarget::LmHead) {
        return false;
    }
    if (t == PinTarget::CudaContext) {
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
        return true;
    }
    ensure_scratch();
    prefetch_buf_ = 1 - compute_buf_;
    prefetch_layer_ = layer;
    prefetch_outstanding_ = true;
    slot_copy_h2d(prefetch_buf_, layer);
    if (compute_layer_ != ~0u && !ffn_is_parked(compute_layer_)) {
        ++overlap_prefetch_count_;
        if (perf_) {
            perf_->add_overlap_prefetch();
        }
    }
    return true;
}

bool TraceStreamer::bind_ffn(uint32_t layer) {
    if (!session_open_ || layer >= cfg_.n_layers) {
        return false;
    }
    if (ffn_is_parked(layer)) {
        last_bind_kind_ = FfnComputeKind::ParkedCuda;
        ++cuda_bind_count_;
        if (perf_) {
            perf_->add_cuda_ffn_bind();
        }
        return true;
    }
    if (prefetch_outstanding_ && prefetch_layer_ == layer) {
        compute_buf_ = prefetch_buf_;
        prefetch_outstanding_ = false;
    } else if (slot_layer_[static_cast<size_t>(compute_buf_)] != layer) {
        slot_copy_h2d(compute_buf_, layer);
    }
    compute_layer_ = layer;
    prefetch_buf_ = 1 - compute_buf_;
    if (cfg_.stream_compute_cuda) {
        last_bind_kind_ = FfnComputeKind::StreamCuda;
        ++cuda_bind_count_;
        if (perf_) {
            perf_->add_cuda_ffn_bind();
        }
    } else {
        last_bind_kind_ = FfnComputeKind::HostGgml;
        ++host_bind_count_;
        if (perf_) {
            perf_->add_host_ffn_bind();
        }
    }
    const uint32_t next = next_streamed_layer(layer);
    if (next != ~0u) {
        prefetch_ffn(next);
    }
    return true;
}

bool TraceStreamer::evict_ffn(uint32_t layer) {
    if (ffn_is_parked(layer)) {
        return true;
    }
    if (compute_layer_ != layer) {
        return false;
    }
    if (slot_layer_[static_cast<size_t>(compute_buf_)] == layer) {
        slot_layer_[static_cast<size_t>(compute_buf_)] = ~0u;
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
