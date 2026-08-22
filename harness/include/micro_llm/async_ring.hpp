#pragma once

// Lock-free SPSC ring for HTR1 records. Decode thread is the only producer.
// UI / background consumer drains at 30-60 Hz. Overwrite-oldest, depth 64.
// No mutex, no JSON, no WebView, no file I/O in the token path.

#include "micro_llm/hook_ring.hpp"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>

namespace micro_llm {

class AsyncHtr1Ring {
public:
    static constexpr uint32_t kDepth = kHtr1RingDepth;

    bool try_push(const uint8_t* rec) {
        if (!rec) {
            return false;
        }
        const uint32_t w = write_.load(std::memory_order_relaxed);
        const uint32_t r = read_.load(std::memory_order_acquire);
        uint32_t next = w + 1u;
        if (next - r >= kDepth) {
            // Overwrite oldest. Live view only; session heat is the MLPT.
            read_.store(next - kDepth, std::memory_order_release);
        }
        std::memcpy(slot(w), rec, kHtr1RecordBytes);
        write_.store(next, std::memory_order_release);
        drops_ = (next - r >= kDepth) ? drops_ + 1u : drops_;
        ++pushed_;
        return true;
    }

    bool try_pop(uint8_t* out) {
        const uint32_t r = read_.load(std::memory_order_relaxed);
        const uint32_t w = write_.load(std::memory_order_acquire);
        if (r == w) {
            return false;
        }
        if (out) {
            std::memcpy(out, slot(r), kHtr1RecordBytes);
        }
        read_.store(r + 1u, std::memory_order_release);
        ++popped_;
        return true;
    }

    uint32_t drain(uint8_t* out_recs, uint32_t max_recs) {
        uint32_t n = 0;
        while (n < max_recs && try_pop(out_recs + static_cast<size_t>(n) * kHtr1RecordBytes)) {
            ++n;
        }
        return n;
    }

    uint32_t size() const {
        const uint32_t w = write_.load(std::memory_order_acquire);
        const uint32_t r = read_.load(std::memory_order_acquire);
        return w - r;
    }

    uint64_t pushed() const { return pushed_; }
    uint64_t popped() const { return popped_; }
    uint64_t drops() const { return drops_; }

    AsyncHtr1Ring() : buf_(static_cast<size_t>(kDepth) * kHtr1RecordBytes, 0) {}

    static AsyncHtr1Ring& live() {
        static AsyncHtr1Ring ring;
        return ring;
    }

private:
    uint8_t* slot(uint32_t seq) { return buf_.data() + (static_cast<size_t>(seq % kDepth) * kHtr1RecordBytes); }
    const uint8_t* slot(uint32_t seq) const {
        return buf_.data() + (static_cast<size_t>(seq % kDepth) * kHtr1RecordBytes);
    }

    alignas(64) std::atomic<uint32_t> write_{0};
    alignas(64) std::atomic<uint32_t> read_{0};
    std::vector<uint8_t> buf_;
    uint64_t pushed_ = 0;
    uint64_t popped_ = 0;
    uint64_t drops_ = 0;
};

// Live HUD numbers. Decode writes atomics; UI reads at 30-60 Hz.
struct LiveStatsAtomics {
    std::atomic<uint32_t> n_tokens{0};
    std::atomic<uint32_t> tok_s_milli{0};  // tokens/s * 1000
    std::atomic<uint32_t> n_parked_ffn{0};
    std::atomic<uint32_t> n_streamed_ffn{0};

    static LiveStatsAtomics& live() {
        static LiveStatsAtomics s;
        return s;
    }
};

inline constexpr uint32_t kUiDrainHzMin = 30;
inline constexpr uint32_t kUiDrainHzMax = 60;
inline constexpr uint32_t kUiDrainPeriodMs = 16;  // ~62 Hz, inside 30-60

}  // namespace micro_llm
