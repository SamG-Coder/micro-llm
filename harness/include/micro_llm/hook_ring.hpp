#pragma once

// HTR1 hook ring. C++ writer. Viewer paints. Not an MLPT.
// Little-endian. Layout matches docs/HOOK_RING.md and viewer/src/ring.js.
// Ring depth 64 (~8.7MB). Do not keep 20k records.

#include "micro_llm/types.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace micro_llm {

class TraceHooks;

inline constexpr char kHtr1Magic[4] = {'H', 'T', 'R', '1'};
inline constexpr uint32_t kHtr1Version = 1;
inline constexpr uint32_t kHtr1HeaderBytes = 16;
inline constexpr uint32_t kHtr1TopkMax = 64;
inline constexpr uint32_t kHtr1RingDepth = 64;
inline constexpr uint32_t kHtr1FlagSpecialOrHighLoss = 1u;

inline constexpr uint32_t kHtr1OffTokenIndex = 0;
inline constexpr uint32_t kHtr1OffSampledId = 4;
inline constexpr uint32_t kHtr1OffFlags = 8;
inline constexpr uint32_t kHtr1OffNTopk = 12;
inline constexpr uint32_t kHtr1OffTopk = 16;
inline constexpr uint32_t kHtr1OffFireEps = 272;
inline constexpr uint32_t kHtr1OffSpikeEps = 276;
inline constexpr uint32_t kHtr1OffFfnFired = 280;
inline constexpr uint32_t kHtr1FfnBitsetBytes = kFloorBitsetBytes;  // 139264
inline constexpr uint32_t kHtr1OffPackRel = kHtr1OffFfnFired + kHtr1FfnBitsetBytes;  // 139544
inline constexpr uint32_t kHtr1OffPackSpike = kHtr1OffPackRel + kNDeltaNetPacks * 4u;  // 139736
inline constexpr uint32_t kHtr1RecordBytes = kHtr1OffPackSpike + 8u;                  // 139744

static_assert(kHtr1FfnBitsetBytes == 139264u, "FFN bitset size");
static_assert(kHtr1OffFfnFired == 280u, "fired bitset offset");
static_assert(kHtr1OffPackRel == 139544u, "pack residual offset");
static_assert(kHtr1OffPackSpike == 139736u, "pack spike offset");
static_assert(kHtr1RecordBytes == 139744u, "HTR1 record size");

struct Htr1TokenMeta {
    uint32_t token_index = 0;
    uint32_t sampled_id = 0;
    uint32_t flags = 0;
    uint32_t n_topk = 0;
    uint32_t topk[kHtr1TopkMax] = {};
};

// Encode one this-token record from the live hooks. Call BEFORE after_logits
// (that clears the per-token fired bitset). out must be kHtr1RecordBytes.
bool encode_htr1_record(const TraceHooks& hooks, const Htr1TokenMeta& meta, uint8_t* out);

// Pack sampled/topk/flags, encode, invoke cb with one RECORD_BYTES buffer.
bool emit_htr1(const TraceHooks& hooks, uint32_t sampled, const uint32_t* topk, uint32_t k,
               bool special_or_high_loss,
               const std::function<void(const uint8_t*, size_t)>& cb);

bool decode_htr1_record(const uint8_t* rec, Htr1TokenMeta* meta, uint8_t* ffn_fired,
                        float* pack_rel, uint64_t* pack_spike);

// Optional 16-byte file header + one record. Viewer accepts either.
void write_htr1_header(uint8_t* out);

// SPSC overwrite ring. Decode thread pushes; UI reads at 60Hz.
// No mutex. No WebView / JSON / file in push().
class HookRing {
public:
    HookRing();

    // Overwrite ring. Depth 64. Returns the slot that was written.
    uint32_t push(const uint8_t* rec);
    uint32_t count() const;
    uint32_t next_slot() const;
    const uint8_t* slot(uint32_t i) const;
    const uint8_t* latest() const;
    uint32_t latest_slot() const;
    // Copy latest into out (kHtr1RecordBytes). False if empty or torn.
    bool copy_latest(uint8_t* out) const;
    uint32_t seq() const { return seq_.load(std::memory_order_acquire); }

private:
    std::vector<uint8_t> buf_;
    std::atomic<uint32_t> seq_{0};
    std::atomic<uint32_t> slot_seq_[kHtr1RingDepth];
};

}  // namespace micro_llm
