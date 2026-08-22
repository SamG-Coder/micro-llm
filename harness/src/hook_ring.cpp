#include "micro_llm/hook_ring.hpp"

#include "micro_llm/trace_hooks.hpp"

#include <cstring>

namespace micro_llm {
namespace {

void write_le_u32(uint8_t* p, uint32_t v) {
    p[0] = static_cast<uint8_t>(v & 0xffu);
    p[1] = static_cast<uint8_t>((v >> 8) & 0xffu);
    p[2] = static_cast<uint8_t>((v >> 16) & 0xffu);
    p[3] = static_cast<uint8_t>((v >> 24) & 0xffu);
}

void write_le_u64(uint8_t* p, uint64_t v) {
    write_le_u32(p, static_cast<uint32_t>(v & 0xffffffffu));
    write_le_u32(p + 4, static_cast<uint32_t>(v >> 32));
}

void write_le_f32(uint8_t* p, float v) {
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    write_le_u32(p, bits);
}

uint32_t read_le_u32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t read_le_u64(const uint8_t* p) {
    return static_cast<uint64_t>(read_le_u32(p)) |
           (static_cast<uint64_t>(read_le_u32(p + 4)) << 32);
}

float read_le_f32(const uint8_t* p) {
    const uint32_t bits = read_le_u32(p);
    float v = 0.f;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

}  // namespace

void write_htr1_header(uint8_t* out) {
    if (!out) {
        return;
    }
    std::memcpy(out, kHtr1Magic, 4);
    write_le_u32(out + 4, kHtr1Version);
    write_le_u32(out + 8, kNLayers);
    write_le_u32(out + 12, kFfnIntermediate);
}

bool emit_htr1(const TraceHooks& hooks, uint32_t sampled, const uint32_t* topk, uint32_t k,
               bool special_or_high_loss,
               const std::function<void(const uint8_t*, size_t)>& cb) {
    if (!cb) {
        return false;
    }
    Htr1TokenMeta meta;
    meta.token_index = hooks.current_token();
    meta.sampled_id = sampled;
    meta.flags = special_or_high_loss ? kHtr1FlagSpecialOrHighLoss : 0u;
    meta.n_topk = k < kHtr1TopkMax ? k : kHtr1TopkMax;
    for (uint32_t i = 0; i < meta.n_topk && topk; ++i) {
        meta.topk[i] = topk[i];
    }
    std::vector<uint8_t> rec(kHtr1RecordBytes, 0);
    if (!encode_htr1_record(hooks, meta, rec.data())) {
        return false;
    }
    cb(rec.data(), rec.size());
    return true;
}

bool encode_htr1_record(const TraceHooks& hooks, const Htr1TokenMeta& meta, uint8_t* out) {
    if (!out) {
        return false;
    }
    std::memset(out, 0, kHtr1RecordBytes);
    write_le_u32(out + kHtr1OffTokenIndex, meta.token_index);
    write_le_u32(out + kHtr1OffSampledId, meta.sampled_id);
    write_le_u32(out + kHtr1OffFlags, meta.flags);
    const uint32_t n_topk = meta.n_topk < kHtr1TopkMax ? meta.n_topk : kHtr1TopkMax;
    write_le_u32(out + kHtr1OffNTopk, n_topk);
    for (uint32_t i = 0; i < kHtr1TopkMax; ++i) {
        write_le_u32(out + kHtr1OffTopk + i * 4u, i < n_topk ? meta.topk[i] : 0u);
    }
    write_le_f32(out + kHtr1OffFireEps, hooks.table().fire_eps);
    write_le_f32(out + kHtr1OffSpikeEps, hooks.table().spike_eps);

    const uint8_t* fired = hooks.token_fired_bits();
    if (fired) {
        std::memcpy(out + kHtr1OffFfnFired, fired, kHtr1FfnBitsetBytes);
    }

    const float* rel = hooks.token_pack_rel();
    uint64_t spike = 0;
    const float spike_eps = hooks.table().spike_eps;
    for (uint32_t p = 0; p < kNDeltaNetPacks; ++p) {
        const float r = rel ? rel[p] : 0.f;
        write_le_f32(out + kHtr1OffPackRel + p * 4u, r);
        if (r > spike_eps) {
            spike |= (uint64_t{1} << p);
        }
    }
    write_le_u64(out + kHtr1OffPackSpike, spike);
    return true;
}

bool decode_htr1_record(const uint8_t* rec, Htr1TokenMeta* meta, uint8_t* ffn_fired,
                        float* pack_rel, uint64_t* pack_spike) {
    if (!rec) {
        return false;
    }
    if (meta) {
        meta->token_index = read_le_u32(rec + kHtr1OffTokenIndex);
        meta->sampled_id = read_le_u32(rec + kHtr1OffSampledId);
        meta->flags = read_le_u32(rec + kHtr1OffFlags);
        meta->n_topk = read_le_u32(rec + kHtr1OffNTopk);
        if (meta->n_topk > kHtr1TopkMax) {
            meta->n_topk = kHtr1TopkMax;
        }
        for (uint32_t i = 0; i < kHtr1TopkMax; ++i) {
            meta->topk[i] = read_le_u32(rec + kHtr1OffTopk + i * 4u);
        }
    }
    if (ffn_fired) {
        std::memcpy(ffn_fired, rec + kHtr1OffFfnFired, kHtr1FfnBitsetBytes);
    }
    if (pack_rel) {
        for (uint32_t p = 0; p < kNDeltaNetPacks; ++p) {
            pack_rel[p] = read_le_f32(rec + kHtr1OffPackRel + p * 4u);
        }
    }
    if (pack_spike) {
        *pack_spike = read_le_u64(rec + kHtr1OffPackSpike);
    }
    return true;
}

HookRing::HookRing() : buf_(static_cast<size_t>(kHtr1RingDepth) * kHtr1RecordBytes, 0) {
    for (uint32_t i = 0; i < kHtr1RingDepth; ++i) {
        slot_seq_[i].store(0, std::memory_order_relaxed);
    }
}

uint32_t HookRing::push(const uint8_t* rec) {
    if (!rec) {
        return next_slot();
    }
    const uint32_t s = seq_.fetch_add(1, std::memory_order_relaxed);
    const uint32_t i = s % kHtr1RingDepth;
    slot_seq_[i].store(0, std::memory_order_relaxed);
    std::memcpy(buf_.data() + static_cast<size_t>(i) * kHtr1RecordBytes, rec, kHtr1RecordBytes);
    slot_seq_[i].store(s + 1u, std::memory_order_release);
    return i;
}

uint32_t HookRing::count() const {
    const uint32_t s = seq_.load(std::memory_order_acquire);
    return s < kHtr1RingDepth ? s : kHtr1RingDepth;
}

uint32_t HookRing::next_slot() const {
    return seq_.load(std::memory_order_acquire) % kHtr1RingDepth;
}

const uint8_t* HookRing::slot(uint32_t i) const {
    if (i >= kHtr1RingDepth || i >= count()) {
        return nullptr;
    }
    return buf_.data() + static_cast<size_t>(i) * kHtr1RecordBytes;
}

const uint8_t* HookRing::latest() const {
    if (count() == 0) {
        return nullptr;
    }
    return slot(latest_slot());
}

uint32_t HookRing::latest_slot() const {
    const uint32_t s = seq_.load(std::memory_order_acquire);
    return s == 0 ? 0u : (s - 1u) % kHtr1RingDepth;
}

bool HookRing::copy_latest(uint8_t* out) const {
    if (!out) {
        return false;
    }
    const uint32_t s = seq_.load(std::memory_order_acquire);
    if (s == 0) {
        return false;
    }
    const uint32_t i = (s - 1u) % kHtr1RingDepth;
    const uint32_t want = s;
    if (slot_seq_[i].load(std::memory_order_acquire) != want) {
        return false;
    }
    std::memcpy(out, buf_.data() + static_cast<size_t>(i) * kHtr1RecordBytes, kHtr1RecordBytes);
    return slot_seq_[i].load(std::memory_order_acquire) == want;
}

}  // namespace micro_llm
