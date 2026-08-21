#pragma once

// Qwen 27B (3.6 / 3.8) constants for the v1 trace streamer.
// Packed remnant tensors later need 256-byte alignment or 5080 kernels stall.

#include <cstddef>
#include <cstdint>

namespace micro_llm {

inline constexpr uint32_t kNLayers = 64;
inline constexpr uint32_t kFfnIntermediate = 17408;
inline constexpr uint32_t kHiddenDim = 5120;
inline constexpr uint32_t kNGroups = 16;
inline constexpr uint32_t kGroupSize = 4;
inline constexpr uint32_t kDeltaNetPerGroup = 3;
inline constexpr uint32_t kNDeltaNetPacks = 48;          // global pack id 0..47
inline constexpr uint32_t kNGatedAttnBlocks = 16;
inline constexpr uint32_t kNKvHeads = 4;
inline constexpr uint32_t kVocabSize = 248320;           // 248k, original tokenizer IDs
inline constexpr uint32_t kTensorAlign = 256;            // packed-tensor alignment
inline constexpr uint32_t kQ4FfnLayerBytes = 150u * 1024u * 1024u;
inline constexpr uint32_t kDoubleBufferPeakBytes = 300u * 1024u * 1024u;
inline constexpr uint32_t kFloorBitsetBytes =
    (kNLayers * kFfnIntermediate) / 8u;                  // 139264 ~ 140KB
inline constexpr uint32_t kVocabBitsetBytes = (kVocabSize + 7u) / 8u;
inline constexpr uint32_t kPruneTableVersion = 1;
inline constexpr char kPruneTableMagic[4] = {'M', 'L', 'P', 'T'};
inline constexpr float kDefaultFireEps = 1.0e-6f;
inline constexpr float kDefaultSpikeEps = 1.0e-6f;

// Remnant GGUF KV. C++ serve refuses unless this key is present and true.
// Export sets it false for --q4-k-to-f16 (host debug only). True on a real Q4 remnant.
inline constexpr const char kKvServeOk[] = "micro_llm.serve_ok";
inline constexpr const char kKvCudaScratchBytes[] = "micro_llm.cuda_scratch_bytes";
inline constexpr const char kKvPerTokenFp16[] = "micro_llm.kv_bytes_per_token_fp16";
inline constexpr const char kKvPerTokenFp8[] = "micro_llm.kv_bytes_per_token_fp8";
inline constexpr const char kKvServeUsableBytes[] = "micro_llm.serve_usable_bytes";
inline constexpr const char kKvWeightBytes[] = "micro_llm.weight_bytes";

// RTX 5080 serve stack. Prefer remnant file size over baked weight_bytes.
// If those KV keys are missing, use these constants.
inline constexpr uint64_t kGiB = 1024ull * 1024ull * 1024ull;
inline constexpr uint64_t kCudaScratchBytes = (9ull * kGiB) / 10ull;           // 0.9 GiB
inline constexpr uint64_t kServeUsableBytes = (152ull * kGiB) / 10ull;         // 15.2 GiB headless
inline constexpr uint64_t kServeUsableDisplayBytes = (145ull * kGiB) / 10ull;  // 14.5 GiB
inline constexpr uint64_t kKvBytesPerTokenFp16 = 65536ull;
inline constexpr uint64_t kKvBytesPerTokenFp8 = 32768ull;
inline constexpr uint64_t kDefaultServeCtx = 8192ull;

// Refuse unless serve_ok is present and true. Then refuse if
// file_size + cuda_scratch + kv_bytes_per_token * ctx > usable (15.2 GiB default).
inline constexpr bool remnant_serve_allowed(
    bool key_present,
    bool serve_ok,
    uint64_t file_size = 0,
    uint64_t ctx = kDefaultServeCtx,
    uint64_t cuda_scratch = kCudaScratchBytes,
    uint64_t kv_bytes_per_token = kKvBytesPerTokenFp16,
    uint64_t usable = kServeUsableBytes) {
    if (!key_present || !serve_ok) {
        return false;
    }
    if (kv_bytes_per_token != 0 && ctx > (~uint64_t{0} / kv_bytes_per_token)) {
        return false;
    }
    const uint64_t kv = kv_bytes_per_token * ctx;
    if (file_size > usable) {
        return false;
    }
    const uint64_t after_w = usable - file_size;
    if (cuda_scratch > after_w) {
        return false;
    }
    return kv <= (after_w - cuda_scratch);
}

// Channel is ONE index across gate, up, and down. Export gathers that same
// index from all three.
struct ChannelIndex {
    static constexpr uint32_t kPackedAlign = kTensorAlign;
    uint32_t layer = 0;
    uint32_t channel = 0;  // 0..17407, shared by gate / up / down
};

struct ChannelTriplet {
    static constexpr uint32_t kPackedAlign = kTensorAlign;
    uint32_t gate = 0;
    uint32_t up = 0;
    uint32_t down = 0;
};

inline constexpr ChannelTriplet channel_triplet(uint32_t channel) {
    return ChannelTriplet{channel, channel, channel};
}

inline constexpr bool same_channel_across_gate_up_down(const ChannelTriplet& t) {
    return t.gate == t.up && t.up == t.down;
}

// Layer pattern: 16 x (3 x DeltaNet + 1 x Gated Attention).
// Layers 3,7,11,...,63 are Gated Attention blocks (QKVO + 4 KV heads).
inline constexpr bool is_gated_attention_layer(uint32_t layer) {
    return (layer % kGroupSize) == (kGroupSize - 1u);
}

inline constexpr bool is_delta_net_layer(uint32_t layer) {
    return !is_gated_attention_layer(layer);
}

// First two and last two layers: do not DROP the layer. FFN still width-cuts.
inline constexpr bool is_layer_drop_floor(uint32_t layer) {
    return layer < 2u || layer >= (kNLayers - 2u);
}

inline constexpr bool ffn_is_scored(uint32_t /*layer*/) {
    // Including the FFN after each of the 16 Gated Attention blocks.
    return true;
}

inline constexpr bool gated_attention_block_may_drop() {
    return false;
}

// Pack id is GLOBAL 0..47. (group, slot) -> layer = 4*group+slot.
// NOT 0..2 inside a group (that is the upstream doc bug).
inline constexpr uint32_t pack_id_from_group_slot(uint32_t group, uint32_t slot) {
    return kDeltaNetPerGroup * group + slot;
}

inline constexpr uint32_t layer_from_group_slot(uint32_t group, uint32_t slot) {
    return kGroupSize * group + slot;
}

inline constexpr uint32_t pack_id_from_delta_layer(uint32_t layer) {
    const uint32_t group = layer / kGroupSize;
    const uint32_t slot = layer % kGroupSize;
    return pack_id_from_group_slot(group, slot);
}

inline constexpr uint32_t delta_layer_from_pack_id(uint32_t pack) {
    const uint32_t group = pack / kDeltaNetPerGroup;
    const uint32_t slot = pack % kDeltaNetPerGroup;
    return layer_from_group_slot(group, slot);
}

inline constexpr uint32_t group_from_pack_id(uint32_t pack) {
    return pack / kDeltaNetPerGroup;
}

inline constexpr uint32_t slot_from_pack_id(uint32_t pack) {
    return pack % kDeltaNetPerGroup;
}

inline constexpr size_t align_up(size_t n, size_t align = kTensorAlign) {
    return (n + align - 1u) / align * align;
}

struct ChannelStat {
    static constexpr uint32_t kPackedAlign = kTensorAlign;
    uint64_t n_fired = 0;
    float sumsq = 0.f;
    float maxabs = 0.f;
};

struct PackStat {
    static constexpr uint32_t kPackedAlign = kTensorAlign;
    uint32_t pack = 0;     // global 0..47
    uint32_t layer = 0;    // 4*group+slot
    uint64_t n_spike = 0;
    double sumsq_residual = 0.0;  // of |hidden_out - hidden_in|
};

inline constexpr bool pack_is_dead(const PackStat& p) {
    return p.n_spike == 0;
}

// Bitset helpers. Bit i lives in byte i>>3, bit (i&7).
inline void bit_set(uint8_t* bits, size_t i) {
    bits[i >> 3] = static_cast<uint8_t>(bits[i >> 3] | (1u << (i & 7u)));
}

inline void bit_clear(uint8_t* bits, size_t i) {
    bits[i >> 3] = static_cast<uint8_t>(bits[i >> 3] & ~(1u << (i & 7u)));
}

inline bool bit_test(const uint8_t* bits, size_t i) {
    return (bits[i >> 3] & static_cast<uint8_t>(1u << (i & 7u))) != 0;
}

inline void bit_or_into(uint8_t* dst, const uint8_t* src, size_t nbytes) {
    for (size_t i = 0; i < nbytes; ++i) {
        dst[i] = static_cast<uint8_t>(dst[i] | src[i]);
    }
}

inline constexpr size_t channel_bit_index(uint32_t layer, uint32_t channel) {
    return static_cast<size_t>(layer) * kFfnIntermediate + channel;
}

}  // namespace micro_llm
