#pragma once

// Qwen 27B (3.6 / 3.8) constants for the v1 trace streamer.
// Packed remnant tensors later need 256-byte alignment or 5080 kernels stall.

#include <cmath>
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
inline constexpr uint32_t kQ4KSuperblock = 256;          // Q4_K last-dim / FFN keep width
inline constexpr uint32_t kWeakKeepMin27B = 13056;       // 25% cap; 51*256
inline constexpr uint32_t kWeakKeepMinRecover27B = 10496; // 40% recover floor; 41*256
// 10445 = ceil(17408*0.60) is NOT valid — not a Q4_K superblock multiple.
inline constexpr uint32_t kQ4FfnLayerBytes = 160u * 1024u * 1024u;
inline constexpr uint32_t kDoubleBufferPeakBytes = 340u * 1024u * 1024u;
// 160 MiB is the pre-measure floor / park-layer estimate. James 254e10c:
// gate+up bound, down off past 160. Runtime A/B = measured
// gate+up+down + align, not this floor, when the sum is larger.
inline constexpr uint64_t kQ4FfnLayerBytesMeasured5080 = 160ull * 1024ull * 1024ull;
inline constexpr uint64_t kStreamSlotBytes = 160ull * 1024ull * 1024ull;
inline constexpr uint64_t kMeasured5080Cuda0MiB = 9851ull;
inline constexpr uint64_t kMeasured5080FreeMiB = 2409ull;
inline constexpr uint32_t kMeasured5080GraphSplits = 340;
inline constexpr float kMeasured5080TokPerSecCbOff = 0.52f;
inline constexpr uint32_t kMeasured5080Park57 = 57;
inline constexpr uint32_t kMeasured5080Stream7 = 7;
// 9bcbfb7: CUDA0 13110 then reserve died at 532. 9a5f0df: 63 ffn_down
// stayed CPU_Mapped, gate/up still CPU, 638 splits, no BENCH.
// Count graph reserve BEFORE any extra park. Do not park overflow.
inline constexpr uint64_t kMeasured5080Cuda0BindMiB = 13110ull;
inline constexpr uint32_t kMeasured5080ReserveSplits = 638;
inline constexpr uint32_t kMeasured5080ReserveSplits532 = 532;
// James 974b0c3: AV after the split dump. Tail is backend_transition,
// not FFN. 62/63 A/B stayed CUDA0 in_buffer.
inline constexpr uint32_t kMeasured5080TailSplitKv = 638;
inline constexpr uint32_t kMeasured5080TailSplitResidual = 639;
inline constexpr uint32_t kMeasured5080TailSplitLayerNorm = 640;
inline constexpr uint32_t kMeasured5080TailSplitOutputNorm = 641;
inline constexpr uint32_t kMeasured5080TailSplitLmHead = 642;
inline constexpr uint64_t kHourGraphReserveBytes = 1026ull * 1024ull * 1024ull;
// Slot A/B FIRST: 160 + 160 = 320, pair budget ≈ 340 MiB with align slack.
// Then park leftover FFN under 14 GiB after KV@20k (0–56). Stream 57–63:
// H2D N+1 while N GEMMs from VRAM. Park 64 is illegal. Unused cudaMalloc
// is not a bind. ggml cannot rebind N Q4 tensors onto two mid-graph slots.
inline constexpr uint32_t kNStreamSlots = 2;
inline constexpr uint64_t kStreamSlotPairBudgetBytes = 340ull * 1024ull * 1024ull;
inline constexpr uint64_t kStreamWorkspaceBytes = kStreamSlotPairBudgetBytes;
inline constexpr uint32_t kMinStreamedFfnLayers = kMeasured5080Stream7;
inline constexpr uint32_t kTargetMaxStreamedFfnLayers = 12;
inline constexpr uint32_t kMaxParkedFfnLayers = kMeasured5080Park57;  // never 64
// Extra GGUF block (block_count=65, nextn_predict_layers=1) is MTP.
inline constexpr uint32_t kMtpExtraBlocks = 1;
inline constexpr uint32_t kFloorBitsetBytes =
    (kNLayers * kFfnIntermediate) / 8u;                  // 139264 ~ 140KB
inline constexpr uint32_t kVocabBitsetBytes = (kVocabSize + 7u) / 8u;
inline constexpr uint32_t kPruneTableVersion = 1;
inline constexpr char kPruneTableMagic[4] = {'M', 'L', 'P', 'T'};
inline constexpr float kDefaultFireEps = 1.0e-6f;
// Relative pack score: r = ||out-in||_2 / (||in||_2 + kRelResidualDenom).
// Identity (out==in) must not spike. Absolute L2-vs-1e-6 was a false-spike trap.
inline constexpr float kDefaultSpikeEps = 0.02f;
inline constexpr float kRelResidualDenom = 1.0e-12f;

// Remnant GGUF KV. C++ serve refuses unless this key is present and true.
// Export sets it false for --q4-k-to-f16 (host debug / F16 dump). True on a real Q4 remnant.
inline constexpr const char kKvServeOk[] = "micro_llm.serve_ok";
inline constexpr const char kKvKeepChannelsN[] = "micro_llm.keep_channels.n";
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
// Hour card: 14GB soft, 15.2 hard. Reserve KV to 20k BEFORE more park.
inline constexpr uint64_t kHourCardSoftBytes = 14ull * kGiB;
inline constexpr uint32_t kHourKvReserveTokens = 20000;
inline constexpr uint64_t kHourKvReserveBytes =
    static_cast<uint64_t>(kHourKvReserveTokens) * kKvBytesPerTokenFp16;  // ~1.22 GiB / ~1.3GB
// 16 GA QKVO at Q4 ~0.6 GiB. Embed ~0.7 GiB. lm_head stays host (logits only).
inline constexpr uint64_t kPinnedGaWeightBytes = (6ull * kGiB) / 10ull;
inline constexpr uint64_t kPinnedEmbedWeightBytes = (7ull * kGiB) / 10ull;
// 7780 live: ngl=99 + FFN CPU overrides. CUDA0 model 6760 MiB (GA+embed+lm_head
// + norms). tok/s 3.55 @ 192, 3.37 @ 1136. Unparked FFN ran host ggml.
inline constexpr uint64_t kMeasured7780Cuda0MiB = 6760ull;
inline constexpr uint32_t kMeasured7780Tok192 = 192;
inline constexpr uint32_t kMeasured7780Tok1136 = 1136;
inline constexpr float kMeasured7780TokPerSec192 = 3.55f;
inline constexpr float kMeasured7780TokPerSec1136 = 3.37f;
// CPU Q4 FFN decode ~4.5ms/layer matches 7780: 64*4.5ms = 288ms = 3.47 tok/s.
inline constexpr float kCpuFfnMsPerLayer7780 = 4.5f;
// PCIe 5.0 x16 practical copy. Unpinned pages serialize well below this.
inline constexpr float kPcie5PracticalGBs = 50.0f;

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

// FFN keep width must be a multiple of 256 (Q4_K superblock).
// 13056 and 10496 are valid; 10445 is not.
inline constexpr bool ffn_keep_width_q4k_ok(uint32_t n) {
    return n != 0 && (n % kQ4KSuperblock) == 0;
}

// Preferred name. True only if the KV is present AND true. False = F16 host dump, refuse.
// Packed FFN intermediate, when known (ffn_keep != 0), must be a Q4_K multiple of 256.
inline constexpr bool remnant_may_serve(bool key_present, bool serve_ok,
                                        uint32_t ffn_keep = 0) {
    if (ffn_keep != 0 && !ffn_keep_width_q4k_ok(ffn_keep)) {
        return false;
    }
    return remnant_serve_allowed(key_present, serve_ok);
}

// Fixed hour card before any parked FFN. KV 20k is reserved first.
inline constexpr uint64_t hour_fixed_card_bytes(
    uint64_t stream_workspace = kStreamWorkspaceBytes) {
    return kPinnedGaWeightBytes + kPinnedEmbedWeightBytes + kCudaScratchBytes +
           kHourKvReserveBytes + stream_workspace;
}

inline constexpr uint32_t ffn_park_layers_that_fit(
    uint64_t card = kHourCardSoftBytes,
    uint64_t stream_workspace = kStreamWorkspaceBytes,
    uint64_t layer_bytes = kQ4FfnLayerBytesMeasured5080) {
    const uint64_t need = hour_fixed_card_bytes(stream_workspace);
    if (need >= card || layer_bytes == 0) {
        return 0;
    }
    uint32_t n = static_cast<uint32_t>((card - need) / layer_bytes);
    if (n > kMaxParkedFfnLayers) {
        n = kMaxParkedFfnLayers;
    }
    return n;
}

inline constexpr uint32_t ffn_stream_layers(uint32_t n_park = ffn_park_layers_that_fit()) {
    return n_park >= kNLayers ? 0u : (kNLayers - n_park);
}

inline constexpr uint64_t hour_park_stream_card_bytes(
    uint32_t n_park, uint64_t layer_bytes = kQ4FfnLayerBytesMeasured5080) {
    return hour_fixed_card_bytes() + static_cast<uint64_t>(n_park) * layer_bytes;
}

inline constexpr bool hour_park_stream_fits(
    uint32_t n_park, uint64_t layer_bytes = kQ4FfnLayerBytesMeasured5080) {
    const uint64_t stack = hour_park_stream_card_bytes(n_park, layer_bytes);
    return n_park <= kMaxParkedFfnLayers && stack <= kHourCardSoftBytes &&
           stack <= kServeUsableBytes;
}

// 160 MiB * 64 physically fits under 14 GiB after a 340 MiB slot pair.
// That path is still illegal — stream the overflow.
inline constexpr bool parked_all_ffn_exceeds_card(
    uint32_t n_ffn_layers = kNLayers,
    uint64_t usable = kServeUsableBytes,
    uint64_t layer_bytes = kQ4FfnLayerBytes) {
    return hour_park_stream_card_bytes(n_ffn_layers, layer_bytes) > usable;
}

inline constexpr bool ffn_park_all_fits_5080_measured() {
    const uint64_t extra7 = static_cast<uint64_t>(kNLayers - kMeasured5080Park57) *
                            kQ4FfnLayerBytesMeasured5080;
    const uint64_t stack64 = hour_fixed_card_bytes() +
                             static_cast<uint64_t>(kNLayers) * kQ4FfnLayerBytesMeasured5080;
    return extra7 <= kMeasured5080FreeMiB * 1024ull * 1024ull &&
           stack64 <= kHourCardSoftBytes;
}

inline constexpr bool hour_never_park_64(uint32_t n_park = ffn_park_layers_that_fit()) {
    return n_park <= kMaxParkedFfnLayers && n_park < kNLayers &&
           ffn_stream_layers(n_park) >= kMinStreamedFfnLayers;
}

inline constexpr uint64_t stream_pcie_bytes_per_tok(
    uint32_t n_stream, uint64_t layer_bytes = kQ4FfnLayerBytesMeasured5080) {
    return static_cast<uint64_t>(n_stream) * layer_bytes;
}

// Hour path streams the overflow (57–63). park=64 / n_stream=0 is illegal.
inline constexpr bool streamed_hour_in_20_tok_band(uint32_t n_stream) {
    return n_stream >= kMinStreamedFfnLayers && n_stream <= kTargetMaxStreamedFfnLayers;
}

// Hook ring is 64. block_count=65 is 64 + MTP; do not count the extra block.
inline constexpr bool gguf_block_count_is_hybrid(uint32_t block_count) {
    return block_count == kNLayers || block_count == kNLayers + kMtpExtraBlocks;
}

inline constexpr uint32_t hook_layer_count_from_blocks(uint32_t block_count) {
    if (block_count == kNLayers + kMtpExtraBlocks) {
        return kNLayers;
    }
    return block_count;
}

// 7780: 64 CPU FFN layers * 4.5ms = 3.47 tok/s. Matches 3.55 / 3.37.
inline constexpr double cpu_ffn_tok_per_sec(double ms_per_layer = kCpuFfnMsPerLayer7780,
                                            uint32_t n_layers = kNLayers) {
    const double ms = ms_per_layer * static_cast<double>(n_layers);
    return ms > 0.0 ? 1000.0 / ms : 0.0;
}

// Streamed FFN H2D only. Parked layers are GDDR-resident (not this bound).
// 7 * 80MiB / 50 GB/s ≈ 11ms → ~89 tok/s PCIe ceiling. Those seven must
// H2D into A/B and GEMM from the CUDA buffer, not CUDA_Host (340 splits).
inline constexpr double stream_pcie_tok_per_sec(
    uint32_t n_stream,
    double gbs = kPcie5PracticalGBs,
    uint64_t layer_bytes = kQ4FfnLayerBytesMeasured5080) {
    const double gb = static_cast<double>(n_stream) * static_cast<double>(layer_bytes) / 1.0e9;
    return gb > 0.0 ? gbs / gb : 0.0;
}

// KV vs tok/s (decode is weight-bandwidth bound; do not drop ctx to fake speed):
//   4k  → 0.25 GiB KV. Same tok/s as short ctx; 16 GA layers only.
//   8k  → 0.50 GiB.
//   16k → 1.00 GiB.
//   20k → 1.22 GiB reserved before park.
//   32k → 2.00 GiB. Soft-14 needs ~5 fewer parked FFN (or spill toward 15.2).
// Measure 4k/8k/16k/32k on the 5080. Do not shrink the model.

// r = ||out-in||_2 / (||in||_2 + 1e-12). Identity => 0.
inline float relative_residual_l2(const float* hidden_in, const float* hidden_out,
                                  uint32_t hidden_dim) {
    double sumsq_d = 0.0;
    double sumsq_in = 0.0;
    for (uint32_t i = 0; i < hidden_dim; ++i) {
        const double in = static_cast<double>(hidden_in[i]);
        const double d = static_cast<double>(hidden_out[i]) - in;
        sumsq_d += d * d;
        sumsq_in += in * in;
    }
    const double num = std::sqrt(sumsq_d);
    const double den = std::sqrt(sumsq_in) + static_cast<double>(kRelResidualDenom);
    return static_cast<float>(num / den);
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
