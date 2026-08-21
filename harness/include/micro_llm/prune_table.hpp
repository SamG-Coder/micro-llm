#pragma once

// Hour-end prune table: one file, load/save API for LLM Export.
// v1: stream all, score, cut after. This object is the score dump, not the cut.

#include "micro_llm/types.hpp"

#include <string>
#include <vector>

namespace micro_llm {

// On-disk header. Little-endian. Exactly 80 bytes.
// See docs/PRUNE_TABLE.md for the full dump format.
struct PruneTableFileHeader {
    static constexpr uint32_t kPackedAlign = kTensorAlign;
    static constexpr uint32_t kSize = 80;

    char magic[4];           // 'M','L','P','T'
    uint32_t version;        // 1
    uint32_t n_layers;       // 64
    uint32_t n_ffn_channels; // 17408
    uint32_t n_packs;        // 48
    uint32_t vocab_size;     // 248320
    uint32_t tensor_align;   // 256
    uint32_t header_size;    // 80
    float fire_eps;
    float spike_eps;
    uint64_t n_tokens;
    uint32_t flags;          // bit0 = floor present; bit1 = layer_hooked trailer after vocab
    uint32_t reserved[7];
};

static_assert(sizeof(PruneTableFileHeader) == PruneTableFileHeader::kSize,
              "PruneTableFileHeader must be 80 bytes");

inline constexpr uint32_t kPruneTableFlagHasFloor = 1u << 0;
// bit1: u64 layer_hooked trailer is present after the vocab bitset.
// Header stays 80 bytes. Trailer is not in the header.
inline constexpr uint32_t kPruneTableFlagLayerHooked = 1u << 1;
inline constexpr size_t kLayerHookedTrailerBytes = 8;

class PruneTable {
public:
    static constexpr uint32_t kPackedAlign = kTensorAlign;

    PruneTable();

    void reset();

    ChannelStat& channel(uint32_t layer, uint32_t ch);
    const ChannelStat& channel(uint32_t layer, uint32_t ch) const;

    PackStat& pack(uint32_t pack_id);
    const PackStat& pack(uint32_t pack_id) const;

    bool floor_keep(uint32_t layer, uint32_t ch) const;
    void set_floor_keep(uint32_t layer, uint32_t ch, bool keep);

    bool vocab_seen(uint32_t token_id) const;
    void set_vocab_seen(uint32_t token_id);

    void mark_reserved_core(uint32_t n_ids);

    const uint8_t* floor_bits() const { return floor_.data(); }
    uint8_t* floor_bits() { return floor_.data(); }
    const uint8_t* vocab_bits() const { return vocab_.data(); }
    uint8_t* vocab_bits() { return vocab_.data(); }

    static constexpr size_t floor_bytes() { return kFloorBitsetBytes; }
    static constexpr size_t vocab_bytes() { return kVocabBitsetBytes; }

    bool pack_is_dead(uint32_t pack_id) const;

    void mark_layer_hooked(uint32_t layer);
    bool layer_was_hooked(uint32_t layer) const;
    // Unwired = tokens ran but this layer's FFN hook never fired. Do not fake n_fired.
    bool layer_is_unwired(uint32_t layer) const;
    // Dead = hooked AND every channel n_fired == 0. Distinct from unwired.
    bool layer_is_dead(uint32_t layer) const;

    uint64_t layer_hooked() const { return layer_hooked_; }
    void set_layer_hooked_bits(uint64_t bits) { layer_hooked_ = bits; }

    float fire_eps = kDefaultFireEps;
    float spike_eps = kDefaultSpikeEps;
    uint64_t n_tokens = 0;
    uint32_t flags = kPruneTableFlagHasFloor;

    bool operator==(const PruneTable& o) const;
    bool operator!=(const PruneTable& o) const { return !(*this == o); }

private:
    std::vector<ChannelStat> channels_;
    std::vector<PackStat> packs_;
    std::vector<uint8_t> floor_;
    std::vector<uint8_t> vocab_;
    uint64_t layer_hooked_ = 0;
};

bool save_prune_table(const PruneTable& table, const std::string& path,
                      std::string* err = nullptr);
bool load_prune_table(PruneTable& table, const std::string& path,
                      std::string* err = nullptr);

}  // namespace micro_llm
