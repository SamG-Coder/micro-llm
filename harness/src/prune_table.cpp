#include "micro_llm/prune_table.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <stdio.h>
#endif

namespace micro_llm {
namespace {

void write_le_u32(std::ostream& os, uint32_t v) {
    const unsigned char b[4] = {
        static_cast<unsigned char>(v & 0xffu),
        static_cast<unsigned char>((v >> 8) & 0xffu),
        static_cast<unsigned char>((v >> 16) & 0xffu),
        static_cast<unsigned char>((v >> 24) & 0xffu),
    };
    os.write(reinterpret_cast<const char*>(b), 4);
}

void write_le_u64(std::ostream& os, uint64_t v) {
    write_le_u32(os, static_cast<uint32_t>(v & 0xffffffffu));
    write_le_u32(os, static_cast<uint32_t>(v >> 32));
}

void write_le_f32(std::ostream& os, float v) {
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    write_le_u32(os, bits);
}

void write_le_f64(std::ostream& os, double v) {
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    write_le_u64(os, bits);
}

bool read_exact(std::istream& is, char* buf, std::streamsize n) {
    is.read(buf, n);
    return static_cast<std::streamsize>(is.gcount()) == n;
}

bool read_le_u32(std::istream& is, uint32_t& v) {
    unsigned char b[4];
    if (!read_exact(is, reinterpret_cast<char*>(b), 4)) {
        return false;
    }
    v = static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) |
        (static_cast<uint32_t>(b[2]) << 16) | (static_cast<uint32_t>(b[3]) << 24);
    return true;
}

bool read_le_u64(std::istream& is, uint64_t& v) {
    uint32_t lo = 0, hi = 0;
    if (!read_le_u32(is, lo) || !read_le_u32(is, hi)) {
        return false;
    }
    v = static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
    return true;
}

bool read_le_f32(std::istream& is, float& v) {
    uint32_t bits = 0;
    if (!read_le_u32(is, bits)) {
        return false;
    }
    std::memcpy(&v, &bits, sizeof(v));
    return true;
}

bool read_le_f64(std::istream& is, double& v) {
    uint64_t bits = 0;
    if (!read_le_u64(is, bits)) {
        return false;
    }
    std::memcpy(&v, &bits, sizeof(v));
    return true;
}

void set_err(std::string* err, const char* msg) {
    if (err) {
        *err = msg;
    }
}

}  // namespace

PruneTable::PruneTable() { reset(); }

void PruneTable::reset() {
    channels_.assign(static_cast<size_t>(kNLayers) * kFfnIntermediate, ChannelStat{});
    packs_.resize(kNDeltaNetPacks);
    for (uint32_t p = 0; p < kNDeltaNetPacks; ++p) {
        packs_[p] = PackStat{};
        packs_[p].pack = p;
        packs_[p].layer = delta_layer_from_pack_id(p);
    }
    floor_.assign(kFloorBitsetBytes, 0);
    vocab_.assign(kVocabBitsetBytes, 0);
    fire_eps = kDefaultFireEps;
    spike_eps = kDefaultSpikeEps;
    n_tokens = 0;
    flags = kPruneTableFlagHasFloor;
    layer_hooked_ = 0;
}

ChannelStat& PruneTable::channel(uint32_t layer, uint32_t ch) {
    return channels_[channel_bit_index(layer, ch)];
}

const ChannelStat& PruneTable::channel(uint32_t layer, uint32_t ch) const {
    return channels_[channel_bit_index(layer, ch)];
}

PackStat& PruneTable::pack(uint32_t pack_id) { return packs_[pack_id]; }

const PackStat& PruneTable::pack(uint32_t pack_id) const { return packs_[pack_id]; }

bool PruneTable::floor_keep(uint32_t layer, uint32_t ch) const {
    return bit_test(floor_.data(), channel_bit_index(layer, ch));
}

void PruneTable::set_floor_keep(uint32_t layer, uint32_t ch, bool keep) {
    const size_t i = channel_bit_index(layer, ch);
    if (keep) {
        bit_set(floor_.data(), i);
    } else {
        bit_clear(floor_.data(), i);
    }
}

bool PruneTable::vocab_seen(uint32_t token_id) const {
    if (token_id >= kVocabSize) {
        return false;
    }
    return bit_test(vocab_.data(), token_id);
}

void PruneTable::set_vocab_seen(uint32_t token_id) {
    if (token_id >= kVocabSize) {
        return;
    }
    bit_set(vocab_.data(), token_id);
}

void PruneTable::mark_reserved_core(uint32_t n_ids) {
    const uint32_t n = n_ids < kVocabSize ? n_ids : kVocabSize;
    for (uint32_t i = 0; i < n; ++i) {
        set_vocab_seen(i);
    }
}

bool PruneTable::pack_is_dead(uint32_t pack_id) const {
    return micro_llm::pack_is_dead(packs_[pack_id]);
}

void PruneTable::mark_layer_hooked(uint32_t layer) {
    if (layer >= kNLayers) {
        return;
    }
    layer_hooked_ |= (uint64_t{1} << layer);
}

bool PruneTable::layer_was_hooked(uint32_t layer) const {
    if (layer >= kNLayers) {
        return false;
    }
    return (layer_hooked_ & (uint64_t{1} << layer)) != 0;
}

bool PruneTable::layer_is_unwired(uint32_t layer) const {
    return n_tokens > 0 && !layer_was_hooked(layer);
}

bool PruneTable::layer_is_dead(uint32_t layer) const {
    if (!layer_was_hooked(layer)) {
        return false;
    }
    for (uint32_t c = 0; c < kFfnIntermediate; ++c) {
        if (channel(layer, c).n_fired != 0) {
            return false;
        }
    }
    return true;
}

uint32_t PruneTable::count_missing_hooks() const {
    uint32_t n = 0;
    for (uint32_t layer = 0; layer < kNLayers; ++layer) {
        if (layer_is_unwired(layer)) {
            ++n;
        }
    }
    return n;
}

bool PruneTable::operator==(const PruneTable& o) const {
    if (fire_eps != o.fire_eps || spike_eps != o.spike_eps || n_tokens != o.n_tokens ||
        flags != o.flags || layer_hooked_ != o.layer_hooked_) {
        return false;
    }
    if (channels_.size() != o.channels_.size() || packs_.size() != o.packs_.size()) {
        return false;
    }
    for (size_t i = 0; i < channels_.size(); ++i) {
        const ChannelStat& a = channels_[i];
        const ChannelStat& b = o.channels_[i];
        if (a.n_fired != b.n_fired || a.sumsq != b.sumsq || a.maxabs != b.maxabs) {
            return false;
        }
    }
    for (size_t i = 0; i < packs_.size(); ++i) {
        const PackStat& a = packs_[i];
        const PackStat& b = o.packs_[i];
        if (a.pack != b.pack || a.layer != b.layer || a.n_spike != b.n_spike ||
            a.sumsq_residual != b.sumsq_residual) {
            return false;
        }
    }
    return floor_ == o.floor_ && vocab_ == o.vocab_;
}

bool save_prune_table(const PruneTable& table, const std::string& path, std::string* err) {
    std::ofstream os(path, std::ios::binary);
    if (!os) {
        set_err(err, "failed to open prune table for write");
        return false;
    }

    os.write(kPruneTableMagic, 4);
    write_le_u32(os, kPruneTableVersion);
    write_le_u32(os, kNLayers);
    write_le_u32(os, kFfnIntermediate);
    write_le_u32(os, kNDeltaNetPacks);
    write_le_u32(os, kVocabSize);
    write_le_u32(os, kTensorAlign);
    write_le_u32(os, PruneTableFileHeader::kSize);
    write_le_f32(os, table.fire_eps);
    write_le_f32(os, table.spike_eps);
    write_le_u64(os, table.n_tokens);
    write_le_u32(os, table.flags | kPruneTableFlagLayerHooked);
    for (int i = 0; i < 7; ++i) {
        write_le_u32(os, 0);
    }

    for (uint32_t layer = 0; layer < kNLayers; ++layer) {
        for (uint32_t ch = 0; ch < kFfnIntermediate; ++ch) {
            const ChannelStat& s = table.channel(layer, ch);
            write_le_u64(os, s.n_fired);
            write_le_f32(os, s.sumsq);
            write_le_f32(os, s.maxabs);
        }
    }

    for (uint32_t p = 0; p < kNDeltaNetPacks; ++p) {
        const PackStat& s = table.pack(p);
        write_le_u32(os, s.pack);
        write_le_u32(os, s.layer);
        write_le_u64(os, s.n_spike);
        write_le_f64(os, s.sumsq_residual);
    }

    os.write(reinterpret_cast<const char*>(table.floor_bits()),
             static_cast<std::streamsize>(PruneTable::floor_bytes()));
    os.write(reinterpret_cast<const char*>(table.vocab_bits()),
             static_cast<std::streamsize>(PruneTable::vocab_bytes()));
    // Trailer after vocab. Header stays 80B.
    write_le_u64(os, table.layer_hooked());

    if (!os) {
        set_err(err, "failed to write prune table");
        return false;
    }
    return true;
}

bool load_prune_table(PruneTable& table, const std::string& path, std::string* err) {
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        set_err(err, "failed to open prune table for read");
        return false;
    }

    char magic[4];
    if (!read_exact(is, magic, 4) || std::memcmp(magic, kPruneTableMagic, 4) != 0) {
        set_err(err, "bad magic (want MLPT)");
        return false;
    }

    uint32_t version = 0, n_layers = 0, n_ch = 0, n_packs = 0, vocab = 0;
    uint32_t align = 0, header_size = 0, flags = 0;
    float fire_eps = 0.f, spike_eps = 0.f;
    uint64_t n_tokens = 0;
    if (!read_le_u32(is, version) || version != kPruneTableVersion) {
        set_err(err, "unsupported prune table version");
        return false;
    }
    if (!read_le_u32(is, n_layers) || !read_le_u32(is, n_ch) || !read_le_u32(is, n_packs) ||
        !read_le_u32(is, vocab) || !read_le_u32(is, align) || !read_le_u32(is, header_size) ||
        !read_le_f32(is, fire_eps) || !read_le_f32(is, spike_eps) || !read_le_u64(is, n_tokens) ||
        !read_le_u32(is, flags)) {
        set_err(err, "truncated header");
        return false;
    }
    for (int i = 0; i < 7; ++i) {
        uint32_t r = 0;
        if (!read_le_u32(is, r)) {
            set_err(err, "truncated header reserved");
            return false;
        }
    }

    if (n_layers != kNLayers || n_ch != kFfnIntermediate || n_packs != kNDeltaNetPacks ||
        vocab != kVocabSize || align != kTensorAlign ||
        header_size != PruneTableFileHeader::kSize) {
        set_err(err, "header dimensions do not match Qwen 27B v1 constants");
        return false;
    }

    table.reset();
    table.fire_eps = fire_eps;
    table.spike_eps = spike_eps;
    table.n_tokens = n_tokens;
    table.flags = flags;

    for (uint32_t layer = 0; layer < kNLayers; ++layer) {
        for (uint32_t ch = 0; ch < kFfnIntermediate; ++ch) {
            ChannelStat& s = table.channel(layer, ch);
            if (!read_le_u64(is, s.n_fired) || !read_le_f32(is, s.sumsq) ||
                !read_le_f32(is, s.maxabs)) {
                set_err(err, "truncated channel stats");
                return false;
            }
        }
    }

    for (uint32_t p = 0; p < kNDeltaNetPacks; ++p) {
        PackStat rec{};
        if (!read_le_u32(is, rec.pack) || !read_le_u32(is, rec.layer) ||
            !read_le_u64(is, rec.n_spike) || !read_le_f64(is, rec.sumsq_residual)) {
            set_err(err, "truncated pack stats");
            return false;
        }
        if (rec.pack != p) {
            set_err(err, "pack id is not global sequential 0..47");
            return false;
        }
        if (rec.layer != delta_layer_from_pack_id(p)) {
            set_err(err, "pack layer is not 4*group+slot");
            return false;
        }
        table.pack(p) = rec;
    }

    if (!read_exact(is, reinterpret_cast<char*>(table.floor_bits()),
                    static_cast<std::streamsize>(PruneTable::floor_bytes()))) {
        set_err(err, "truncated floor bitset");
        return false;
    }
    if (!read_exact(is, reinterpret_cast<char*>(table.vocab_bits()),
                    static_cast<std::streamsize>(PruneTable::vocab_bytes()))) {
        set_err(err, "truncated vocab bitset");
        return false;
    }

    uint64_t hooked = 0;
    if (read_le_u64(is, hooked)) {
        table.set_layer_hooked_bits(hooked);
        table.flags |= kPruneTableFlagLayerHooked;
    } else if (flags & kPruneTableFlagLayerHooked) {
        set_err(err, "truncated layer_hooked trailer");
        return false;
    } else {
        table.set_layer_hooked_bits(0);
    }
    return true;
}

bool checkpoint_prune_table(const PruneTable& table, const std::string& path, std::string* err) {
    if (path.empty()) {
        set_err(err, "checkpoint path is empty");
        return false;
    }
    const std::string tmp = path + ".tmp";
    // Scores / prune table only (~18MB MLPT). Do not pack a remnant GGUF.
    if (!save_prune_table(table, tmp, err)) {
        return false;
    }
#ifdef _WIN32
    if (!MoveFileExA(tmp.c_str(), path.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        std::remove(tmp.c_str());
        set_err(err, "atomic rename of checkpoint failed");
        return false;
    }
#else
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        set_err(err, "atomic rename of checkpoint failed");
        return false;
    }
#endif
    return true;
}

}  // namespace micro_llm
