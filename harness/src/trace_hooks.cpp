#include "micro_llm/trace_hooks.hpp"

#include "micro_llm/ffn_reduce.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace micro_llm {

TraceHooks::TraceHooks(float fire_eps, float spike_eps)
    : token_fired_(kFloorBitsetBytes, 0),
      abs_scratch_(kFfnIntermediate, 0.f) {
    table_.fire_eps = fire_eps;
    table_.spike_eps = spike_eps;
}

void TraceHooks::begin_token(uint32_t token_index) {
    token_index_ = token_index;
    token_open_ = true;
    std::fill(token_fired_.begin(), token_fired_.end(), 0);
    token_pack_rel_.fill(0.f);
}

bool TraceHooks::on_ffn_activations(uint32_t layer, const float* gate, const float* up,
                                    uint32_t n_channels) {
    if (!gate || !up || layer >= kNLayers || n_channels == 0 ||
        n_channels > kFfnIntermediate) {
        return false;
    }
    if (!token_open_) {
        begin_token(token_index_);
    }
    // Per-token scratch only (17408 floats). Never [chunk x 17408].
    ffn_reduce_token(gate, up, abs_scratch_.data(), nullptr, n_channels,
                     table_.fire_eps);
    const bool ok = on_ffn_abs(layer, abs_scratch_.data(), n_channels);
    if (ok) {
        table_.mark_layer_hooked(layer);
    }
    return ok;
}

bool TraceHooks::on_ffn_activations_device(uint32_t layer, const float* d_gate,
                                           const float* d_up,
                                           uint32_t n_channels) {
    if (!d_gate || !d_up || layer >= kNLayers || n_channels == 0 ||
        n_channels > kFfnIntermediate) {
        return false;
    }
    if (!token_open_) {
        begin_token(token_index_);
    }
    const int rc = ffn_reduce_token_cuda_device(d_gate, d_up, abs_scratch_.data(),
                                                nullptr, n_channels, table_.fire_eps);
    if (rc < 0) {
        return false;
    }
    const bool ok = on_ffn_abs(layer, abs_scratch_.data(), n_channels);
    if (ok) {
        table_.mark_layer_hooked(layer);
    }
    return ok;
}

bool TraceHooks::on_ffn_abs(uint32_t layer, const float* abs_act, uint32_t n_channels) {
    if (!abs_act || layer >= kNLayers || n_channels == 0 ||
        n_channels > kFfnIntermediate) {
        return false;
    }
    if (!token_open_) {
        begin_token(token_index_);
    }
    // Intentionally no floor write here. Loss is unknown until logits.
    for (uint32_t c = 0; c < n_channels; ++c) {
        const float a = abs_act[c];
        if (a > table_.fire_eps) {
            ChannelStat& s = table_.channel(layer, c);
            s.n_fired += 1;
            s.sumsq += a * a;
            if (a > s.maxabs) {
                s.maxabs = a;
            }
            bit_set(token_fired_.data(), channel_bit_index(layer, c));
        }
    }
    table_.mark_layer_hooked(layer);
    return true;
}

bool TraceHooks::on_delta_residual(uint32_t pack_id, float relative_r) {
    if (pack_id >= kNDeltaNetPacks) {
        return false;
    }
    PackStat& p = table_.pack(pack_id);
    p.pack = pack_id;
    p.layer = delta_layer_from_pack_id(pack_id);
    const double r = static_cast<double>(relative_r);
    p.sumsq_residual += r * r;
    token_pack_rel_[pack_id] = relative_r;
    if (relative_r > table_.spike_eps) {
        p.n_spike += 1;
    }
    return true;
}

bool TraceHooks::on_delta_hidden(uint32_t pack_id, const float* hidden_in,
                                 const float* hidden_out, uint32_t hidden_dim) {
    if (!hidden_in || !hidden_out || pack_id >= kNDeltaNetPacks || hidden_dim == 0) {
        return false;
    }
    double sumsq = 0.0;
    for (uint32_t i = 0; i < hidden_dim; ++i) {
        const double d = static_cast<double>(hidden_out[i]) - static_cast<double>(hidden_in[i]);
        sumsq += d * d;
    }
    const float r = relative_residual_l2(hidden_in, hidden_out, hidden_dim);
    token_pack_rel_[pack_id] = r;
    PackStat& p = table_.pack(pack_id);
    p.pack = pack_id;
    p.layer = delta_layer_from_pack_id(pack_id);
    p.sumsq_residual += sumsq;
    if (r > table_.spike_eps) {
        p.n_spike += 1;
    }
    return true;
}

bool TraceHooks::on_vocab_id(uint32_t token_id) {
    if (token_id >= kVocabSize) {
        return false;
    }
    table_.set_vocab_seen(token_id);
    return true;
}

bool TraceHooks::on_topk_ids(const uint32_t* ids, uint32_t k) {
    if (!ids) {
        return false;
    }
    bool ok = true;
    for (uint32_t i = 0; i < k; ++i) {
        ok = on_vocab_id(ids[i]) && ok;
    }
    return ok;
}

void TraceHooks::mark_reserved_core(uint32_t n_ids) { table_.mark_reserved_core(n_ids); }

void TraceHooks::after_logits(uint32_t token_index, bool special_or_high_loss) {
    token_index_ = token_index;
    if (special_or_high_loss) {
        bit_or_into(table_.floor_bits(), token_fired_.data(), kFloorBitsetBytes);
    }
    std::fill(token_fired_.begin(), token_fired_.end(), 0);
    token_open_ = false;
    table_.n_tokens += 1;
}

bool TraceHooks::token_fired(uint32_t layer, uint32_t channel) const {
    if (layer >= kNLayers || channel >= kFfnIntermediate) {
        return false;
    }
    return bit_test(token_fired_.data(), channel_bit_index(layer, channel));
}

}  // namespace micro_llm
