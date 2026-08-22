#include "micro_llm/graph_hooks.hpp"

#include "micro_llm/hook_ring.hpp"

#include <cstring>

namespace micro_llm {
namespace {

bool parse_layer_suffix(std::string_view name, int* layer) {
    const auto pos = name.find_last_of("-#");
    if (pos == std::string_view::npos || pos + 1 >= name.size()) {
        return false;
    }
    int v = 0;
    bool any = false;
    for (size_t i = pos + 1; i < name.size(); ++i) {
        const char c = name[i];
        if (c < '0' || c > '9') {
            break;
        }
        any = true;
        v = v * 10 + (c - '0');
    }
    if (!any || v < 0 || v >= static_cast<int>(kNLayers)) {
        return false;
    }
    *layer = v;
    return true;
}

bool starts_with(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.substr(0, p.size()) == p;
}

}  // namespace

GraphHookSite classify_graph_tensor(std::string_view name, int* layer_out) {
    if (layer_out) {
        *layer_out = -1;
    }
    if (name == "model.input_embed" || name == "input_embed") {
        return GraphHookSite::InputEmbed;
    }
    if (name == "result_output" || starts_with(name, "result_output")) {
        return GraphHookSite::Logits;
    }

    int layer = -1;
    const bool have_layer = parse_layer_suffix(name, &layer);
    if (have_layer && layer_out) {
        *layer_out = layer;
    }

    auto stem = name;
    if (have_layer) {
        const auto pos = name.find_last_of("-#");
        stem = name.substr(0, pos);
    }

    if (stem == "ffn_gate_par" || starts_with(stem, "ffn_gate_par")) {
        return GraphHookSite::FfnGatePar;
    }
    if (stem == "ffn_gate" || stem == "ffn_gate.weight") {
        return GraphHookSite::FfnGate;
    }
    if (stem == "ffn_up" || stem == "ffn_up.weight") {
        return GraphHookSite::FfnUp;
    }
    if (stem == "attn_residual") {
        return GraphHookSite::AttnResidual;
    }
    if (stem == "l_out" || stem == "post_ffn") {
        return GraphHookSite::LayerOut;
    }
    return GraphHookSite::None;
}

GraphHookSession::GraphHookSession(TraceHooks& hooks, TraceStreamer& streamer)
    : hooks_(hooks),
      streamer_(streamer),
      hidden_in_(kHiddenDim, 0.f),
      pending_gate_(kFfnIntermediate, 0.f) {}

void GraphHookSession::begin_token(uint32_t token_index) {
    hooks_.begin_token(token_index);
    pending_gate_layer_ = -1;
}

void GraphHookSession::finish_token(uint32_t sampled, const uint32_t* topk, uint32_t k,
                                    bool special_or_high_loss) {
    hooks_.on_vocab_id(sampled);
    if (topk && k) {
        hooks_.on_topk_ids(topk, k);
    }
    if (on_htr1_) {
        emit_htr1(hooks_, sampled, topk, k, special_or_high_loss, on_htr1_);
    }
    streamer_.enter_logits();
    hooks_.after_logits(hooks_.current_token(), special_or_high_loss);
    streamer_.leave_logits();
}

bool GraphHookSession::on_tensor(const GraphTensorView& t, bool ask) {
    int layer = -1;
    const GraphHookSite site = classify_graph_tensor(t.name, &layer);
    if (site == GraphHookSite::None || site == GraphHookSite::FfnGatePar) {
        return false;
    }
    if (ask) {
        if (site == GraphHookSite::FfnGate || site == GraphHookSite::FfnUp) {
            if (layer >= 0 && static_cast<uint32_t>(layer) < kNLayers) {
                const uint32_t next = static_cast<uint32_t>(layer) + 1;
                if (next < kNLayers && !streamer_.ffn_is_parked(next)) {
                    streamer_.prefetch_ffn(next);
                }
                streamer_.bind_ffn(static_cast<uint32_t>(layer));
            }
        }
        return true;
    }
    if (!t.data || t.ne0 == 0) {
        return false;
    }

    if (site == GraphHookSite::InputEmbed || site == GraphHookSite::LayerOut) {
        const uint32_t n = t.ne0 < kHiddenDim ? t.ne0 : kHiddenDim;
        if (!t.on_device) {
            std::memcpy(hidden_in_.data(), t.data, n * sizeof(float));
        }
        return true;
    }

    if (site == GraphHookSite::FfnGate && layer >= 0) {
        const uint32_t n = t.ne0 < kFfnIntermediate ? t.ne0 : kFfnIntermediate;
        if (t.on_device) {
            pending_gate_device_ = true;
            pending_gate_layer_ = layer;
            // Keep the live pointer in pending_gate_ via a side channel: we
            // cannot store a device pointer in the host vector. Flush happens
            // on FfnUp using t.data of gate via a dedicated field.
            pending_gate_.assign(1, 0.f);  // mark device pending
            device_gate_ = t.data;
            device_gate_n_ = n;
        } else {
            pending_gate_device_ = false;
            pending_gate_.assign(kFfnIntermediate, 0.f);
            std::memcpy(pending_gate_.data(), t.data, n * sizeof(float));
            pending_gate_layer_ = layer;
            device_gate_ = nullptr;
            device_gate_n_ = n;
        }
        ++ffn_gate_hits_;
        return true;
    }

    if (site == GraphHookSite::FfnUp && layer >= 0) {
        ++ffn_up_hits_;
        const uint32_t ntok = t.ne1 == 0 ? 1u : t.ne1;
        for (uint32_t col = 0; col < ntok; ++col) {
            const float* up = t.data + static_cast<size_t>(col) * t.ne0;
            const uint32_t n = t.ne0 < kFfnIntermediate ? t.ne0 : kFfnIntermediate;
            bool ok = false;
            if (t.on_device && pending_gate_device_ && device_gate_) {
                ok = hooks_.on_ffn_activations_device(static_cast<uint32_t>(layer),
                                                      device_gate_ + static_cast<size_t>(col) * device_gate_n_,
                                                      up, n);
            } else if (!t.on_device && !pending_gate_device_ && pending_gate_layer_ == layer) {
                ok = hooks_.on_ffn_activations(static_cast<uint32_t>(layer),
                                               pending_gate_.data(), up, n);
            }
            (void)ok;
        }
        if (!streamer_.ffn_is_parked(static_cast<uint32_t>(layer))) {
            streamer_.evict_ffn(static_cast<uint32_t>(layer));
        }
        pending_gate_layer_ = -1;
        device_gate_ = nullptr;
        return true;
    }

    if (site == GraphHookSite::AttnResidual && layer >= 0 &&
        is_delta_net_layer(static_cast<uint32_t>(layer))) {
        const uint32_t ntok = t.ne1 == 0 ? 1u : t.ne1;
        for (uint32_t col = 0; col < ntok; ++col) {
            const float* hout = t.data + static_cast<size_t>(col) * t.ne0;
            maybe_delta(static_cast<uint32_t>(layer), hout, t.ne0, col);
        }
        return true;
    }

    return site != GraphHookSite::None;
}

void GraphHookSession::maybe_delta(uint32_t layer, const float* hidden_out, uint32_t dim,
                                   uint32_t /*token_col*/) {
    if (!hidden_out || dim == 0) {
        return;
    }
    const uint32_t n = dim < kHiddenDim ? dim : kHiddenDim;
    if (is_delta_net_layer(layer)) {
        const uint32_t pack = pack_id_from_delta_layer(layer);
        hooks_.on_delta_hidden(pack, hidden_in_.data(), hidden_out, n);
        ++delta_hits_;
    }
    std::memcpy(hidden_in_.data(), hidden_out, n * sizeof(float));
}

}  // namespace micro_llm
