#pragma once

// llama.cpp / ggml graph-eval attach. The live tensor names come from
// qwen35.cpp cb() calls: "ffn_gate-L", "ffn_up-L", "attn_residual-L",
// "l_out-L", "model.input_embed", "result_output".
//
// This matcher is compile-tested without linking llama.cpp.

#include "micro_llm/streamer.hpp"
#include "micro_llm/trace_hooks.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace micro_llm {

enum class GraphHookSite : uint8_t {
    None = 0,
    FfnGate,
    FfnUp,
    FfnGatePar,  // already SiLU(gate)*up — skip, we want raw gate/up
    AttnResidual,
    LayerOut,
    InputEmbed,
    Logits,
};

struct GraphTensorView {
    std::string name;
    const float* data = nullptr;  // host or device
    bool on_device = false;
    uint32_t ne0 = 0;  // inner dim (n_ff or n_embd)
    uint32_t ne1 = 1;  // tokens in this ubatch
};

GraphHookSite classify_graph_tensor(std::string_view name, int* layer_out);

// Session that turns graph tensors into TraceHooks + streamer calls.
class GraphHookSession {
public:
    GraphHookSession(TraceHooks& hooks, TraceStreamer& streamer);

    // ask=true: do we want this tensor? ask=false: data is ready.
    bool on_tensor(const GraphTensorView& t, bool ask);

    void begin_token(uint32_t token_index);
    void finish_token(uint32_t sampled, const uint32_t* topk, uint32_t k,
                      bool special_or_high_loss);

    // Called with one HTR1 record (kHtr1RecordBytes) BEFORE after_logits.
    void set_on_htr1(std::function<void(const uint8_t*, size_t)> fn) { on_htr1_ = std::move(fn); }

    uint32_t ffn_gate_hits() const { return ffn_gate_hits_; }
    uint32_t ffn_up_hits() const { return ffn_up_hits_; }
    uint32_t delta_hits() const { return delta_hits_; }
    // Run total: FFN gate/up activations that arrived on host, not CUDA.
    uint64_t host_ffn_binds() const { return host_ffn_binds_; }
    // This token: FFN layers whose hook did not run (no fire tap).
    uint32_t missing_hooks_this_token() const;

private:
    void maybe_delta(uint32_t layer, const float* hidden_out, uint32_t dim,
                     uint32_t token_col);

    TraceHooks& hooks_;
    TraceStreamer& streamer_;
    std::function<void(const uint8_t*, size_t)> on_htr1_;
    std::vector<float> hidden_in_;
    std::vector<float> pending_gate_;
    const float* device_gate_ = nullptr;
    uint32_t device_gate_n_ = 0;
    int pending_gate_layer_ = -1;
    bool pending_gate_device_ = false;
    uint32_t ffn_gate_hits_ = 0;
    uint32_t ffn_up_hits_ = 0;
    uint32_t delta_hits_ = 0;
    uint64_t host_ffn_binds_ = 0;
    uint64_t ffn_seen_mask_ = 0;
};

}  // namespace micro_llm
