#include "micro_llm/serve.hpp"
#include "micro_llm/gguf_meta.hpp"

namespace micro_llm {

ServeGate read_serve_gate(const std::string& gguf_path) {
    ServeGate g;
    GgufKv meta;
    std::string err;
    if (!read_gguf_meta(gguf_path, meta, &err)) {
        g.reason = err.empty() ? "failed to read GGUF" : err;
        return g;
    }
    g.key_present = meta.serve_ok_present;
    g.serve_ok = meta.serve_ok;
    g.ffn_width_ok = true;
    if (!meta.keep_channel_n.empty()) {
        for (uint32_t n : meta.keep_channel_n) {
            if (!ffn_keep_width_q4k_ok(n)) {
                g.ffn_width_ok = false;
                break;
            }
        }
    } else if (meta.n_ff != 0 && !ffn_keep_width_q4k_ok(meta.n_ff)) {
        g.ffn_width_ok = false;
    }
    g.may_serve = remnant_may_serve(g.key_present, g.serve_ok) && g.ffn_width_ok;
    if (!g.key_present) {
        g.reason = "micro_llm.serve_ok missing; refuse serve";
    } else if (!g.serve_ok) {
        g.reason = "micro_llm.serve_ok=false (F16 host dump); refuse serve";
    } else if (!g.ffn_width_ok) {
        g.reason = "packed FFN intermediate is not a multiple of 256 (Q4_K superblock); "
                   "13056 and 10496 are valid, 10445 is not";
    } else {
        g.reason = "serve_ok";
    }
    return g;
}

bool remnant_may_serve_file(const std::string& gguf_path, std::string* err) {
    const ServeGate g = read_serve_gate(gguf_path);
    if (!g.may_serve && err) {
        *err = g.reason;
    }
    return g.may_serve;
}

}  // namespace micro_llm
