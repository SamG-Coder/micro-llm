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
    g.may_serve = remnant_may_serve(g.key_present, g.serve_ok);
    if (!g.key_present) {
        g.reason = "micro_llm.serve_ok missing; refuse serve";
    } else if (!g.serve_ok) {
        g.reason = "micro_llm.serve_ok=false (F16 host dump); refuse serve";
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
