#pragma once

// Serve-path gate. Packed remnant only. F16 host dump (--q4-k-to-f16) must
// refuse: micro_llm.serve_ok missing or false.

#include "micro_llm/types.hpp"

#include <string>

namespace micro_llm {

struct ServeGate {
    bool key_present = false;
    bool serve_ok = false;
    bool ffn_width_ok = true;
    bool may_serve = false;
    std::string reason;
};

// True only if key is present AND true. ffn_keep=0 skips the Q4_K width check.
inline bool remnant_may_serve_gate(bool key_present, bool serve_ok,
                                  uint32_t ffn_keep = 0) {
    return remnant_may_serve(key_present, serve_ok, ffn_keep);
}

// Read micro_llm.serve_ok from a GGUF. Missing file / missing key / false
// all refuse. False is the F16 host-dump path.
ServeGate read_serve_gate(const std::string& gguf_path);

bool remnant_may_serve_file(const std::string& gguf_path, std::string* err = nullptr);

}  // namespace micro_llm
