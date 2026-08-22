#include "micro_llm/gguf_meta.hpp"

#include <cstring>
#include <fstream>
#include <vector>

namespace micro_llm {
namespace {

enum class GgufType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

bool read_exact(std::istream& is, char* buf, std::streamsize n) {
    is.read(buf, n);
    return static_cast<std::streamsize>(is.gcount()) == n;
}

bool read_u32(std::istream& is, uint32_t& v) {
    unsigned char b[4];
    if (!read_exact(is, reinterpret_cast<char*>(b), 4)) {
        return false;
    }
    v = static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) |
        (static_cast<uint32_t>(b[2]) << 16) | (static_cast<uint32_t>(b[3]) << 24);
    return true;
}

bool read_u64(std::istream& is, uint64_t& v) {
    uint32_t lo = 0, hi = 0;
    if (!read_u32(is, lo) || !read_u32(is, hi)) {
        return false;
    }
    v = static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
    return true;
}

bool read_string(std::istream& is, std::string& s) {
    uint64_t n = 0;
    if (!read_u64(is, n) || n > (1ull << 20)) {
        return false;
    }
    s.assign(static_cast<size_t>(n), '\0');
    return n == 0 || read_exact(is, s.data(), static_cast<std::streamsize>(n));
}

void write_u32(std::ostream& os, uint32_t v) {
    const unsigned char b[4] = {
        static_cast<unsigned char>(v & 0xffu),
        static_cast<unsigned char>((v >> 8) & 0xffu),
        static_cast<unsigned char>((v >> 16) & 0xffu),
        static_cast<unsigned char>((v >> 24) & 0xffu),
    };
    os.write(reinterpret_cast<const char*>(b), 4);
}

void write_u64(std::ostream& os, uint64_t v) {
    write_u32(os, static_cast<uint32_t>(v & 0xffffffffu));
    write_u32(os, static_cast<uint32_t>(v >> 32));
}

void write_string(std::ostream& os, const std::string& s) {
    write_u64(os, s.size());
    os.write(s.data(), static_cast<std::streamsize>(s.size()));
}

bool skip_value(std::istream& is, uint32_t type);

bool skip_array(std::istream& is) {
    uint32_t et = 0;
    uint64_t n = 0;
    if (!read_u32(is, et) || !read_u64(is, n) || n > (1ull << 28)) {
        return false;
    }
    for (uint64_t i = 0; i < n; ++i) {
        if (!skip_value(is, et)) {
            return false;
        }
    }
    return true;
}

bool skip_value(std::istream& is, uint32_t type) {
    switch (static_cast<GgufType>(type)) {
        case GgufType::UINT8:
        case GgufType::INT8:
        case GgufType::BOOL:
            return is.seekg(1, std::ios::cur).good();
        case GgufType::UINT16:
        case GgufType::INT16:
            return is.seekg(2, std::ios::cur).good();
        case GgufType::UINT32:
        case GgufType::INT32:
        case GgufType::FLOAT32:
            return is.seekg(4, std::ios::cur).good();
        case GgufType::UINT64:
        case GgufType::INT64:
        case GgufType::FLOAT64:
            return is.seekg(8, std::ios::cur).good();
        case GgufType::STRING: {
            std::string tmp;
            return read_string(is, tmp);
        }
        case GgufType::ARRAY:
            return skip_array(is);
        default:
            return false;
    }
}

bool read_u32_value(std::istream& is, uint32_t type, uint32_t& out) {
    if (static_cast<GgufType>(type) == GgufType::UINT32) {
        return read_u32(is, out);
    }
    if (static_cast<GgufType>(type) == GgufType::UINT64) {
        uint64_t v = 0;
        if (!read_u64(is, v)) {
            return false;
        }
        out = static_cast<uint32_t>(v);
        return true;
    }
    if (static_cast<GgufType>(type) == GgufType::INT32) {
        uint32_t v = 0;
        if (!read_u32(is, v)) {
            return false;
        }
        out = v;
        return true;
    }
    return skip_value(is, type);
}

bool is_vision_tensor_name(const std::string& name) {
    auto starts = [&](const char* p) { return name.rfind(p, 0) == 0; };
    return starts("v.") || starts("mm.") || starts("vision.") || starts("clip.") ||
           name.find(".vision.") != std::string::npos;
}

}  // namespace

bool read_gguf_meta(const std::string& path, GgufKv& out, std::string* err) {
    out = GgufKv{};
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        out.error = "failed to open GGUF";
        if (err) {
            *err = out.error;
        }
        return false;
    }
    char magic[4];
    if (!read_exact(is, magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
        out.error = "not a GGUF (bad magic)";
        if (err) {
            *err = out.error;
        }
        return false;
    }
    uint32_t version = 0;
    if (!read_u32(is, version) || version < 2 || version > 3) {
        out.error = "unsupported GGUF version";
        if (err) {
            *err = out.error;
        }
        return false;
    }
    if (!read_u64(is, out.n_tensors) || !read_u64(is, out.n_kv)) {
        out.error = "truncated GGUF header";
        if (err) {
            *err = out.error;
        }
        return false;
    }

    std::string arch_prefix;
    for (uint64_t i = 0; i < out.n_kv; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!read_string(is, key) || !read_u32(is, type)) {
            out.error = "truncated GGUF KV";
            if (err) {
                *err = out.error;
            }
            return false;
        }
        if (static_cast<GgufType>(type) == GgufType::STRING) {
            std::string s;
            if (!read_string(is, s)) {
                out.error = "truncated GGUF string KV";
                if (err) {
                    *err = out.error;
                }
                return false;
            }
            if (key == "general.architecture") {
                out.architecture = s;
                arch_prefix = s + ".";
            }
        } else if (static_cast<GgufType>(type) == GgufType::BOOL) {
            unsigned char b = 0;
            if (!read_exact(is, reinterpret_cast<char*>(&b), 1)) {
                out.error = "truncated GGUF bool KV";
                if (err) {
                    *err = out.error;
                }
                return false;
            }
            if (key == kKvServeOk) {
                out.serve_ok_present = true;
                out.serve_ok = b != 0;
            }
        } else if (static_cast<GgufType>(type) == GgufType::UINT32 ||
                   static_cast<GgufType>(type) == GgufType::INT32 ||
                   static_cast<GgufType>(type) == GgufType::UINT64) {
            uint32_t v = 0;
            if (!read_u32_value(is, type, v)) {
                out.error = "truncated GGUF int KV";
                if (err) {
                    *err = out.error;
                }
                return false;
            }
            if (!arch_prefix.empty()) {
                if (key == arch_prefix + "block_count") {
                    out.n_layers = v;
                } else if (key == arch_prefix + "embedding_length") {
                    out.n_embd = v;
                } else if (key == arch_prefix + "feed_forward_length") {
                    out.n_ff = v;
                }
            }
            if (key == "qwen35.block_count" || key == "qwen3next.block_count") {
                out.n_layers = v;
            } else if (key == "qwen35.embedding_length" ||
                       key == "qwen3next.embedding_length") {
                out.n_embd = v;
            } else if (key == "qwen35.feed_forward_length" ||
                       key == "qwen3next.feed_forward_length") {
                out.n_ff = v;
            }
        } else if (static_cast<GgufType>(type) == GgufType::ARRAY) {
            uint32_t et = 0;
            uint64_t n = 0;
            if (!read_u32(is, et) || !read_u64(is, n)) {
                out.error = "truncated GGUF array KV";
                if (err) {
                    *err = out.error;
                }
                return false;
            }
            if (key == "tokenizer.ggml.tokens") {
                out.n_vocab = static_cast<uint32_t>(n > UINT32_MAX ? UINT32_MAX : n);
            }
            const bool keep_n = (key == kKvKeepChannelsN);
            if (keep_n) {
                out.keep_channel_n.clear();
                out.keep_channel_n.reserve(static_cast<size_t>(n > 256 ? 256 : n));
            }
            for (uint64_t j = 0; j < n; ++j) {
                if (keep_n && (static_cast<GgufType>(et) == GgufType::UINT32 ||
                               static_cast<GgufType>(et) == GgufType::INT32)) {
                    uint32_t v = 0;
                    if (!read_u32(is, v)) {
                        out.error = "truncated keep_channels.n";
                        if (err) {
                            *err = out.error;
                        }
                        return false;
                    }
                    out.keep_channel_n.push_back(v);
                } else if (!skip_value(is, et)) {
                    out.error = "truncated GGUF array values";
                    if (err) {
                        *err = out.error;
                    }
                    return false;
                }
            }
        } else if (!skip_value(is, type)) {
            out.error = "truncated GGUF KV value";
            if (err) {
                *err = out.error;
            }
            return false;
        }
    }

    for (uint64_t t = 0; t < out.n_tensors && t < 4096; ++t) {
        std::string name;
        uint32_t n_dims = 0;
        if (!read_string(is, name) || !read_u32(is, n_dims)) {
            break;
        }
        for (uint32_t d = 0; d < n_dims; ++d) {
            uint64_t dim = 0;
            if (!read_u64(is, dim)) {
                break;
            }
        }
        uint32_t ttype = 0;
        uint64_t off = 0;
        if (!read_u32(is, ttype) || !read_u64(is, off)) {
            break;
        }
        if (is_vision_tensor_name(name)) {
            out.has_vision_tensors = true;
        }
        if (name.find("nextn") != std::string::npos || name.find("shared_head") != std::string::npos) {
            out.has_mtp_tensors = true;
        }
    }

    out.ok = true;
    return true;
}

bool gguf_looks_like_qwen27b_hybrid(const GgufKv& m) {
    const bool arch_ok = m.architecture == "qwen35" || m.architecture == "qwen3next" ||
                         m.architecture == "qwen35moe";
    if (!arch_ok && m.architecture.find("qwen3") == std::string::npos) {
        return false;
    }
    if (m.n_layers != 0 && !gguf_block_count_is_hybrid(m.n_layers)) {
        return false;
    }
    if (m.n_embd != 0 && m.n_embd != kHiddenDim) {
        return false;
    }
    if (m.n_ff != 0 && m.n_ff != kFfnIntermediate) {
        return false;
    }
    return arch_ok || (m.n_layers == kNLayers && m.n_embd == kHiddenDim);
}

bool write_gguf_kv_stub(const std::string& path, const std::string& architecture,
                        bool serve_ok_present, bool serve_ok, uint32_t n_layers,
                        uint32_t n_embd, uint32_t n_ff, std::string* err,
                        const std::vector<uint32_t>* keep_channel_n) {
    std::ofstream os(path, std::ios::binary);
    if (!os) {
        if (err) {
            *err = "failed to write stub GGUF";
        }
        return false;
    }
    const std::string prefix = architecture + ".";
    uint64_t n_kv = 4;  // arch + 3 dims
    if (serve_ok_present) {
        n_kv += 1;
    }
    if (keep_channel_n && !keep_channel_n->empty()) {
        n_kv += 1;
    }
    os.write("GGUF", 4);
    write_u32(os, 3);
    write_u64(os, 0);  // n_tensors
    write_u64(os, n_kv);

    write_string(os, "general.architecture");
    write_u32(os, static_cast<uint32_t>(GgufType::STRING));
    write_string(os, architecture);

    write_string(os, prefix + "block_count");
    write_u32(os, static_cast<uint32_t>(GgufType::UINT32));
    write_u32(os, n_layers);

    write_string(os, prefix + "embedding_length");
    write_u32(os, static_cast<uint32_t>(GgufType::UINT32));
    write_u32(os, n_embd);

    write_string(os, prefix + "feed_forward_length");
    write_u32(os, static_cast<uint32_t>(GgufType::UINT32));
    write_u32(os, n_ff);

    if (serve_ok_present) {
        write_string(os, kKvServeOk);
        write_u32(os, static_cast<uint32_t>(GgufType::BOOL));
        const unsigned char b = serve_ok ? 1 : 0;
        os.write(reinterpret_cast<const char*>(&b), 1);
    }
    if (keep_channel_n && !keep_channel_n->empty()) {
        write_string(os, kKvKeepChannelsN);
        write_u32(os, static_cast<uint32_t>(GgufType::ARRAY));
        write_u32(os, static_cast<uint32_t>(GgufType::INT32));
        write_u64(os, keep_channel_n->size());
        for (uint32_t v : *keep_channel_n) {
            write_u32(os, v);
        }
    }
    if (!os) {
        if (err) {
            *err = "failed to finish stub GGUF";
        }
        return false;
    }
    return true;
}

}  // namespace micro_llm
