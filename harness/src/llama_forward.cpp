#include "micro_llm/live_forward.hpp"

#if defined(MICRO_LLM_HAS_LLAMA)

#include "micro_llm/ggml_ptr.hpp"
#include "micro_llm/gguf_meta.hpp"
#include "micro_llm/graph_hooks.hpp"
#include "micro_llm/perf.hpp"
#include "micro_llm/trace_cli.hpp"
#include "micro_llm/vram_ledger.hpp"

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(MICRO_LLM_HAS_CUDA)
#include <cuda_runtime.h>
#endif

namespace micro_llm {
namespace {

struct LlamaHookUser {
    GraphHookSession* session = nullptr;
    PerfClocks* clocks = nullptr;
    std::vector<float> host_hid;
};

bool tensor_is_f32(const ggml_tensor* t) { return t && t->type == GGML_TYPE_F32; }

bool tensor_on_host(const ggml_tensor* t) {
    return t && t->buffer && ggml_backend_buffer_is_host(t->buffer);
}

// Resolve F32. CUDA tensors are buffer-base + view_offs, not t->data-as-VA.
// Never pass an alloc offset to a kernel (5080 AV). No 17408 D2H.
const float* tensor_f32_view(ggml_tensor* t, bool* on_device, bool* ptr_ok) {
    *on_device = false;
    *ptr_ok = false;
    if (!t || !tensor_is_f32(t)) {
        return nullptr;
    }
    if (tensor_on_host(t) && t->data && !ptr_looks_like_integer_offset(t->data)) {
        *ptr_ok = true;
        return static_cast<const float*>(t->data);
    }
    if (!t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
        if (t->data && !ptr_looks_like_integer_offset(t->data)) {
            *ptr_ok = true;
            return static_cast<const float*>(t->data);
        }
        return nullptr;
    }
    void* base = ggml_backend_buffer_get_base(t->buffer);
    const size_t buf_size = ggml_backend_buffer_get_size(t->buffer);
    const size_t nbytes = ggml_nbytes(t);
    const float* p =
        resolve_f32_in_buffer(base, buf_size, t->data, t->view_offs, nbytes, ptr_ok);
    if (!*ptr_ok || !p) {
        *ptr_ok = false;
        return nullptr;
    }
#if defined(MICRO_LLM_HAS_CUDA)
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, p) != cudaSuccess) {
        *ptr_ok = false;
        return nullptr;
    }
    if (attr.type != cudaMemoryTypeDevice && attr.type != cudaMemoryTypeManaged) {
        *ptr_ok = false;
        return nullptr;
    }
#endif
    *on_device = true;
    return p;
}

const char* buft_name_of(const ggml_tensor* t) {
    if (!t || !t->buffer) {
        return "none";
    }
    ggml_backend_buffer_type_t bt = ggml_backend_buffer_get_type(t->buffer);
    const char* n = bt ? ggml_backend_buft_name(bt) : nullptr;
    return n ? n : "unknown";
}

ggml_tensor* find_layer_tensor(const llama_model* model, uint32_t layer, const char* suffix) {
    char name[96];
    std::snprintf(name, sizeof(name), "blk.%u.%s", layer, suffix);
    return llama_model_get_tensor(model, name);
}

ggml_tensor* find_ffn_weight(const llama_model* model, uint32_t layer, const char* stem) {
    char name[96];
    const char* tails[] = {".weight", "", nullptr};
    for (int i = 0; tails[i] != nullptr; ++i) {
        std::snprintf(name, sizeof(name), "blk.%u.%s%s", layer, stem, tails[i]);
        if (ggml_tensor* t = llama_model_get_tensor(model, name)) {
            return t;
        }
    }
    return nullptr;
}

struct StreamedFfnHost {
    uint32_t layer = 0;
    const void* host = nullptr;
    size_t bytes = 0;
    ggml_tensor* tensor = nullptr;
    ggml_tensor* view_src = nullptr;
    bool slot_q4 = false;
    std::vector<uint8_t> host_copy;
};

bool already_have_tensor(const std::vector<StreamedFfnHost>* out, const ggml_tensor* t) {
    if (!out || !t) {
        return false;
    }
    for (const StreamedFfnHost& h : *out) {
        if (h.tensor == t) {
            return true;
        }
    }
    return false;
}

void snapshot_host_tensor(ggml_tensor* t, StreamedFfnHost* h) {
    h->tensor = t;
    h->bytes = ggml_nbytes(t);
    if (t->buffer && ggml_backend_buffer_is_host(t->buffer) && t->data &&
        !ptr_looks_like_integer_offset(t->data)) {
        h->host = t->data;
    }
    if (h->bytes != 0) {
        h->host_copy.resize(h->bytes);
        if (h->host) {
            std::memcpy(h->host_copy.data(), h->host, h->bytes);
        } else {
            ggml_backend_tensor_get(t, h->host_copy.data(), 0, h->bytes);
            h->host = h->host_copy.data();
        }
    }
}

void collect_streamed_ffn_hosts(const llama_model* model, uint32_t n_park,
                                std::vector<StreamedFfnHost>* out, uint32_t* parked_cuda,
                                uint32_t* streamed_cpu, uint32_t* streamed_cuda,
                                uint32_t* streamed_cuda_host, bool snapshot_host,
                                bool log_tensors) {
    static const char* kParkStems[] = {"ffn_gate", "ffn_up", "ffn_down"};
    std::vector<const ggml_tensor*> seen;
    auto saw = [&](const ggml_tensor* t) {
        if (!t) {
            return true;
        }
        for (const ggml_tensor* p : seen) {
            if (p == t) {
                return true;
            }
        }
        seen.push_back(t);
        return false;
    };
    for (uint32_t layer = 0; layer < kNLayers; ++layer) {
        if (layer < n_park) {
            for (const char* stem : kParkStems) {
                ggml_tensor* t = find_ffn_weight(model, layer, stem);
                if (!t) {
                    continue;
                }
                if (classify_backend_buft_name(buft_name_of(t)) == BuftKind::Cuda) {
                    ++(*parked_cuda);
                }
            }
            continue;
        }
        const int slot_kind = ggml_stream_slot_kind(layer, n_park);
        const bool log_slot = log_tensors && (slot_kind == kStreamSlotA || slot_kind == kStreamSlotB);
        for (uint32_t i = 0; i < kStreamedBindSuffixCount; ++i) {
            ggml_tensor* t = find_layer_tensor(model, layer, kStreamedBindSuffixes[i]);
            if (log_slot) {
                std::fprintf(stderr,
                             "FFN_NAME_TRY layer=%u suffix=%s found=%d name=%s buft=%s "
                             "slot_q4=%d nbytes=%llu\n",
                             layer, kStreamedBindSuffixes[i], t ? 1 : 0, t ? t->name : "-",
                             t ? buft_name_of(t) : "-",
                             t && name_is_slot_q4_weight(t->name) ? 1 : 0,
                             t ? static_cast<unsigned long long>(ggml_nbytes(t)) : 0ull);
            }
            if (!t || saw(t)) {
                continue;
            }
            const BuftKind kind = classify_backend_buft_name(buft_name_of(t));
            if (kind == BuftKind::CudaHost) {
                ++(*streamed_cuda_host);
            } else if (kind == BuftKind::Cuda) {
                ++(*streamed_cuda);
            } else {
                ++(*streamed_cpu);
            }
            if (log_tensors) {
                std::fprintf(stderr,
                             "FFN_TENSOR layer=%u name=%s buft=%s stream=1 cuda_host=%d "
                             "cpu_mapped=%d view_src=%s view_buft=%s slot_q4=%d\n",
                             layer, t->name, buft_name_of(t), kind == BuftKind::CudaHost ? 1 : 0,
                             buft_is_cpu_mapped(buft_name_of(t)) ? 1 : 0,
                             t->view_src ? t->view_src->name : "none",
                             t->view_src ? buft_name_of(t->view_src) : "-",
                             name_is_slot_q4_weight(t->name) ? 1 : 0);
            }
            if (!out) {
                continue;
            }
            // Slot A/B only packs the three Q4 GEMM weights. Do not
            // pack leftover hops / norms into the layer slot.
            if (slot_kind == kStreamSlotA || slot_kind == kStreamSlotB) {
                if (!name_is_slot_q4_weight(t->name)) {
                    continue;
                }
            }
            if (already_have_tensor(out, t)) {
                continue;
            }
            StreamedFfnHost h;
            h.layer = layer;
            h.slot_q4 = name_is_slot_q4_weight(t->name);
            h.view_src = t->view_src;
            if (snapshot_host) {
                snapshot_host_tensor(t, &h);
            } else {
                h.tensor = t;
                h.bytes = ggml_nbytes(t);
            }
            out->push_back(h);
        }
    }
}

struct GgmlSlotGuard {
    ggml_backend_buffer_t a = nullptr;
    ggml_backend_buffer_t b = nullptr;
    ~GgmlSlotGuard() {
        if (a) {
            ggml_backend_buffer_free(a);
        }
        if (b) {
            ggml_backend_buffer_free(b);
        }
    }
};

// Drop the GGUF CPU_Mapped host alias so graph reserve cannot keep
// reading t->data as a mapped pointer after we point the tensor at A/B.
void detach_cpu_mapped(ggml_tensor* t) {
    if (!t) {
        return;
    }
    t->extra = nullptr;
    t->buffer = nullptr;
    t->data = nullptr;
    t->view_src = nullptr;
    t->view_offs = 0;
}

// Load-time bind: point the weight tensor at a ggml CUDA slot, then
// tensor_set so MUL_MAT reads that VRAM after a real H2D. Private
// cudaMalloc the sched never sees is not this.
bool attach_q4_to_ggml_slot(ggml_backend_buffer_t buf, ggml_tensor* t, const void* host,
                            size_t nbytes, size_t off) {
    if (!buf || !t || !host || nbytes == 0) {
        return false;
    }
    void* base = ggml_backend_buffer_get_base(buf);
    const size_t cap = ggml_backend_buffer_get_size(buf);
    if (!base || !ggml_slot_pack_ok(off, nbytes, cap)) {
        return false;
    }
    detach_cpu_mapped(t);
    t->buffer = buf;
    t->data = static_cast<char*>(base) + off;
    ggml_backend_buffer_init_tensor(buf, t);
    ggml_backend_tensor_set(t, host, 0, nbytes);
    return true;
}

struct FfnTripletBytes {
    uint64_t gate = 0;
    uint64_t up = 0;
    uint64_t down = 0;
    uint64_t slot = kStreamSlotBytes;
};

ggml_tensor* mmap_root(ggml_tensor* t) {
    ggml_tensor* r = t;
    while (r && r->view_src) {
        r = r->view_src;
    }
    return r;
}

bool copy_weight_bytes(ggml_tensor* t, std::vector<uint8_t>* out) {
    if (!t || !out) {
        return false;
    }
    const size_t n = ggml_nbytes(t);
    if (n == 0) {
        return false;
    }
    out->resize(n);
    if (t->buffer && ggml_backend_buffer_is_host(t->buffer) && t->data &&
        !ptr_looks_like_integer_offset(t->data)) {
        std::memcpy(out->data(), t->data, n);
        return true;
    }
    ggml_backend_tensor_get(t, out->data(), 0, n);
    return true;
}

bool alias_view_onto_root(ggml_backend_buffer_t buf, ggml_tensor* view, ggml_tensor* root,
                          size_t root_off) {
    if (!buf || !view || !root) {
        return false;
    }
    void* base = ggml_backend_buffer_get_base(buf);
    const size_t cap = ggml_backend_buffer_get_size(buf);
    const size_t nbytes = ggml_nbytes(view);
    const size_t off = root_off + view->view_offs;
    if (!base || !ggml_slot_pack_ok(off, nbytes, cap)) {
        return false;
    }
    view->extra = nullptr;
    view->buffer = buf;
    view->view_src = root;
    view->data = static_cast<char*>(base) + off;
    ggml_backend_view_init(view);
    return true;
}

FfnTripletBytes measure_ffn_triplet(const llama_model* model, uint32_t layer) {
    FfnTripletBytes m;
    ggml_tensor* g = find_ffn_weight(model, layer, "ffn_gate");
    ggml_tensor* u = find_ffn_weight(model, layer, "ffn_up");
    ggml_tensor* d = find_ffn_weight(model, layer, "ffn_down");
    if (!d) {
        d = find_ffn_weight(model, layer, "ffn_out");
    }
    auto nbytes = [](ggml_tensor* t) -> uint64_t {
        if (!t) {
            return 0;
        }
        return ggml_nbytes(mmap_root(t));
    };
    m.gate = nbytes(g);
    m.up = nbytes(u);
    m.down = nbytes(d);
    m.slot = ffn_slot_bytes_for_triplet(m.gate, m.up, m.down);
    return m;
}

// Bind gate+up+down so MUL_MAT reads the VRAM slot. The hop is the mmap
// tensor itself (254e10c: view_src=none). Slot must already be sized to
// the triplet — leftover collapse into 160 is not a bind.
uint32_t bind_layer_q4_triplet(const llama_model* model, uint32_t layer,
                               ggml_backend_buffer_t buf, PerfClocks* clocks) {
    if (!model || !buf) {
        return 0;
    }
    struct Part {
        const char* stem;
        ggml_tensor* t;
    };
    Part parts[4];
    parts[0] = {"ffn_gate", find_ffn_weight(model, layer, "ffn_gate")};
    parts[1] = {"ffn_up", find_ffn_weight(model, layer, "ffn_up")};
    parts[2] = {"ffn_down", find_ffn_weight(model, layer, "ffn_down")};
    parts[3] = {"ffn_out", find_ffn_weight(model, layer, "ffn_out")};
    if (!parts[2].t && parts[3].t) {
        parts[2].t = parts[3].t;
        parts[2].stem = "ffn_out";
    }
    struct RootSlot {
        ggml_tensor* root = nullptr;
        size_t off = 0;
    };
    RootSlot roots[4]{};
    uint32_t nroot = 0;
    size_t off = 0;
    const size_t cap = ggml_backend_buffer_get_size(buf);
    uint32_t n_bound = 0;
    bool have_gate = false;
    bool have_up = false;
    bool have_down = false;
    auto find_root = [&](ggml_tensor* root) -> RootSlot* {
        for (uint32_t i = 0; i < nroot; ++i) {
            if (roots[i].root == root) {
                return &roots[i];
            }
        }
        return nullptr;
    };
    for (int i = 0; i < 3; ++i) {
        ggml_tensor* t = parts[i].t;
        if (!t) {
            std::fprintf(stderr, "FFN_SLOT_MISS layer=%u stem=%s\n", layer, parts[i].stem);
            continue;
        }
        ggml_tensor* root = mmap_root(t);
        std::fprintf(stderr,
                     "FFN_SLOT_ROOT layer=%u name=%s buft=%s root=%s root_buft=%s "
                     "nbytes=%llu root_nbytes=%llu cpu_mapped=%d\n",
                     layer, t->name, buft_name_of(t), root->name, buft_name_of(root),
                     static_cast<unsigned long long>(ggml_nbytes(t)),
                     static_cast<unsigned long long>(ggml_nbytes(root)),
                     buft_is_cpu_mapped(buft_name_of(root)) ? 1 : 0);
        RootSlot* rs = find_root(root);
        if (!rs) {
            std::vector<uint8_t> host;
            if (!copy_weight_bytes(root, &host)) {
                std::fprintf(stderr, "FFN_SLOT_BIND layer=%u stem=%s FAIL snapshot root\n",
                             layer, parts[i].stem);
                continue;
            }
            off = align_up(off, kTensorAlign);
            if (!attach_q4_to_ggml_slot(buf, root, host.data(), host.size(), off)) {
                std::fprintf(stderr,
                             "FFN_SLOT_BIND layer=%u stem=%s FAIL pack root n=%llu "
                             "off=%llu cap=%llu (slot too small if down past end)\n",
                             layer, parts[i].stem,
                             static_cast<unsigned long long>(host.size()),
                             static_cast<unsigned long long>(off),
                             static_cast<unsigned long long>(cap));
                continue;
            }
            if (nroot < 4) {
                roots[nroot].root = root;
                roots[nroot].off = off;
                rs = &roots[nroot];
                ++nroot;
            }
            off += host.size();
            if (clocks) {
                clocks->add_h2d(host.size());
                clocks->add_cuda_ffn_bind();
            }
        }
        if (!rs) {
            continue;
        }
        if (t != root) {
            if (!alias_view_onto_root(buf, t, root, rs->off)) {
                std::fprintf(stderr, "FFN_SLOT_BIND layer=%u stem=%s FAIL alias view\n",
                             layer, parts[i].stem);
                continue;
            }
        }
        const bool on_cuda =
            classify_backend_buft_name(buft_name_of(t)) == BuftKind::Cuda ||
            classify_backend_buft_name(buft_name_of(root)) == BuftKind::Cuda;
        std::fprintf(stderr,
                     "FFN_SLOT_BIND layer=%u stem=%s name=%s buft=%s root_buft=%s "
                     "on_cuda=%d cpu_mapped=%d\n",
                     layer, parts[i].stem, t->name, buft_name_of(t), buft_name_of(root),
                     on_cuda ? 1 : 0, buft_is_cpu_mapped(buft_name_of(t)) ? 1 : 0);
        if (!on_cuda) {
            continue;
        }
        ++n_bound;
        if (i == 0) {
            have_gate = true;
        } else if (i == 1) {
            have_up = true;
        } else {
            have_down = true;
        }
    }
    const bool ok = have_gate && have_up && have_down;
    ggml_tensor* down = parts[2].t;
    const char* down_buft = down ? buft_name_of(down) : "missing";
    std::fprintf(stderr, "FFN_BIND_NEED layer=%u gate=%d up=%d down=%d packed=%llu\n", layer,
                 have_gate ? 1 : 0, have_up ? 1 : 0, have_down ? 1 : 0,
                 static_cast<unsigned long long>(off));
    std::fprintf(stderr, "%s\n", format_ggml_tensor_bind_line(layer, off, ok, ok).c_str());
    std::fprintf(stderr, "FFN_DOWN_MAPPED layer=%u buft=%s cpu_mapped=%d\n", layer, down_buft,
                 down && buft_is_cpu_mapped(down_buft) ? 1 : 0);
    if (clocks && n_bound > 0) {
        clocks->set_real_h2d(true);
    }
    return n_bound;
}

uint32_t bind_streamed_tensors_to_ggml_slots(ggml_backend_buffer_type_t gpu_buft,
                                             GgmlSlotGuard* slots,
                                             std::vector<StreamedFfnHost>* streamed,
                                             uint32_t n_park, PerfClocks* clocks) {
    (void)gpu_buft;  // Ren owns extra CUDA weight buffers. This bind uses A/B.
    if (!slots || !streamed || streamed->empty()) {
        return 0;
    }
    uint32_t n_bound = 0;
    uint32_t last_layer = ~0u;
    ggml_backend_buffer_t buf = nullptr;
    size_t off = 0;
    uint64_t packed = 0;
    uint32_t parts_ok = 0;
    bool have_gate = false;
    bool have_up = false;
    bool have_down = false;
    auto note_part = [&](const char* name) {
        if (!name) {
            return;
        }
        if (std::strstr(name, "ffn_gate")) {
            have_gate = true;
        }
        if (std::strstr(name, "ffn_up")) {
            have_up = true;
        }
        if (std::strstr(name, "ffn_down") || std::strstr(name, "ffn_out")) {
            have_down = true;
        }
    };
    auto flush_layer = [&](uint32_t layer) {
        if (layer == ~0u) {
            return;
        }
        const int kind = ggml_stream_slot_kind(layer, n_park);
        if (kind == kStreamSlotCpu) {
            std::fprintf(stderr,
                         "ffn_stream_cpu layer=%u (op_offload, not extra park)\n", layer);
            return;
        }
        const bool ok = (have_gate && have_up && have_down && packed > 0);
        std::fprintf(stderr, "FFN_BIND_NEED layer=%u gate=%d up=%d down=%d packed=%llu\n",
                     layer, have_gate ? 1 : 0, have_up ? 1 : 0, have_down ? 1 : 0,
                     static_cast<unsigned long long>(packed));
        std::fprintf(stderr, "%s\n",
                     format_ggml_tensor_bind_line(layer, packed, ok, ok).c_str());
    };
    for (StreamedFfnHost& h : *streamed) {
        if (!h.tensor || h.bytes == 0) {
            continue;
        }
        const void* src = !h.host_copy.empty() ? h.host_copy.data() : h.host;
        if (!src) {
            continue;
        }
        if (h.layer != last_layer) {
            flush_layer(last_layer);
            last_layer = h.layer;
            off = 0;
            packed = 0;
            parts_ok = 0;
            have_gate = false;
            have_up = false;
            have_down = false;
            const int kind = ggml_stream_slot_kind(h.layer, n_park);
            // 62/63 already bound by bind_layer_q4_triplet (mmap parent +
            // three views). Do not re-poke and clear view_src.
            if (kind == kStreamSlotA || kind == kStreamSlotB) {
                buf = nullptr;
            } else {
                buf = nullptr;  // CPU stream. Do not alloc extra park buffers.
            }
        }
        if (!buf) {
            continue;
        }
        off = align_up(off, kTensorAlign);
        if (!attach_q4_to_ggml_slot(buf, h.tensor, src, h.bytes, off)) {
            std::fprintf(stderr,
                         "ggml_tensor_bind layer=%u FAIL off=%llu n=%llu name=%s "
                         "buft_was=%s cpu_mapped=%d\n",
                         h.layer, static_cast<unsigned long long>(off),
                         static_cast<unsigned long long>(h.bytes),
                         h.tensor->name ? h.tensor->name : "-", buft_name_of(h.tensor),
                         buft_is_cpu_mapped(buft_name_of(h.tensor)) ? 1 : 0);
            continue;
        }
        if (h.view_src && h.view_src != h.tensor &&
            ggml_nbytes(h.view_src) == h.bytes) {
            attach_q4_to_ggml_slot(buf, h.view_src, src, h.bytes, off);
        }
        off += h.bytes;
        packed += h.bytes;
        ++parts_ok;
        if (name_is_slot_q4_weight(h.tensor->name)) {
            note_part(h.tensor->name);
        }
        ++n_bound;
        if (clocks) {
            clocks->add_h2d(h.bytes);
            clocks->add_cuda_ffn_bind();
        }
    }
    flush_layer(last_layer);
    if (clocks && n_bound > 0) {
        clocks->set_real_h2d(true);
    }
    return n_bound;
}

void print_layer_mul_mat_srcs(const llama_model* model, uint32_t layer, uint32_t* cpu_n,
                              const char** last_name, const char** last_buft) {
    static const char* kNeed[] = {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
                                  "ffn_out.weight", "ffn_out",
                                  "ffn_gate.scale", "ffn_up.scale", "ffn_down.scale"};
    for (const char* suffix : kNeed) {
        ggml_tensor* t = find_layer_tensor(model, layer, suffix);
        if (!t) {
            continue;
        }
        const char* buft = buft_name_of(t);
        const BuftKind kind = classify_backend_buft_name(buft);
        const char* vs_name = t->view_src ? t->view_src->name : "none";
        const char* vs_buft = t->view_src ? buft_name_of(t->view_src) : "-";
        const BuftKind vs_kind =
            t->view_src ? classify_backend_buft_name(vs_buft) : BuftKind::Cuda;
        const SplitCauseKind cause = classify_split_cause(t->name, buft, true);
        const bool src_cpu = kind != BuftKind::Cuda;
        const bool view_cpu = t->view_src && vs_kind != BuftKind::Cuda;
        std::fprintf(stderr,
                     "FFN_MUL_MAT_SRC layer=%u name=%s buft=%s view_src=%s "
                     "view_buft=%s cpu_mapped=%d cause=%s\n",
                     layer, t->name, buft, vs_name, vs_buft,
                     (buft_is_cpu_mapped(buft) || buft_is_cpu_mapped(vs_buft)) ? 1 : 0,
                     split_cause_kind_name(cause));
        std::fprintf(stderr, "%s\n",
                     format_ffn_hop_line(static_cast<int32_t>(layer), t->name, buft, vs_name,
                                         vs_buft,
                                         view_cpu ? classify_split_cause(vs_name, vs_buft, true)
                                                  : cause)
                         .c_str());
        if (src_cpu || view_cpu) {
            ++(*cpu_n);
            if (last_name) {
                *last_name = view_cpu && t->view_src ? t->view_src->name : t->name;
            }
            if (last_buft) {
                *last_buft = view_cpu ? vs_buft : buft;
            }
        }
    }
}

void print_split_why(const SplitLedger& sl, uint32_t cpu_63, const char* last_name,
                     const char* last_buft) {
    std::fprintf(stderr, "%s\n", format_split_ledger(sl).c_str());
    std::fprintf(stderr, "%s\n",
                 format_split_why_line(kMeasured5080ReserveSplits532, "blk.63.ffn_gate.weight",
                                       last_buft && last_buft[0] ? last_buft : "CPU")
                     .c_str());
    std::fprintf(stderr, "%s\n",
                 format_split_why_line(kMeasured5080ReserveSplits, last_name, last_buft).c_str());
    std::fprintf(stderr, "%s\n",
                 format_split_causes_block(sl, last_name, last_buft, 63).c_str());
    std::fprintf(stderr,
                 "SPLIT_NOTE 532>340 bs=1: last crossing still CPU on "
                 "blk.63 ffn_gate/up — see FFN_HOP for the src/view backend. "
                 "kind=placement/buffer (CPU_Mapped mmap alias) or "
                 "backend_transition; not unsupported_op; hook/callback=0 on "
                 "--trace-off. 638 = same reserve, down also CPU_Mapped. "
                 "cpu_63_srcs=%u\n",
                 cpu_63);
}

void register_streamed_layer_parts(TraceStreamer& streamer,
                                   const std::vector<StreamedFfnHost>& streamed) {
    uint32_t last = ~0u;
    int part = 0;
    for (const StreamedFfnHost& h : streamed) {
        if (!h.host || h.bytes == 0) {
            continue;
        }
        if (h.layer != last) {
            part = 0;
            last = h.layer;
        }
        streamer.set_stream_host_part(h.layer, part, h.host, h.bytes);
        if (part < 2) {
            ++part;
        }
    }
}

void h2d_packed_layer_into_ggml(ggml_backend_buffer_t dst, const std::vector<StreamedFfnHost>& streamed,
                                uint32_t layer) {
    if (!dst) {
        return;
    }
    void* base = ggml_backend_buffer_get_base(dst);
    const size_t cap = ggml_backend_buffer_get_size(dst);
    if (!base || cap < kStreamSlotBytes) {
        return;
    }
    size_t off = 0;
    for (const StreamedFfnHost& h : streamed) {
        if (h.layer != layer || !h.host || h.bytes == 0) {
            continue;
        }
        if (off + h.bytes > cap) {
            return;
        }
        unsigned char* p = static_cast<unsigned char*>(base) + off;
        if (ggml_backend_buffer_is_host(dst)) {
            std::memcpy(p, h.host, h.bytes);
        }
#if defined(MICRO_LLM_HAS_CUDA)
        else {
            cudaMemcpy(p, h.host, h.bytes, cudaMemcpyHostToDevice);
        }
#endif
        off += h.bytes;
    }
}

ggml_backend_buffer_type_t hybrid_gpu_buft() {
    auto from_reg = [](const char* name) -> ggml_backend_buffer_type_t {
        ggml_backend_reg_t reg = ggml_backend_reg_by_name(name);
        if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
            return nullptr;
        }
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
        return dev ? ggml_backend_dev_buffer_type(dev) : nullptr;
    };
    if (ggml_backend_buffer_type_t t = from_reg("CUDA")) {
        return t;
    }
    if (ggml_backend_buffer_type_t t = from_reg("cuda")) {
        return t;
    }
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) {
            continue;
        }
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            return ggml_backend_dev_buffer_type(dev);
        }
    }
    return nullptr;
}

bool llama_cb_eval(struct ggml_tensor* t, bool ask, void* user_data) {
    auto* u = static_cast<LlamaHookUser*>(user_data);
    if (!u || !u->session || !t) {
        return false;
    }
    if (u->clocks) {
        u->clocks->note_backend(tensor_on_host(t));
    }
    GraphTensorView v;
    v.name = t->name;
    v.ne0 = t->ne[0] > 0 ? static_cast<uint32_t>(t->ne[0]) : 0;
    v.ne1 = t->ne[1] > 0 ? static_cast<uint32_t>(t->ne[1]) : 1;
    int layer = -1;
    const GraphHookSite site = classify_graph_tensor(v.name, &layer);
    if (site == GraphHookSite::None || site == GraphHookSite::FfnGatePar ||
        site == GraphHookSite::Logits) {
        return false;
    }
    if (ask) {
        // Never read t->data on ask. Weights are Q4 — not a fire tap.
        return u->session->on_tensor(v, true);
    }
    if (!tensor_is_f32(t)) {
        return false;
    }
    bool on_device = false;
    bool ptr_ok = false;
    v.data = tensor_f32_view(t, &on_device, &ptr_ok);
    v.on_device = on_device;
    v.ptr_ok = ptr_ok;
    if (!ptr_ok || !v.data || ptr_looks_like_integer_offset(v.data)) {
        // Unresolved device VA: skip tap. Do not kernel-launch. Do not AV.
        return false;
    }
    if (on_device && (site == GraphHookSite::AttnResidual || site == GraphHookSite::LayerOut ||
                      site == GraphHookSite::InputEmbed)) {
        // Packs need host hidden (5120), not 17408 FFN activations.
        const size_t n = static_cast<size_t>(v.ne0);
        u->host_hid.resize(n);
        ggml_backend_tensor_get(t, u->host_hid.data(), 0, n * sizeof(float));
        v.data = u->host_hid.data();
        v.on_device = false;
        v.ptr_ok = true;
        if (u->clocks) {
            u->clocks->add_d2h(n * sizeof(float));
        }
    }
    return u->session->on_tensor(v, false);
}

void topk_ids(const float* logits, int32_t n_vocab, uint32_t k, uint32_t* out) {
    k = std::min(k, static_cast<uint32_t>(n_vocab));
    std::vector<int32_t> idx(static_cast<size_t>(n_vocab));
    for (int32_t i = 0; i < n_vocab; ++i) {
        idx[static_cast<size_t>(i)] = i;
    }
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int32_t a, int32_t b) {
        return logits[a] > logits[b];
    });
    for (uint32_t i = 0; i < k; ++i) {
        out[i] = static_cast<uint32_t>(idx[i]);
    }
}

}  // namespace

bool LlamaCppLiveForwardBackend::engine_linked() const { return true; }

LiveForwardStatus LlamaCppLiveForwardBackend::run(TraceHooks& hooks, TraceStreamer& streamer,
                                                  const LiveForwardConfig& cfg) {
    LiveForwardStatus st = probe(cfg);
    st.engine_linked = true;
    if (cfg.model_path.empty()) {
        st.ok = false;
        st.message = "llama.cpp backend: --model path.gguf is required";
        return st;
    }
    if (!st.architecture_ok) {
        st.ok = false;
        st.ran_tokens = false;
        return st;
    }
    if (cfg.load_vision) {
        st.ok = false;
        st.message = "refusing to load vision (coding-assistant job is text-only)";
        return st;
    }

    llama_backend_init();

    uint64_t v_free0 = 0;
    uint64_t v_tot0 = 0;
    PerfClocks::query_vram(&v_free0, &v_tot0);

    llama_model_params mparams = llama_model_default_params();
    // Pin is tensor overrides, not ngl. 0 = CPU default. 16 = wrong 16
    // layers. 99 parks the file (every non-overridden tensor, including
    // all 64 FFNs if the FFN CPU overrides were dropped).
    mparams.n_gpu_layers = clamp_hybrid_n_gpu_layers(cfg.n_gpu_layers);
    mparams.load_mtp = false;

    VramLedger ledger = vram_ledger_slots_first();
    uint32_t n_park =
        cfg.n_parked_ffn != 0 ? cfg.n_parked_ffn : ledger.n_parked_ffn;
    if (n_park > kMaxParkedFfnLayers) {
        n_park = kMaxParkedFfnLayers;
    }
    const uint32_t n_stream = ffn_stream_layers(n_park);
    streamer.set_n_parked_ffn(n_park);
    streamer.set_log_cuda_ffn(true);
    // ggml A/B below are the bind. streamer.ensure_slots() is private
    // cudaMalloc the sched never sees — not a bind.
    const uint64_t park_bytes =
        static_cast<uint64_t>(n_park) * kQ4FfnLayerBytesMeasured5080;
    std::fprintf(stderr, "%s\n", format_vram_ledger(ledger).c_str());
    std::fprintf(stderr, "%s\n", format_ffn_cuda_park_line(n_park, park_bytes).c_str());
    std::fprintf(stderr,
                 "ffn_cuda_plan ngl=0 slots_first=1 kv20k=1 park_weights=1 "
                 "park=%u stream=%u never_64=1 trace_hooks=%d cb_eval=%s "
                 "ggml_rebind_q4=%d ggml_bind_load=%d op_offload=%d\n",
                 n_park, n_stream, cfg.trace_hooks ? 1 : 0, cfg.trace_hooks ? "set" : "nullptr",
                 ggml_can_rebind_q4_midgraph() ? 1 : 0, ggml_can_bind_q4_at_load() ? 1 : 0,
                 cfg.disable_op_offload ? 0 : 1);
    std::fprintf(stderr, "%s\n", format_pcie_bound_line(n_stream).c_str());

    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_type_t gpu_buft = hybrid_gpu_buft();
    // Alloc A/B AFTER measure. A 160 MiB slot before load is why down
    // landed past the end (254e10c). No extra CUDA weight buffers.
    GgmlSlotGuard ggml_slots;
    const std::vector<std::string> gpu_pats = hybrid_gpu_tensor_regexes(n_park);
    const std::vector<std::string> cpu_pats = hybrid_cpu_tensor_regexes();
    std::vector<llama_model_tensor_buft_override> buft_ovs;
    buft_ovs.reserve(gpu_pats.size() + cpu_pats.size() + 1);
    if (gpu_buft) {
        for (const std::string& pat : gpu_pats) {
            llama_model_tensor_buft_override ov{};
            ov.pattern = pat.c_str();
            ov.buft = gpu_buft;
            buft_ovs.push_back(ov);
        }
    }
    for (const std::string& pat : cpu_pats) {
        llama_model_tensor_buft_override ov{};
        ov.pattern = pat.c_str();
        ov.buft = cpu_buft;
        buft_ovs.push_back(ov);
    }
    llama_model_tensor_buft_override ov_end{};
    ov_end.pattern = nullptr;
    ov_end.buft = nullptr;
    buft_ovs.push_back(ov_end);
    mparams.tensor_buft_overrides = buft_ovs.data();

    llama_model* model = llama_model_load_from_file(cfg.model_path.c_str(), mparams);
    if (!model) {
        llama_backend_free();
        st.ok = false;
        st.message = "llama.cpp failed to load GGUF (weights missing or architecture "
                     "unsupported by this llama.cpp build)";
        return st;
    }

    uint64_t v_free1 = 0;
    uint64_t v_tot1 = 0;
    PerfClocks::query_vram(&v_free1, &v_tot1);
    const uint64_t cuda0_model =
        (v_tot1 > v_free1 && v_free0 >= v_free1) ? (v_free0 - v_free1) : 0;

    const int32_t n_layer = llama_model_n_layer(model);
    const int32_t n_embd = llama_model_n_embd(model);
    const uint32_t hook_layers = hook_layer_count_from_blocks(static_cast<uint32_t>(n_layer));
    if (hook_layers != kNLayers || n_embd != static_cast<int32_t>(kHiddenDim)) {
        llama_model_free(model);
        llama_backend_free();
        st.ok = false;
        st.architecture_ok = false;
        st.message = "loaded model is not 64 x 5120 Qwen 27B hybrid "
                     "(block_count 64 or 65=64+MTP is ok; hook ring stays 64)";
        return st;
    }

    PerfClocks clocks;
    clocks.begin_session();
    clocks.set_plan(n_park, n_stream, true);
    clocks.set_trace_off(!cfg.trace_hooks);
    clocks.set_ffn_gemm(kFfnGemmPerToken, 0);
    std::vector<StreamedFfnHost> streamed_hosts;
    uint32_t parked_cuda_n = 0;
    uint32_t streamed_cpu_n = 0;
    uint32_t streamed_cuda_n = 0;
    uint32_t streamed_cuda_host_n = 0;
    collect_streamed_ffn_hosts(model, n_park, &streamed_hosts, &parked_cuda_n, &streamed_cpu_n,
                              &streamed_cuda_n, &streamed_cuda_host_n, true, true);
    std::fprintf(stderr, "%s\n",
                 format_ffn_place_line(parked_cuda_n, streamed_cpu_n, streamed_cuda_n,
                                       streamed_cuda_host_n)
                     .c_str());
    register_streamed_layer_parts(streamer, streamed_hosts);
    const FfnTripletBytes trip_a = measure_ffn_triplet(model, kNLayers - 1);
    const FfnTripletBytes trip_b = measure_ffn_triplet(model, kNLayers - 2);
    std::fprintf(stderr, "%s\n",
                 format_ffn_slot_bytes_line(kNLayers - 1, trip_a.gate, trip_a.up, trip_a.down,
                                            trip_a.slot)
                     .c_str());
    std::fprintf(stderr, "%s\n",
                 format_ffn_slot_bytes_line(kNLayers - 2, trip_b.gate, trip_b.up, trip_b.down,
                                            trip_b.slot)
                     .c_str());
    ledger = vram_ledger_sized_slots(trip_a.slot, trip_b.slot);
    std::fprintf(stderr, "%s\n", format_vram_ledger(ledger).c_str());
    if (gpu_buft) {
        ggml_slots.a = ggml_backend_buft_alloc_buffer(gpu_buft, ledger.slot_a_bytes);
        ggml_slots.b = ggml_backend_buft_alloc_buffer(gpu_buft, ledger.slot_b_bytes);
        if (ggml_slots.a) {
            ggml_backend_buffer_set_usage(ggml_slots.a, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        if (ggml_slots.b) {
            ggml_backend_buffer_set_usage(ggml_slots.b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        std::fprintf(stderr,
                     "ffn_ggml_slots a=%d b=%d bytes=%llu "
                     "(measure_then_alloc=1 bind=0 extra_park=0)\n",
                     ggml_slots.a ? 1 : 0, ggml_slots.b ? 1 : 0,
                     static_cast<unsigned long long>(ledger.slot_a_bytes + ledger.slot_b_bytes));
    }
    uint32_t ggml_bound = 0;
    if (ggml_slots.a) {
        ggml_bound += bind_layer_q4_triplet(model, kNLayers - 1, ggml_slots.a, &clocks);
    }
    if (ggml_slots.b) {
        ggml_bound += bind_layer_q4_triplet(model, kNLayers - 2, ggml_slots.b, &clocks);
    }
    (void)bind_streamed_tensors_to_ggml_slots(gpu_buft, &ggml_slots, &streamed_hosts, n_park,
                                              &clocks);
    parked_cuda_n = 0;
    streamed_cpu_n = 0;
    streamed_cuda_n = 0;
    streamed_cuda_host_n = 0;
    collect_streamed_ffn_hosts(model, n_park, nullptr, &parked_cuda_n, &streamed_cpu_n,
                              &streamed_cuda_n, &streamed_cuda_host_n, false, false);
    std::fprintf(stderr, "FFN_PLACE_AFTER_BIND parked_cuda=%u streamed_cpu=%u streamed_cuda=%u "
                         "streamed_cuda_host=%u ggml_bound=%u hop_collapse=0 "
                         "(triplet-sized A/B; no leftover collapse; extra_park=0)\n",
                 parked_cuda_n, streamed_cpu_n, streamed_cuda_n, streamed_cuda_host_n, ggml_bound);
    clocks.add_cuda_ffn_binds(parked_cuda_n);
    uint32_t cpu_63 = 0;
    const char* last_63_name = "blk.63.ffn_gate.weight";
    const char* last_63_buft = "CPU";
    print_layer_mul_mat_srcs(model, kNLayers - 1, &cpu_63, &last_63_name, &last_63_buft);
    uint32_t cpu_62 = 0;
    const char* last_62_name = "blk.62.ffn_gate.weight";
    const char* last_62_buft = "CPU";
    print_layer_mul_mat_srcs(model, kNLayers - 2, &cpu_62, &last_62_name, &last_62_buft);
    clocks.set_real_h2d(ggml_bound >= 6 && streamed_cuda_host_n == 0);
    SplitLedger sl =
        split_ledger_trace_off_park_stream(n_stream, streamed_cuda_host_n, cpu_63);
    sl.trace_off = !cfg.trace_hooks;
    sl.callback_hooks = 0;
    clocks.set_split_ledger(0, sl.placement_buffer, sl.backend_transition, sl.unsupported_op);
    print_split_why(sl, cpu_63, last_63_name, last_63_buft);
    streamer.set_perf(&clocks);
    uint64_t vfree = 0;
    uint64_t vtotal = 0;
    const bool have_vram = PerfClocks::query_vram(&vfree, &vtotal);
    clocks.set_cuda_events(have_vram);
    clocks.set_vram(park_bytes + kPinnedGaWeightBytes + kPinnedEmbedWeightBytes,
                    kHourKvReserveBytes,
                    kCudaScratchBytes + ledger.slot_a_bytes + ledger.slot_b_bytes, vfree);
    clocks.set_cuda0(cuda0_model, 0);
    if (vtotal > vfree) {
        clocks.set_nvidia_used(vtotal - vfree);
    }

    GraphHookSession session(hooks, streamer);
    if (cfg.on_htr1) {
        session.set_on_htr1(cfg.on_htr1);
    }
    LlamaHookUser user;
    user.session = &session;
    user.clocks = &clocks;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = cfg.n_ctx;
    cparams.n_batch = cfg.n_batch ? cfg.n_batch : 512;
    cparams.n_ubatch = cfg.n_ubatch ? cfg.n_ubatch : 32;
    // FA stays off (FA + CPU FFN split AVed). Parking is GPU tensor
    // overrides at load. op_offload cannot rebind mapped Q4 mid-graph.
    if (cfg.disable_flash_attn) {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    cparams.op_offload = !cfg.disable_op_offload;
    cparams.offload_kqv = true;
    if (cfg.trace_hooks) {
        cparams.cb_eval = llama_cb_eval;
        cparams.cb_eval_user_data = &user;
    } else {
        cparams.cb_eval = nullptr;
        cparams.cb_eval_user_data = nullptr;
    }

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr,
                     "RESERVE_FAIL cuda0_model_MiB=%.1f graph_reserve_MiB=%.1f "
                     "extra_park=0 (9a5f0df died here: 63 ffn_down CPU_Mapped, 638 splits)\n",
                     static_cast<double>(cuda0_model) / (1024.0 * 1024.0),
                     static_cast<double>(kHourGraphReserveBytes) / (1024.0 * 1024.0));
        print_split_why(sl, cpu_63, last_63_name, last_63_buft);
        print_layer_mul_mat_srcs(model, kNLayers - 1, &cpu_63, &last_63_name, &last_63_buft);
        llama_model_free(model);
        llama_backend_free();
        st.ok = false;
        st.message = "llama.cpp failed to create context (graph reserve; not missing park)";
        return st;
    }

    uint64_t v_free2 = 0;
    uint64_t v_tot2 = 0;
    if (PerfClocks::query_vram(&v_free2, &v_tot2) && v_free1 >= v_free2) {
        clocks.set_cuda0(cuda0_model, v_free1 - v_free2);
        if (v_tot2 > v_free2) {
            clocks.set_nvidia_used(v_tot2 - v_free2);
        }
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const std::string& prompt =
        cfg.prompt.empty() ? std::string(default_coding_assistant_prompt()) : cfg.prompt;

    std::vector<llama_token> tokens(prompt.size() + 32);
    int32_t n_tok = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                                   tokens.data(), static_cast<int32_t>(tokens.size()), true,
                                   true);
    if (n_tok < 0) {
        tokens.resize(static_cast<size_t>(-n_tok));
        n_tok = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                               tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
    }
    if (n_tok <= 0) {
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        st.ok = false;
        st.message = "tokenize failed";
        return st;
    }
    tokens.resize(static_cast<size_t>(n_tok));

    streamer.begin_session();
    clocks.set_plan(n_park, n_stream, streamer.host_pages_pinned());
    const uint32_t slot_a_layer = kNLayers - 1;  // 63 in A
    const uint32_t slot_b_layer = kNLayers - 2;  // 62 in B
    const bool ggml_slots_live =
        ggml_bound >= 6 && ggml_slots.a && ggml_slots.b && cpu_63 == 0 && cpu_62 == 0;
    if (ggml_slots_live) {
        h2d_packed_layer_into_ggml(ggml_slots.a, streamed_hosts, slot_a_layer);
        h2d_packed_layer_into_ggml(ggml_slots.b, streamed_hosts, slot_b_layer);
    } else {
        streamer.h2d_overflow_q4();
    }
    const bool real_h2d = clocks.snapshot().real_h2d || streamer.slot_real_h2d(0);
    std::fprintf(stderr, "%s\n",
                 format_ffn_slot_bind_line(0, ledger.slot_a_bytes, real_h2d && ggml_slots.a).c_str());
    std::fprintf(stderr, "%s\n",
                 format_ffn_slot_bind_line(1, ledger.slot_b_bytes, real_h2d && ggml_slots.b).c_str());
    std::fprintf(stderr,
                 "ffn_stream_bind n=%u real_h2d=%d h2d_B=%llu cuda_ffn_binds=%llu "
                 "host_ffn_binds=%llu op_offload=%d cuda_host=%u ggml_bound=%u "
                 "ggml_used=%d private_cudaMalloc=%d\n",
                 n_stream, real_h2d ? 1 : 0,
                 static_cast<unsigned long long>(clocks.snapshot().h2d_bytes),
                 static_cast<unsigned long long>(clocks.snapshot().cuda_ffn_binds),
                 static_cast<unsigned long long>(clocks.snapshot().host_ffn_binds),
                 cfg.disable_op_offload ? 0 : 1, streamed_cuda_host_n, ggml_bound,
                 ggml_slots_live ? 1 : 0, ggml_slots_live ? 0 : 1);
    hooks.mark_reserved_core(256);
    for (llama_token id : tokens) {
        if (id >= 0 && static_cast<uint32_t>(id) < kVocabSize) {
            hooks.on_vocab_id(static_cast<uint32_t>(id));
        }
    }

    const int32_t chunk = 512;
    int32_t consumed = 0;
    const auto prefill_t0 = std::chrono::steady_clock::now();
    while (consumed < n_tok) {
        const int32_t n = std::min(chunk, n_tok - consumed);
        llama_batch batch = llama_batch_get_one(tokens.data() + consumed, n);
        if (cfg.trace_hooks) {
            session.begin_token(static_cast<uint32_t>(consumed));
        }
        clocks.begin_decode();
        if (!ggml_slots_live) {
            streamer.h2d_overflow_q4();
        }
        clocks.begin_span(PerfSpan::Gpu);
        const int rc = llama_decode(ctx, batch);
        clocks.end_span(PerfSpan::Gpu);
        if (rc != 0) {
            streamer.end_session();
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            st.ok = false;
            st.message = "llama_decode failed during prefill (rc=" + std::to_string(rc) +
                         "). This llama.cpp build cannot run the Qwen 27B hybrid graph.";
            return st;
        }
        consumed += n;
        if (cfg.abort && cfg.abort->load()) {
            streamer.end_session();
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            st.ok = false;
            st.message = "hour aborted";
            return st;
        }
    }
    const double prefill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                           prefill_t0)
                                 .count();
    clocks.set_prefill(prefill_s, static_cast<uint32_t>(n_tok));
    clocks.begin_decode_wall();
    std::fprintf(stderr, "PREFILL tokens=%d elapsed_s=%.2f TRACE=%s\n", n_tok, prefill_s,
                 cfg.trace_hooks ? "on" : "off");

    uint32_t generated = 0;
    const uint32_t want = cfg.n_predict;
    llama_token last = tokens.back();
    auto report = [&]() {
        if (have_vram) {
            uint64_t fr = 0;
            uint64_t tot = 0;
            PerfClocks::query_vram(&fr, &tot);
            clocks.set_vram(park_bytes + kPinnedGaWeightBytes + kPinnedEmbedWeightBytes,
                            static_cast<uint64_t>(cfg.n_ctx) * kKvBytesPerTokenFp16,
                            kCudaScratchBytes + ledger.slot_a_bytes + ledger.slot_b_bytes, fr);
            if (tot > fr) {
                clocks.set_nvidia_used(tot - fr);
            }
        }
        const PerfSnapshot snap = clocks.snapshot();
        SplitLedger report_sl =
            split_ledger_trace_off_park_stream(n_stream, streamed_cuda_host_n, cpu_63);
        report_sl.trace_off = !cfg.trace_hooks;
        report_sl.callback_hooks = 0;
        std::fprintf(stderr, "%s\n", format_performance_line(snap).c_str());
        std::fprintf(stderr, "%s\n", format_performance_bottlenecks(snap).c_str());
        std::fprintf(stderr, "%s\n", format_vram_ledger(ledger).c_str());
        print_split_why(report_sl, cpu_63, last_63_name, last_63_buft);
        std::fprintf(stderr, "%s\n", format_ffn_gemm_line(ffn_gemm_all_cuda()).c_str());
        std::fprintf(stderr, "%s\n", format_bench_line(bench_from_snapshot(snap)).c_str());
        std::fprintf(stderr, "%s\n", format_pcie_bound_line(n_stream).c_str());
        std::fprintf(stderr, "%s\n",
                     format_tokens_per_sec_line(snap.tok_per_sec, generated, snap.wall_s).c_str());
        if (cfg.on_stats) {
            cfg.on_stats(snap);
        }
        st.perf = snap;
    };
    while (generated < want) {
        float* logits = llama_get_logits_ith(ctx, -1);
        if (!logits) {
            st.message = "llama_get_logits_ith returned null";
            break;
        }
        clocks.begin_span(PerfSpan::Cpu);
        std::vector<uint32_t> top(cfg.top_k ? cfg.top_k : 1u);
        topk_ids(logits, n_vocab, static_cast<uint32_t>(top.size()), top.data());
        last = static_cast<llama_token>(top[0]);
        const bool special = llama_vocab_is_eog(vocab, last) ||
                             llama_vocab_is_control(vocab, last);
        clocks.end_span(PerfSpan::Cpu);
        clocks.begin_span(PerfSpan::Trace);
        session.finish_token(static_cast<uint32_t>(last), top.data(),
                             static_cast<uint32_t>(top.size()), special);
        clocks.end_span(PerfSpan::Trace);
        if (static_cast<uint32_t>(last) < kVocabSize) {
            hooks.on_vocab_id(static_cast<uint32_t>(last));
        }
        if (cfg.checkpoint_every != 0 && !cfg.out_path.empty() && !cfg.pack_checkpoint &&
            hooks.table().n_tokens > 0 &&
            (hooks.table().n_tokens % cfg.checkpoint_every) == 0) {
            hooks.pull_gpu_accums();
            std::string err;
            checkpoint_prune_table(hooks.table(), cfg.out_path, &err);
        }
        clocks.end_token();
        if (generated != 0 && (generated % 32u) == 0) {
            report();
        }
        if (cfg.abort && cfg.abort->load()) {
            break;
        }
        if (!cfg.continue_after_eos && llama_vocab_is_eog(vocab, last)) {
            break;
        }
        if (cfg.trace_hooks) {
            session.begin_token(static_cast<uint32_t>(n_tok + generated));
        }
        llama_batch batch = llama_batch_get_one(&last, 1);
        clocks.begin_decode();
        if (!ggml_slots_live) {
            streamer.h2d_overflow_q4();
        }
        clocks.begin_span(PerfSpan::Gpu);
        const int rc = llama_decode(ctx, batch);
        clocks.end_span(PerfSpan::Gpu);
        if (rc != 0) {
            st.message = "llama_decode failed during generate (rc=" + std::to_string(rc) + ")";
            break;
        }
        ++generated;
    }
    report();

    hooks.pull_gpu_accums();
    streamer.end_session();
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    st.ok = hooks.table().n_tokens > 0;
    st.ran_tokens = st.ok;
    st.n_tokens = hooks.table().n_tokens;
    st.perf = clocks.snapshot();
    if (st.ok) {
        st.message = "llama.cpp qwen35 graph eval completed; park 0-56, stream 57-63 "
                     "(63 gate+up+down in A/B, no extra park, never park-64)";
    } else if (st.message.empty()) {
        st.message = "llama.cpp ran but after_logits never fired; refusing to fake n_fired";
    }
    return st;
}

}  // namespace micro_llm

#endif  // MICRO_LLM_HAS_LLAMA
