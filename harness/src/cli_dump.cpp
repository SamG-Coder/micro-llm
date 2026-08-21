#include "micro_llm/micro_llm.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s [--out PATH] [--tokens N] [--eps F]\n"
                 "Dump one prune table from synthetic hook traffic.\n",
                 argv0);
}

}  // namespace

int main(int argc, char** argv) {
    std::string out = "prune_table.bin";
    uint32_t n_tokens = 8;
    float eps = micro_llm::kDefaultFireEps;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out = argv[++i];
        } else if (std::strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            n_tokens = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "--eps") == 0 && i + 1 < argc) {
            eps = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    using namespace micro_llm;

    StreamerConfig scfg;
    scfg.ffn_scratch_bytes = 4096;  // synthetic: do not allocate 300MB
    scfg.pin_host_pages = true;
    TraceStreamer streamer(scfg);
    streamer.begin_session();

    TraceHooks hooks(eps, eps);
    hooks.mark_reserved_core(16);

    std::vector<float> gate(kFfnIntermediate, 0.f);
    std::vector<float> up(kFfnIntermediate, 0.f);
    std::vector<float> hin(32, 0.f);
    std::vector<float> hout(32, 0.f);

    const uint32_t prompt_ids[] = {1, 17, 99, 1000};
    for (uint32_t id : prompt_ids) {
        hooks.on_vocab_id(id);
    }

    for (uint32_t t = 0; t < n_tokens; ++t) {
        streamer.leave_logits();
        hooks.begin_token(t);

        for (uint32_t layer = 0; layer < kNLayers; ++layer) {
            const uint32_t next = layer + 1;
            if (next < kNLayers) {
                streamer.prefetch_ffn(next);
            }
            streamer.bind_ffn(layer);

            std::fill(gate.begin(), gate.end(), 0.f);
            std::fill(up.begin(), up.end(), 0.f);
            // Fire a few channels from the live (synthetic) activations.
            const uint32_t ch0 = (layer * 17u + t * 3u) % kFfnIntermediate;
            const uint32_t ch1 = (ch0 + 100u) % kFfnIntermediate;
            gate[ch0] = 2.0f;
            up[ch0] = 1.5f;
            if (t != 3) {
                gate[ch1] = 1.0f;
                up[ch1] = 0.5f;
            }
            hooks.on_ffn_activations(layer, gate.data(), up.data());
            streamer.evict_ffn(layer);
        }

        if (is_delta_net_layer(t % kNLayers) || true) {
            // Packs 0..3 spike; 4..47 stay dead. Pack 2 spikes once (identity
            // average would still not be dead).
            for (uint32_t p = 0; p < 4; ++p) {
                if (p == 2 && t != 0) {
                    hooks.on_delta_residual(p, 0.f);  // identity this token
                } else {
                    std::fill(hin.begin(), hin.end(), 0.1f);
                    std::fill(hout.begin(), hout.end(), 0.1f);
                    hout[0] = 0.1f + 0.5f * static_cast<float>(p + 1);
                    hooks.on_delta_hidden(p, hin.data(), hout.data(),
                                          static_cast<uint32_t>(hin.size()));
                }
            }
        }

        const uint32_t sampled = 2000 + t;
        hooks.on_vocab_id(sampled);
        const uint32_t topk[] = {3000 + t, 4000 + t, 5};
        hooks.on_topk_ids(topk, 3);

        streamer.enter_logits();
        const bool special_or_high_loss = (t == 0) || (t == 5);
        hooks.after_logits(t, special_or_high_loss);
        streamer.leave_logits();
    }

    streamer.end_session();

    std::string err;
    if (!save_prune_table(hooks.table(), out, &err)) {
        std::fprintf(stderr, "save failed: %s\n", err.c_str());
        return 1;
    }

    uint64_t n_fired_ch = 0;
    uint64_t n_floor = 0;
    uint64_t n_vocab = 0;
    uint32_t n_dead_packs = 0;
    uint32_t n_spike_packs = 0;
    for (uint32_t L = 0; L < kNLayers; ++L) {
        for (uint32_t c = 0; c < kFfnIntermediate; ++c) {
            if (hooks.table().channel(L, c).n_fired > 0) {
                ++n_fired_ch;
            }
            if (hooks.table().floor_keep(L, c)) {
                ++n_floor;
            }
        }
    }
    for (uint32_t p = 0; p < kNDeltaNetPacks; ++p) {
        if (hooks.table().pack_is_dead(p)) {
            ++n_dead_packs;
        } else {
            ++n_spike_packs;
        }
    }
    for (uint32_t id = 0; id < kVocabSize; ++id) {
        if (hooks.table().vocab_seen(id)) {
            ++n_vocab;
        }
    }

    std::printf("wrote %s\n", out.c_str());
    std::printf("tokens=%llu fired_channels=%llu floor_channels=%llu\n",
                static_cast<unsigned long long>(hooks.table().n_tokens),
                static_cast<unsigned long long>(n_fired_ch),
                static_cast<unsigned long long>(n_floor));
    std::printf("packs spike=%u dead=%u vocab_ids=%llu\n", n_spike_packs, n_dead_packs,
                static_cast<unsigned long long>(n_vocab));
    std::printf("backend=%s tensor_align=%u pack_ids=0..%u\n",
                ffn_reduce_cuda_available() ? "cuda" : "cpu", kTensorAlign,
                kNDeltaNetPacks - 1);
    std::printf("lm_head_pinned_after_session=%d mid_shrink_allowed=%d peak_ffn=%zu\n",
                streamer.lm_head_resident() ? 1 : 0,
                streamer.mid_session_shrink_allowed() ? 1 : 0, streamer.peak_ffn_bytes());
    return 0;
}
