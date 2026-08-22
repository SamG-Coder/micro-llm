#pragma once

// Host the committed hotspot map (harness/ui) in a native window.
// Windows: WebView2. --ui alone is the sample ring (no GGUF).
// --model opens the live hour window and posts one HTR1 record per token.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace micro_llm {

struct HotspotUiOptions {
    bool check_only = false;
    bool live = false;
    std::string title;
    // Invoked on a worker thread (Windows) or this thread (no WebView2)
    // once the page can receive records. Return the hour exit code.
    std::function<int()> run_hour;
};

// Directory that contains index.html. Empty if the committed UI is missing.
std::string hotspot_ui_dir(std::string* err = nullptr);

// Open the sample map, or the live hour window when opt.live.
int run_hotspot_ui(const HotspotUiOptions& opt = {});

// Thread-safe. Queue one HTR1 record for the page (Windows WebView2).
void hotspot_live_push_htr1(const uint8_t* rec, size_t nbytes);

// Thread-safe. JSON object posted as a WebView2 host message.
void hotspot_live_push_json(const std::string& json_utf8);

// Set when the live window is closing so the worker can leave llama_decode.
std::atomic<bool>& hotspot_live_abort();

}  // namespace micro_llm
