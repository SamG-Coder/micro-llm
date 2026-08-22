#pragma once

// Host the committed sample hotspot map (harness/ui) in a native window.
// Windows: WebView2. Does not load a GGUF and does not start the hour.

#include <string>

namespace micro_llm {

struct HotspotUiOptions {
    bool check_only = false;
};

// Directory that contains index.html. Empty if the committed UI is missing.
std::string hotspot_ui_dir(std::string* err = nullptr);

// Open the sample map. On non-Windows, prints the UI path and returns 0
// unless check_only is set (then it only verifies files).
int run_hotspot_ui(const HotspotUiOptions& opt = {});

}  // namespace micro_llm
