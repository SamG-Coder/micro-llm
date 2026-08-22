#include "micro_llm/hotspot_ui.hpp"

#include <atomic>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

#ifdef MICRO_LLM_HAS_WEBVIEW2
int micro_llm_run_hotspot_ui_win32(const micro_llm::HotspotUiOptions& opt,
                                   const std::string& ui_dir);
void micro_llm_hotspot_win32_push_htr1(const uint8_t* rec, size_t nbytes);
void micro_llm_hotspot_win32_push_json(const std::string& json_utf8);
#endif

namespace micro_llm {
namespace {

#ifdef MICRO_LLM_UI_SOURCE_DIR
constexpr const char* kSourceUiDir = MICRO_LLM_UI_SOURCE_DIR;
#else
constexpr const char* kSourceUiDir = "";
#endif

std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    const char last = a.back();
    if (last == '/' || last == '\\') return a + b;
#ifdef _WIN32
    return a + "\\" + b;
#else
    return a + "/" + b;
#endif
}

bool is_dir_with_index(const std::string& dir) {
    if (dir.empty()) return false;
    const std::string index = join_path(dir, "index.html");
    std::ifstream in(index);
    return in.good();
}

std::string executable_dir() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return {};
    std::wstring w(buf, n);
    const auto slash = w.find_last_of(L"\\/");
    if (slash == std::wstring::npos) return {};
    w.resize(slash);
    if (w.empty()) return {};
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()),
                                          nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<size_t>(bytes), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), out.data(), bytes,
                        nullptr, nullptr);
    return out;
#else
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return {};
    buf[n] = '\0';
    std::string path(buf, static_cast<size_t>(n));
    const auto slash = path.find_last_of('/');
    if (slash == std::string::npos) return {};
    return path.substr(0, slash);
#endif
}

void add_candidate(std::vector<std::string>& out, const std::string& dir) {
    if (dir.empty()) return;
    for (const auto& existing : out) {
        if (existing == dir) return;
    }
    out.push_back(dir);
}

}  // namespace

std::string hotspot_ui_dir(std::string* err) {
    std::vector<std::string> candidates;
    const std::string exe = executable_dir();
    add_candidate(candidates, join_path(exe, "ui"));
    add_candidate(candidates, join_path(join_path(exe, ".."), "ui"));
    if (kSourceUiDir[0] != '\0') {
        add_candidate(candidates, kSourceUiDir);
    }
    add_candidate(candidates, "harness/ui");
    add_candidate(candidates, "ui");

    for (const auto& dir : candidates) {
        if (is_dir_with_index(dir)) {
            if (err) err->clear();
            return dir;
        }
    }
    if (err) {
        *err = "committed hotspot UI not found (expected ui/index.html next to the exe, "
               "or harness/ui in the source tree)";
    }
    return {};
}

HookRing& hotspot_live_ring() {
    static HookRing ring;
    return ring;
}

void hotspot_live_push_htr1(const uint8_t* rec, size_t nbytes) {
    if (rec && nbytes >= kHtr1RecordBytes) {
        hotspot_live_ring().push(rec);
    }
#ifdef MICRO_LLM_HAS_WEBVIEW2
    // Win32 window polls the ring at 60Hz. Do not PostWebMessage here.
    (void)nbytes;
#else
    (void)nbytes;
#endif
}

std::atomic<bool>& hotspot_live_abort() {
    static std::atomic<bool> abort{false};
    return abort;
}

void hotspot_live_push_json(const std::string& json_utf8) {
#ifdef MICRO_LLM_HAS_WEBVIEW2
    micro_llm_hotspot_win32_push_json(json_utf8);
#else
    (void)json_utf8;
#endif
}

int run_hotspot_ui(const HotspotUiOptions& opt) {
    std::string err;
    const std::string dir = hotspot_ui_dir(&err);
    if (dir.empty()) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 2;
    }
    std::printf("hotspot ui: %s\n", dir.c_str());
    if (opt.check_only) {
        return 0;
    }
#ifdef MICRO_LLM_HAS_WEBVIEW2
    return micro_llm_run_hotspot_ui_win32(opt, dir);
#else
    std::fprintf(stderr,
                 "micro-llm-trace: the native hotspot window is Windows + WebView2.\n"
                 "Evergreen runtime: https://go.microsoft.com/fwlink/p/?LinkId=2124703\n"
                 "Static files are already committed; do not run npm to open the map.\n");
    if (opt.live && opt.run_hour) {
        return opt.run_hour();
    }
    return 0;
#endif
}

}  // namespace micro_llm
