#include "micro_llm/hotspot_ui.hpp"

#include <cstring>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <shellapi.h>

int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    micro_llm::HotspotUiOptions opt;
    for (int i = 1; i < argc; ++i) {
        if (argv && lstrcmpW(argv[i], L"--ui-check") == 0) {
            opt.check_only = true;
        }
    }
    if (argv) LocalFree(argv);
    return micro_llm::run_hotspot_ui(opt);
}
#else
int main(int argc, char** argv) {
    micro_llm::HotspotUiOptions opt;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--ui-check") == 0) {
            opt.check_only = true;
        }
    }
    return micro_llm::run_hotspot_ui(opt);
}
#endif
