#ifdef MICRO_LLM_HAS_WEBVIEW2

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif

#include <windows.h>
#include <wrl.h>
#include <WebView2.h>

#include "micro_llm/hook_ring.hpp"
#include "micro_llm/hotspot_ui.hpp"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using Microsoft::WRL::Callback;
using Microsoft::WRL::ComPtr;

namespace {

constexpr UINT WM_HTR1 = WM_APP + 41;
constexpr UINT WM_HOST_JSON = WM_APP + 42;
constexpr UINT WM_HOUR_DONE = WM_APP + 43;

ComPtr<ICoreWebView2Controller> g_controller;
ComPtr<ICoreWebView2> g_webview;
ComPtr<ICoreWebView2Environment> g_env;
ComPtr<ICoreWebView2SharedBuffer> g_shared;
HWND g_hwnd = nullptr;

std::mutex g_mu;
std::deque<std::vector<uint8_t>> g_htr1;
std::deque<std::wstring> g_json;
std::atomic<bool> g_page_ready{false};
std::atomic<int> g_hour_rc{0};

std::wstring utf8_to_wide(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring out(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), out.data(), n);
    return out;
}

std::wstring file_url(const std::wstring& dir) {
    std::wstring path = dir;
    for (auto& ch : path) {
        if (ch == L'\\') ch = L'/';
    }
    if (!path.empty() && path.back() != L'/') path += L'/';
    path += L"index.html";
    if (path.size() >= 2 && path[1] == L':') {
        return L"file:///" + path;
    }
    return L"file://" + path;
}

void post_to_hwnd(UINT msg) {
    if (g_hwnd) {
        PostMessageW(g_hwnd, msg, 0, 0);
    }
}

void ensure_shared_buffer() {
    if (g_shared || !g_env) {
        return;
    }
    ComPtr<ICoreWebView2Environment12> env12;
    if (FAILED(g_env.As(&env12))) {
        return;
    }
    env12->CreateSharedBuffer(micro_llm::kHtr1RecordBytes, &g_shared);
}

void post_one_htr1(const uint8_t* rec, size_t nbytes) {
    if (!g_webview || !rec || nbytes < micro_llm::kHtr1RecordBytes) {
        return;
    }
    ensure_shared_buffer();
    ComPtr<ICoreWebView2_17> wv17;
    if (g_shared && SUCCEEDED(g_webview.As(&wv17))) {
        BYTE* dest = nullptr;
        if (SUCCEEDED(g_shared->get_Buffer(&dest)) && dest) {
            std::memcpy(dest, rec, micro_llm::kHtr1RecordBytes);
            wv17->PostSharedBufferToScript(g_shared, COREWEBVIEW2_SHARED_BUFFER_ACCESS_READ_ONLY,
                                           L"{\"type\":\"htr1\"}");
        }
    }
    // Always also post a host message so chrome.webview 'message' fires.
    g_webview->PostWebMessageAsJson(L"{\"type\":\"htr1\"}");
}

void drain_host_queue() {
    if (!g_webview || !g_page_ready.load()) {
        return;
    }
    std::vector<std::vector<uint8_t>> recs;
    std::vector<std::wstring> jsons;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        while (!g_htr1.empty()) {
            recs.push_back(std::move(g_htr1.front()));
            g_htr1.pop_front();
        }
        while (!g_json.empty()) {
            jsons.push_back(std::move(g_json.front()));
            g_json.pop_front();
        }
    }
    for (const auto& js : jsons) {
        g_webview->PostWebMessageAsJson(js.c_str());
    }
    for (const auto& rec : recs) {
        post_one_htr1(rec.data(), rec.size());
    }
}

LRESULT CALLBACK hotspot_wndproc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    switch (msg) {
        case WM_SIZE:
            if (g_controller) {
                RECT rc{};
                GetClientRect(hwnd, &rc);
                g_controller->put_Bounds(rc);
            }
            return 0;
        case WM_HTR1:
        case WM_HOST_JSON:
            drain_host_queue();
            return 0;
        case WM_HOUR_DONE:
            return 0;
        case WM_DESTROY:
            micro_llm::hotspot_live_abort().store(true);
            g_hwnd = nullptr;
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProcW(hwnd, msg, wparam, lparam);
    }
}

void show_runtime_error() {
    MessageBoxW(nullptr,
                L"The hotspot window needs the Microsoft Edge WebView2 Runtime (Evergreen).\n\n"
                L"It usually ships with Windows 11 and Microsoft Edge.\n"
                L"If this machine is missing it, install the Evergreen bootstrapper:\n"
                L"https://go.microsoft.com/fwlink/p/?LinkId=2124703\n\n"
                L"This exe hosts committed static files. Do not run npm.",
                L"micro-llm hotspot", MB_OK | MB_ICONERROR);
}

}  // namespace

void micro_llm_hotspot_win32_push_htr1(const uint8_t* rec, size_t nbytes) {
    if (!rec || nbytes == 0) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(g_mu);
        g_htr1.emplace_back(rec, rec + nbytes);
        while (g_htr1.size() > micro_llm::kHtr1RingDepth) {
            g_htr1.pop_front();
        }
    }
    post_to_hwnd(WM_HTR1);
}

void micro_llm_hotspot_win32_push_json(const std::string& json_utf8) {
    if (json_utf8.empty()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(g_mu);
        g_json.push_back(utf8_to_wide(json_utf8));
    }
    post_to_hwnd(WM_HOST_JSON);
}

int micro_llm_run_hotspot_ui_win32(const micro_llm::HotspotUiOptions& opt,
                                   const std::string& ui_dir) {
    if (!opt.live) {
        FreeConsole();
    } else {
        micro_llm::hotspot_live_abort().store(false);
    }

    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    const bool did_init = SUCCEEDED(hr);
    if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
        show_runtime_error();
        return 1;
    }

    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = hotspot_wndproc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
    wc.lpszClassName = L"MicroLLMHotspot";
    RegisterClassExW(&wc);

    const std::string title_utf8 =
        opt.title.empty()
            ? (opt.live ? std::string("micro-llm hotspot — live hour")
                        : std::string("micro-llm hotspot — sample ring"))
            : opt.title;
    const std::wstring title = utf8_to_wide(title_utf8);

    HWND hwnd = CreateWindowExW(0, wc.lpszClassName, title.c_str(), WS_OVERLAPPEDWINDOW,
                                CW_USEDEFAULT, CW_USEDEFAULT, 1440, 900, nullptr, nullptr,
                                wc.hInstance, nullptr);
    if (!hwnd) {
        if (did_init) CoUninitialize();
        return 1;
    }
    g_hwnd = hwnd;
    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    wchar_t tmp[MAX_PATH];
    GetTempPathW(MAX_PATH, tmp);
    std::wstring user_data = std::wstring(tmp) + L"micro-llm-hotspot";
    const std::wstring ui_wide = utf8_to_wide(ui_dir);
    const std::wstring fallback = file_url(ui_wide);
    const bool live = opt.live;
    std::function<int()> run_hour = opt.run_hour;
    std::thread worker;
    std::atomic<bool> worker_started{false};

    auto start_hour = [&]() {
        if (!live || !run_hour || worker_started.exchange(true)) {
            return;
        }
        HWND host = hwnd;
        worker = std::thread([run_hour, host]() {
            const int rc = run_hour();
            g_hour_rc.store(rc);
            if (host) {
                PostMessageW(host, WM_HOUR_DONE, static_cast<WPARAM>(rc), 0);
            }
        });
    };

    hr = CreateCoreWebView2EnvironmentWithOptions(
        nullptr, user_data.c_str(), nullptr,
        Callback<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler>(
            [hwnd, ui_wide, fallback, live, start_hour](HRESULT result,
                                                        ICoreWebView2Environment* env) -> HRESULT {
                if (FAILED(result) || !env) {
                    show_runtime_error();
                    DestroyWindow(hwnd);
                    return result;
                }
                g_env = env;
                return env->CreateCoreWebView2Controller(
                    hwnd,
                    Callback<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler>(
                        [hwnd, ui_wide, fallback, live, start_hour](
                            HRESULT result, ICoreWebView2Controller* controller) -> HRESULT {
                            if (FAILED(result) || !controller) {
                                show_runtime_error();
                                DestroyWindow(hwnd);
                                return result;
                            }
                            g_controller = controller;
                            controller->get_CoreWebView2(&g_webview);
                            RECT rc{};
                            GetClientRect(hwnd, &rc);
                            controller->put_Bounds(rc);

                            ComPtr<ICoreWebView2Controller2> controller2;
                            if (SUCCEEDED(controller->QueryInterface(IID_PPV_ARGS(&controller2)))) {
                                COREWEBVIEW2_COLOR bg{255, 4, 5, 10};
                                controller2->put_DefaultBackgroundColor(bg);
                            }

                            ComPtr<ICoreWebView2Settings> settings;
                            if (SUCCEEDED(g_webview->get_Settings(&settings))) {
                                settings->put_IsStatusBarEnabled(FALSE);
                                settings->put_AreDefaultContextMenusEnabled(TRUE);
                                settings->put_AreDevToolsEnabled(FALSE);
                            }

                            g_webview->add_NavigationCompleted(
                                Callback<ICoreWebView2NavigationCompletedEventHandler>(
                                    [live, start_hour](ICoreWebView2*,
                                                       ICoreWebView2NavigationCompletedEventArgs* args)
                                        -> HRESULT {
                                        BOOL ok = TRUE;
                                        if (args) {
                                            args->get_IsSuccess(&ok);
                                        }
                                        if (ok) {
                                            g_page_ready.store(true);
                                            drain_host_queue();
                                            if (live) {
                                                start_hour();
                                            }
                                        }
                                        return S_OK;
                                    })
                                    .Get(),
                                nullptr);

                            ComPtr<ICoreWebView2_3> wv3;
                            if (SUCCEEDED(g_webview.As(&wv3))) {
                                const HRESULT map = wv3->SetVirtualHostNameToFolderMapping(
                                    L"hotspot.local", ui_wide.c_str(),
                                    COREWEBVIEW2_HOST_RESOURCE_ACCESS_KIND_ALLOW);
                                if (SUCCEEDED(map)) {
                                    g_webview->Navigate(L"https://hotspot.local/index.html");
                                    return S_OK;
                                }
                            }
                            g_webview->Navigate(fallback.c_str());
                            return S_OK;
                        })
                        .Get());
            })
            .Get());

    if (FAILED(hr)) {
        show_runtime_error();
        DestroyWindow(hwnd);
        if (did_init) CoUninitialize();
        return 1;
    }

    MSG msg{};
    while (GetMessageW(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    if (worker.joinable()) {
        worker.join();
    }

    g_page_ready.store(false);
    g_shared.Reset();
    g_webview.Reset();
    g_controller.Reset();
    g_env.Reset();
    g_hwnd = nullptr;
    if (did_init) CoUninitialize();
    if (live) {
        return g_hour_rc.load();
    }
    return static_cast<int>(msg.wParam);
}

#endif
