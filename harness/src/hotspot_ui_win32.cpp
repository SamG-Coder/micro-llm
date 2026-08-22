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

#include <WebView2.h>
#include <windows.h>
#include <wrl.h>

#include <string>

using Microsoft::WRL::Callback;
using Microsoft::WRL::ComPtr;

namespace {

ComPtr<ICoreWebView2Controller> g_controller;
ComPtr<ICoreWebView2> g_webview;

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

LRESULT CALLBACK hotspot_wndproc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    switch (msg) {
        case WM_SIZE:
            if (g_controller) {
                RECT rc{};
                GetClientRect(hwnd, &rc);
                g_controller->put_Bounds(rc);
            }
            return 0;
        case WM_DESTROY:
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

int micro_llm_run_hotspot_ui_win32(const std::string& ui_dir) {
    FreeConsole();

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

    HWND hwnd = CreateWindowExW(
        0, wc.lpszClassName, L"micro-llm hotspot — sample ring",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 1440, 900, nullptr, nullptr,
        wc.hInstance, nullptr);
    if (!hwnd) {
        if (did_init) CoUninitialize();
        return 1;
    }
    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    wchar_t tmp[MAX_PATH];
    GetTempPathW(MAX_PATH, tmp);
    std::wstring user_data = std::wstring(tmp) + L"micro-llm-hotspot";
    const std::wstring ui_wide = utf8_to_wide(ui_dir);
    const std::wstring fallback = file_url(ui_wide);

    hr = CreateCoreWebView2EnvironmentWithOptions(
        nullptr, user_data.c_str(), nullptr,
        Callback<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler>(
            [hwnd, ui_wide, fallback](HRESULT result, ICoreWebView2Environment* env) -> HRESULT {
                if (FAILED(result) || !env) {
                    show_runtime_error();
                    DestroyWindow(hwnd);
                    return result;
                }
                return env->CreateCoreWebView2Controller(
                    hwnd,
                    Callback<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler>(
                        [hwnd, ui_wide, fallback](HRESULT result,
                                                  ICoreWebView2Controller* controller) -> HRESULT {
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

    g_webview.Reset();
    g_controller.Reset();
    if (did_init) CoUninitialize();
    return static_cast<int>(msg.wParam);
}

#endif
