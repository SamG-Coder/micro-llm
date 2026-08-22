cmake_minimum_required(VERSION 3.16)

# CUDA is optional and guarded. CPU tests must configure without nvcc.
project(micro_llm_harness LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MICRO_LLM_CUDA "Build the CUDA FFN-reduce kernel (requires nvcc)" OFF)
option(MICRO_LLM_LLAMA "Link llama.cpp for a real Qwen 27B live forward" OFF)
set(MICRO_LLM_LLAMA_DIR "" CACHE PATH "Path to a llama.cpp checkout (optional)")
set(WEBVIEW2_SDK_PATH "" CACHE PATH "Microsoft.Web.WebView2 nupkg extract (Windows)")

function(micro_llm_set_warnings tgt)
  if(MSVC)
    target_compile_options(${tgt} PRIVATE /W3 /utf-8 /permissive-)
  else()
    target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endfunction()

set(MICRO_LLM_HAS_CUDA OFF)
if(MICRO_LLM_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(MICRO_LLM_HAS_CUDA ON)
    message(STATUS "micro-llm: CUDA kernel enabled (${CMAKE_CUDA_COMPILER})")
  else()
    message(WARNING "MICRO_LLM_CUDA=ON but nvcc was not found; using CPU fallback")
  endif()
endif()

add_library(micro_llm
  src/prune_table.cpp
  src/trace_hooks.cpp
  src/streamer.cpp
  src/ffn_reduce.cpp
  src/gguf_meta.cpp
  src/serve.cpp
  src/graph_hooks.cpp
  src/live_forward.cpp
  src/llama_forward.cpp
  src/hook_ring.cpp
  src/trace_cli.cpp
)
target_include_directories(micro_llm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(micro_llm PUBLIC cxx_std_17)
micro_llm_set_warnings(micro_llm)

if(MICRO_LLM_HAS_CUDA)
  target_sources(micro_llm PRIVATE src/ffn_reduce.cu)
  target_compile_definitions(micro_llm PUBLIC MICRO_LLM_HAS_CUDA=1)
  set_target_properties(micro_llm PROPERTIES CUDA_STANDARD 17)
endif()

set(MICRO_LLM_HAS_LLAMA OFF)
if(MICRO_LLM_LLAMA)
  if(MICRO_LLM_LLAMA_DIR)
    if(NOT EXISTS "${MICRO_LLM_LLAMA_DIR}/CMakeLists.txt")
      message(FATAL_ERROR "MICRO_LLM_LLAMA_DIR=${MICRO_LLM_LLAMA_DIR} is not a llama.cpp tree")
    endif()
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
    add_subdirectory(${MICRO_LLM_LLAMA_DIR} ${CMAKE_BINARY_DIR}/llama.cpp EXCLUDE_FROM_ALL)
    target_link_libraries(micro_llm PRIVATE llama)
    set(MICRO_LLM_HAS_LLAMA ON)
  else()
    find_package(llama CONFIG QUIET)
    if(llama_FOUND)
      target_link_libraries(micro_llm PRIVATE llama)
      set(MICRO_LLM_HAS_LLAMA ON)
    else()
      message(FATAL_ERROR
        "MICRO_LLM_LLAMA=ON but llama.cpp was not found. "
        "Set -DMICRO_LLM_LLAMA_DIR=/path/to/llama.cpp")
    endif()
  endif()
endif()
if(MICRO_LLM_HAS_LLAMA)
  target_compile_definitions(micro_llm PUBLIC MICRO_LLM_HAS_LLAMA=1)
  message(STATUS "micro-llm: llama.cpp live forward enabled")
else()
  message(STATUS "micro-llm: llama.cpp not linked (probe-only 27B backend)")
endif()

set(MICRO_LLM_UI_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ui")
if(NOT EXISTS "${MICRO_LLM_UI_SOURCE_DIR}/index.html")
  message(FATAL_ERROR
    "Committed hotspot UI missing at ${MICRO_LLM_UI_SOURCE_DIR}/index.html. "
    "These static files are part of the tree; do not run npm to build the exe.")
endif()

add_library(micro_llm_ui STATIC src/hotspot_ui.cpp)
target_include_directories(micro_llm_ui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(micro_llm_ui PUBLIC cxx_std_17)
target_compile_definitions(micro_llm_ui PRIVATE
  MICRO_LLM_UI_SOURCE_DIR="${MICRO_LLM_UI_SOURCE_DIR}")
micro_llm_set_warnings(micro_llm_ui)

set(MICRO_LLM_HAS_WEBVIEW2 OFF)
if(WIN32)
  set(_wv2_root "")
  if(WEBVIEW2_SDK_PATH AND EXISTS "${WEBVIEW2_SDK_PATH}/build/native/include/WebView2.h")
    set(_wv2_root "${WEBVIEW2_SDK_PATH}")
  else()
    include(FetchContent)
    FetchContent_Declare(microsoft_webview2
      URL https://www.nuget.org/api/v2/package/Microsoft.Web.WebView2/1.0.3179.45
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_GetProperties(microsoft_webview2)
    if(NOT microsoft_webview2_POPULATED)
      FetchContent_Populate(microsoft_webview2)
    endif()
    if(EXISTS "${microsoft_webview2_SOURCE_DIR}/build/native/include/WebView2.h")
      set(_wv2_root "${microsoft_webview2_SOURCE_DIR}")
    endif()
  endif()

  if(_wv2_root)
    target_sources(micro_llm_ui PRIVATE src/hotspot_ui_win32.cpp)
    target_include_directories(micro_llm_ui PRIVATE "${_wv2_root}/build/native/include")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "[Aa][Rr][Mm]64")
      set(_wv2_arch arm64)
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(_wv2_arch x64)
    else()
      set(_wv2_arch x86)
    endif()
    set(_wv2_lib "${_wv2_root}/build/native/${_wv2_arch}/WebView2Loader.dll.lib")
    if(NOT EXISTS "${_wv2_lib}")
      message(FATAL_ERROR "WebView2 loader missing: ${_wv2_lib}")
    endif()
    target_link_libraries(micro_llm_ui PRIVATE
      "${_wv2_lib}" ole32 oleaut32 user32 gdi32 shlwapi advapi32 version shell32)
    target_compile_definitions(micro_llm_ui PRIVATE MICRO_LLM_HAS_WEBVIEW2=1 UNICODE _UNICODE)
    set(MICRO_LLM_WEBVIEW2_DLL
        "${_wv2_root}/build/native/${_wv2_arch}/WebView2Loader.dll")
    set(MICRO_LLM_HAS_WEBVIEW2 ON)
    message(STATUS "micro-llm: WebView2 host enabled (${_wv2_arch})")
  else()
    message(WARNING
      "Windows build but WebView2 SDK was not found. "
      "Set -DWEBVIEW2_SDK_PATH=... or allow FetchContent. "
      "The exe will not open a native window.")
  endif()
endif()

add_executable(micro-llm-dump src/cli_dump.cpp)
target_link_libraries(micro-llm-dump PRIVATE micro_llm)
micro_llm_set_warnings(micro-llm-dump)

add_executable(micro-llm-trace src/cli_trace.cpp)
target_link_libraries(micro-llm-trace PRIVATE micro_llm micro_llm_ui)
micro_llm_set_warnings(micro-llm-trace)

add_executable(micro-llm-view src/cli_view.cpp)
target_link_libraries(micro-llm-view PRIVATE micro_llm_ui)
micro_llm_set_warnings(micro-llm-view)
if(WIN32)
  set_target_properties(micro-llm-view PROPERTIES WIN32_EXECUTABLE TRUE)
endif()

function(micro_llm_copy_ui tgt)
  add_custom_command(TARGET ${tgt} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${MICRO_LLM_UI_SOURCE_DIR}"
            "$<TARGET_FILE_DIR:${tgt}>/ui"
    COMMENT "Copy committed hotspot UI next to ${tgt}")
  if(MICRO_LLM_HAS_WEBVIEW2 AND MICRO_LLM_WEBVIEW2_DLL)
    add_custom_command(TARGET ${tgt} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${MICRO_LLM_WEBVIEW2_DLL}"
              "$<TARGET_FILE_DIR:${tgt}>/WebView2Loader.dll")
  endif()
endfunction()

micro_llm_copy_ui(micro-llm-trace)
micro_llm_copy_ui(micro-llm-view)

add_executable(micro_llm_tests
  tests/test_main.cpp
  tests/test_serialize.cpp
  tests/test_pack_id.cpp
  tests/test_channel_align.cpp
  tests/test_bitset_floor.cpp
  tests/test_dead_spike.cpp
  tests/test_serve_ok.cpp
  tests/test_layer_hooked.cpp
  tests/test_relative_spike.cpp
  tests/test_device_tap.cpp
  tests/test_live_forward.cpp
  tests/test_graph_hooks.cpp
  tests/test_gguf_serve.cpp
  tests/test_hotspot_ui.cpp
  tests/test_hook_ring.cpp
  tests/test_hour_cli.cpp
)
target_include_directories(micro_llm_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(micro_llm_tests PRIVATE micro_llm micro_llm_ui)
micro_llm_set_warnings(micro_llm_tests)

enable_testing()
add_test(NAME micro_llm_tests COMMAND micro_llm_tests)
add_test(NAME micro_llm_dump_smoke
         COMMAND micro-llm-dump --out ${CMAKE_CURRENT_BINARY_DIR}/smoke_prune.bin --tokens 2)
add_test(NAME micro_llm_trace_no_weights
         COMMAND micro-llm-trace --model ${CMAKE_CURRENT_BINARY_DIR}/missing.gguf --out ${CMAKE_CURRENT_BINARY_DIR}/no.bin)
set_tests_properties(micro_llm_trace_no_weights PROPERTIES WILL_FAIL TRUE)
add_test(NAME micro_llm_trace_stub
         COMMAND micro-llm-trace --stub --n-predict 2 --out ${CMAKE_CURRENT_BINARY_DIR}/stub_prune.bin)
add_test(NAME micro_llm_trace_ui_check
         COMMAND micro-llm-trace --ui-check)
add_test(NAME micro_llm_trace_ui_keeps_model
         COMMAND micro-llm-trace --ui --model ${CMAKE_CURRENT_BINARY_DIR}/missing.gguf --out ${CMAKE_CURRENT_BINARY_DIR}/ui_model.bin)
set_tests_properties(micro_llm_trace_ui_keeps_model PROPERTIES WILL_FAIL TRUE)
add_test(NAME micro_llm_view_ui_check
         COMMAND micro-llm-view --ui-check)

install(TARGETS micro_llm micro-llm-dump micro-llm-trace micro-llm-view
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/micro_llm DESTINATION include)
install(DIRECTORY ui DESTINATION share/micro-llm)
