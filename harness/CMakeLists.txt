cmake_minimum_required(VERSION 3.16)

# CUDA is optional and guarded. CPU tests must configure without nvcc.
project(micro_llm_harness LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MICRO_LLM_CUDA "Build the CUDA FFN-reduce kernel (requires nvcc)" OFF)
option(MICRO_LLM_LLAMA "Link llama.cpp for a real Qwen 27B live forward" OFF)
set(MICRO_LLM_LLAMA_DIR "" CACHE PATH "Path to a llama.cpp checkout (optional)")

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
)
target_include_directories(micro_llm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(micro_llm PUBLIC cxx_std_17)
target_compile_options(micro_llm PRIVATE -Wall -Wextra -Wpedantic)

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

add_executable(micro-llm-dump src/cli_dump.cpp)
target_link_libraries(micro-llm-dump PRIVATE micro_llm)
target_compile_options(micro-llm-dump PRIVATE -Wall -Wextra -Wpedantic)

add_executable(micro-llm-trace src/cli_trace.cpp)
target_link_libraries(micro-llm-trace PRIVATE micro_llm)
target_compile_options(micro-llm-trace PRIVATE -Wall -Wextra -Wpedantic)

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
)
target_include_directories(micro_llm_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(micro_llm_tests PRIVATE micro_llm)
target_compile_options(micro_llm_tests PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_test(NAME micro_llm_tests COMMAND micro_llm_tests)
add_test(NAME micro_llm_dump_smoke
         COMMAND micro-llm-dump --out ${CMAKE_CURRENT_BINARY_DIR}/smoke_prune.bin --tokens 2)
add_test(NAME micro_llm_trace_no_weights
         COMMAND micro-llm-trace --model ${CMAKE_CURRENT_BINARY_DIR}/missing.gguf --out ${CMAKE_CURRENT_BINARY_DIR}/no.bin)
set_tests_properties(micro_llm_trace_no_weights PROPERTIES WILL_FAIL TRUE)
add_test(NAME micro_llm_trace_stub
         COMMAND micro-llm-trace --stub --n-predict 2 --out ${CMAKE_CURRENT_BINARY_DIR}/stub_prune.bin)

install(TARGETS micro_llm micro-llm-dump micro-llm-trace
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/micro_llm DESTINATION include)
