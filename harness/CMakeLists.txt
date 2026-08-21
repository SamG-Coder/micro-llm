cmake_minimum_required(VERSION 3.16)

# CUDA is optional and guarded. CPU tests must configure without nvcc.
project(micro_llm_harness LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MICRO_LLM_CUDA "Build the CUDA FFN-reduce kernel (requires nvcc)" OFF)

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
)
target_include_directories(micro_llm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(micro_llm PUBLIC cxx_std_17)
target_compile_options(micro_llm PRIVATE -Wall -Wextra -Wpedantic)

if(MICRO_LLM_HAS_CUDA)
  target_sources(micro_llm PRIVATE src/ffn_reduce.cu)
  target_compile_definitions(micro_llm PUBLIC MICRO_LLM_HAS_CUDA=1)
  set_target_properties(micro_llm PROPERTIES CUDA_STANDARD 17)
endif()

add_executable(micro-llm-dump src/cli_dump.cpp)
target_link_libraries(micro-llm-dump PRIVATE micro_llm)
target_compile_options(micro-llm-dump PRIVATE -Wall -Wextra -Wpedantic)

add_executable(micro_llm_tests
  tests/test_main.cpp
  tests/test_serialize.cpp
  tests/test_pack_id.cpp
  tests/test_channel_align.cpp
  tests/test_bitset_floor.cpp
  tests/test_dead_spike.cpp
)
target_include_directories(micro_llm_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(micro_llm_tests PRIVATE micro_llm)
target_compile_options(micro_llm_tests PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_test(NAME micro_llm_tests COMMAND micro_llm_tests)
add_test(NAME micro_llm_dump_smoke
         COMMAND micro-llm-dump --out ${CMAKE_CURRENT_BINARY_DIR}/smoke_prune.bin --tokens 2)

install(TARGETS micro_llm micro-llm-dump
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/micro_llm DESTINATION include)
