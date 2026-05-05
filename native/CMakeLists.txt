cmake_minimum_required(VERSION 3.27)

set(LKJAI_DEFAULT_CUDA_ARCHITECTURES
  86-real 86-virtual
  89-real 89-virtual
  90-real 90-virtual
  120-real 120-virtual
)
if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
  set(LKJAI_CUDA_ARCH_SOURCE "cmake")
elseif(DEFINED ENV{LKJAI_CUDA_ARCHS} AND NOT "$ENV{LKJAI_CUDA_ARCHS}" STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES "$ENV{LKJAI_CUDA_ARCHS}" CACHE STRING "CUDA architectures" FORCE)
  set(LKJAI_CUDA_ARCH_SOURCE "env")
else()
  set(CMAKE_CUDA_ARCHITECTURES "${LKJAI_DEFAULT_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures" FORCE)
  set(LKJAI_CUDA_ARCH_SOURCE "default")
endif()

project(lkjai_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
message(STATUS "LKJAI CUDA architectures (${LKJAI_CUDA_ARCH_SOURCE}): ${CMAKE_CUDA_ARCHITECTURES}")
execute_process(
  COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR}/.. rev-parse --short HEAD
  OUTPUT_VARIABLE LKJAI_GIT_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(NOT LKJAI_GIT_COMMIT)
  set(LKJAI_GIT_COMMIT unknown)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(LKJAI_BUILD_TYPE unknown)
else()
  set(LKJAI_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
endif()
string(REPLACE ";" "," LKJAI_CUDA_ARCH_FLAGS "${CMAKE_CUDA_ARCHITECTURES}")
find_package(CUDAToolkit REQUIRED)
find_library(CUDNN_LIBRARY cudnn REQUIRED)
include(cmake/NativeSources.cmake)

add_library(lkjai_cuda STATIC ${LKJAI_CUDA_SOURCES})
target_link_libraries(lkjai_cuda PUBLIC
  CUDA::cudart
  CUDA::cublas
  CUDA::cublasLt
  ${CUDNN_LIBRARY}
)
target_compile_options(lkjai_cuda PUBLIC
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
target_compile_definitions(lkjai_cuda PUBLIC LKJAI_WITH_CUDA=1)

add_library(lkjai_native_core STATIC ${LKJAI_NATIVE_CORE_SOURCES})
target_include_directories(lkjai_native_core PUBLIC src)
target_link_libraries(lkjai_native_core PUBLIC lkjai_cuda)
target_compile_definitions(lkjai_native_core PUBLIC
  LKJAI_GIT_COMMIT="${LKJAI_GIT_COMMIT}"
  LKJAI_BUILD_TYPE="${LKJAI_BUILD_TYPE}"
  LKJAI_CUDA_ARCH_FLAGS="${LKJAI_CUDA_ARCH_FLAGS}"
  LKJAI_CUDA_ARCH_SOURCE="${LKJAI_CUDA_ARCH_SOURCE}"
)

add_executable(lkjai-native-server src/server_main.cpp src/http_server.cpp)
target_link_libraries(lkjai-native-server PRIVATE lkjai_native_core)

add_executable(lkjai-native-runtime
  src/runtime_main.cpp
  src/http_server.cpp
)
target_link_libraries(lkjai-native-runtime PRIVATE lkjai_native_core)

add_executable(lkjai-native-train src/train_main.cpp)
target_link_libraries(lkjai-native-train PRIVATE lkjai_native_core)

add_executable(lkjai-native-inspect src/inspect_main.cpp)
target_link_libraries(lkjai-native-inspect PRIVATE lkjai_native_core)

add_executable(lkjai-native-infer src/infer_main.cpp)
target_link_libraries(lkjai-native-infer PRIVATE lkjai_native_core)

add_executable(lkjai-native-dense-check src/dense_check_main.cpp)
target_link_libraries(lkjai-native-dense-check PRIVATE lkjai_native_core)

add_executable(lkjai-native-device-check src/runtime_device_check_main.cpp)
target_link_libraries(lkjai-native-device-check PRIVATE lkjai_native_core)

add_executable(lkjai-native-logits-check src/logits_check_main.cpp)
target_link_libraries(lkjai-native-logits-check PRIVATE lkjai_native_core)

add_executable(lkjai-native-packed-cache src/packed_cache_main.cpp)
target_link_libraries(lkjai-native-packed-cache PRIVATE lkjai_native_core)

add_executable(lkjai-native-repo-check
  src/repo_check_main.cpp
  src/repo_check_common.cpp
  src/repo_check_docs.cpp
  src/repo_check_quality.cpp
  src/repo_check_corpus.cpp
)
target_link_libraries(lkjai-native-repo-check PRIVATE lkjai_native_core)

add_executable(lkjai-native-packed-cache-reader-check
  tests/packed_cache_reader_check.cpp
)
target_link_libraries(lkjai-native-packed-cache-reader-check
  PRIVATE lkjai_native_core
)

add_executable(lkjai-native-packed-cache-strict-check
  tests/packed_cache_strict_check.cpp
)
target_link_libraries(lkjai-native-packed-cache-strict-check
  PRIVATE lkjai_native_core
)

add_executable(lkjai-native-decoder-cuda-norm-check
  tests/decoder_cuda_norm_check.cpp
)
target_link_libraries(lkjai-native-decoder-cuda-norm-check
  PRIVATE lkjai_native_core
)

add_executable(lkjai-native-decoder-tokenizer-contract
  tests/decoder_tokenizer_contract.cpp
)
target_link_libraries(lkjai-native-decoder-tokenizer-contract
  PRIVATE lkjai_native_core
)

add_executable(lkjai-native-runtime-contract
  tests/runtime_contract_check.cpp
)
target_link_libraries(lkjai-native-runtime-contract
  PRIVATE lkjai_native_core
)

enable_testing()
add_test(NAME native_docs_topology COMMAND lkjai-native-repo-check docs-topology --repo ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_test(NAME native_docs_links COMMAND lkjai-native-repo-check docs-links --repo ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_test(NAME native_line_limits COMMAND lkjai-native-repo-check line-limits --repo ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_test(NAME native_no_node COMMAND lkjai-native-repo-check no-node --repo ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_test(
  NAME native_corpus_actions
  COMMAND lkjai-native-repo-check corpus-actions --
    ${CMAKE_CURRENT_SOURCE_DIR}/../corpus/generated/kimi-sft-60m-v2/train/train-000001.jsonl
    ${CMAKE_CURRENT_SOURCE_DIR}/../corpus/generated/kimi-sft-60m-v2/val/val-000001.jsonl
    ${CMAKE_CURRENT_SOURCE_DIR}/../corpus/generated/kimi-sft-60m-v2/holdout/holdout-000001.jsonl
)
add_test(NAME native_inspect_missing COMMAND lkjai-native-inspect --model-dir /missing)
set_tests_properties(native_inspect_missing PROPERTIES WILL_FAIL TRUE)
add_test(
  NAME native_train_smoke
  COMMAND sh -c "$<TARGET_FILE:lkjai-native-train> --smoke --steps 2 > /tmp/lkjai-native-smoke.json && grep -q '\"dense_cuda_path\":true' /tmp/lkjai-native-smoke.json && grep -q '\"status\":\"success\"' /tmp/lkjai-native-smoke.json"
)
add_test(NAME native_device_tensor_check COMMAND lkjai-native-device-check)
add_test(
  NAME native_packed_cache_reader_check
  COMMAND lkjai-native-packed-cache-reader-check
)
add_test(
  NAME native_packed_cache_strict_check
  COMMAND lkjai-native-packed-cache-strict-check
)
add_test(
  NAME native_decoder_cuda_rmsnorm_parity
  COMMAND lkjai-native-decoder-cuda-norm-check
)
add_test(
  NAME native_decoder_tokenizer_contract
  COMMAND lkjai-native-decoder-tokenizer-contract
)
add_test(
  NAME native_runtime_contract
  COMMAND lkjai-native-runtime-contract
)
set_tests_properties(native_decoder_tokenizer_contract PROPERTIES
  ENVIRONMENT "LKJAI_REPO_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/.."
)
add_test(
  NAME native_smoke_export
  COMMAND sh -c "rm -rf /tmp/lkjai-native-ctest && DATA_DIR=/tmp/lkjai-native-ctest MODEL_NAME=smoke ./lkjai-native-train --smoke --steps 2 && ./lkjai-native-inspect --model-dir /tmp/lkjai-native-ctest/exports/smoke && ./lkjai-native-logits-check --model-dir /tmp/lkjai-native-ctest/exports/smoke --tokens 1,2,3"
)
