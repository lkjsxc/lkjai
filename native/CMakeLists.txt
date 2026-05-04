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

add_executable(lkjai-native-packed-cache-reader-check
  tests/packed_cache_reader_check.cpp
)
target_link_libraries(lkjai-native-packed-cache-reader-check
  PRIVATE lkjai_native_core
)

enable_testing()
add_test(NAME native_config_contract COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/config_contract.py ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_test(NAME native_cuda_arch_contract COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda_arch_contract.py ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_test(NAME native_inspect_missing COMMAND lkjai-native-inspect --model-dir /missing)
set_tests_properties(native_inspect_missing PROPERTIES WILL_FAIL TRUE)
add_test(
  NAME native_train_smoke
  COMMAND sh -c "$<TARGET_FILE:lkjai-native-train> --smoke --steps 2 > /tmp/lkjai-native-smoke.json && python3 -c 'import json; p=json.load(open(\"/tmp/lkjai-native-smoke.json\")); assert p[\"dense_cuda_path\"] is True and p[\"loss\"] < p[\"initial_loss\"]'"
)
add_test(NAME native_dense_cuda_parity COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/dense_check_contract.py $<TARGET_FILE:lkjai-native-dense-check>)
add_test(NAME native_device_tensor_check COMMAND lkjai-native-device-check)
add_test(
  NAME native_packed_cache_reader_check
  COMMAND lkjai-native-packed-cache-reader-check
)
add_test(
  NAME native_packed_train_contract
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/packed_train_contract.py $<TARGET_FILE:lkjai-native-train> $<TARGET_FILE:lkjai-native-logits-check> $<TARGET_FILE:lkjai-native-inspect>
)
add_test(
  NAME native_dense_infer_contract
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/dense_infer_contract.py $<TARGET_FILE:lkjai-native-train> $<TARGET_FILE:lkjai-native-infer>
)
add_test(
  NAME native_packed_cache_migration_contract
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/packed_cache_migration_contract.py $<TARGET_FILE:lkjai-native-train> $<TARGET_FILE:lkjai-native-packed-cache>
)
add_test(
  NAME native_transformer_train_contract
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/transformer_train_contract.py $<TARGET_FILE:lkjai-native-train> $<TARGET_FILE:lkjai-native-logits-check> $<TARGET_FILE:lkjai-native-inspect>
)
add_test(
  NAME native_decoder_train_contract
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/decoder_train_contract.py $<TARGET_FILE:lkjai-native-train> $<TARGET_FILE:lkjai-native-logits-check> $<TARGET_FILE:lkjai-native-inspect> $<TARGET_FILE:lkjai-native-server>
)
add_test(
  NAME native_decoder_acceptance_gate
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/decoder_acceptance_gate.py
)
add_test(
  NAME native_benchmark_report_parser
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/benchmark_report_parser.py
)
add_test(
  NAME native_accepted_training_report_validation
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/accepted_training_report_validation.py
)
add_test(
  NAME native_promote_dense_debug_validation
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/promote_dense_debug_validation.py
)
add_test(
  NAME native_smoke_export
  COMMAND sh -c "rm -rf /tmp/lkjai-native-ctest && DATA_DIR=/tmp/lkjai-native-ctest MODEL_NAME=smoke ./lkjai-native-train --smoke --steps 2 && ./lkjai-native-inspect --model-dir /tmp/lkjai-native-ctest/exports/smoke && ./lkjai-native-logits-check --model-dir /tmp/lkjai-native-ctest/exports/smoke --tokens 1,2,3"
)
add_test(
  NAME native_server_chat_contract
  COMMAND sh -c "rm -rf /tmp/lkjai-native-server-test && DATA_DIR=/tmp/lkjai-native-server-test/train MODEL_NAME=smoke ./lkjai-native-train --smoke --steps 2 && python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/server_chat_contract.py $<TARGET_FILE:lkjai-native-server> /tmp/lkjai-native-server-test/models smoke"
)
