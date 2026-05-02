cmake_minimum_required(VERSION 3.27)
project(lkjai_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86-real 86-virtual 90-real 90-virtual)
find_package(CUDAToolkit REQUIRED)
find_library(CUDNN_LIBRARY cudnn REQUIRED)

add_library(lkjai_cuda STATIC
  src/cuda_probe.cu
  src/dense_cuda.cu
)
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

add_library(lkjai_native_core STATIC
  src/artifact.cpp
  src/dense_model.cpp
  src/env.cpp
  src/json_min.cpp
  src/packed_cache.cpp
  src/train_data.cpp
  src/dense_train.cpp
  src/dense_train_artifact.cpp
  src/dense_train_math.cpp
  src/transformer_artifact.cpp
  src/transformer_config.cpp
  src/transformer_forward.cpp
  src/transformer_init.cpp
  src/transformer_optim.cpp
  src/transformer_train.cpp
  src/train_real.cpp
)
target_include_directories(lkjai_native_core PUBLIC src)
target_link_libraries(lkjai_native_core PUBLIC lkjai_cuda)

add_executable(lkjai-native-server src/server_main.cpp src/http_server.cpp)
target_link_libraries(lkjai-native-server PRIVATE lkjai_native_core)

add_executable(lkjai-native-train src/train_main.cpp)
target_link_libraries(lkjai-native-train PRIVATE lkjai_native_core)

add_executable(lkjai-native-inspect src/inspect_main.cpp)
target_link_libraries(lkjai-native-inspect PRIVATE lkjai_native_core)

add_executable(lkjai-native-dense-check src/dense_check_main.cpp)
target_link_libraries(lkjai-native-dense-check PRIVATE lkjai_native_core)

add_executable(lkjai-native-logits-check src/logits_check_main.cpp)
target_link_libraries(lkjai-native-logits-check PRIVATE lkjai_native_core)

add_executable(lkjai-native-packed-cache src/packed_cache_main.cpp)
target_link_libraries(lkjai-native-packed-cache PRIVATE lkjai_native_core)

enable_testing()
add_test(NAME native_inspect_missing COMMAND lkjai-native-inspect --model-dir /missing)
set_tests_properties(native_inspect_missing PROPERTIES WILL_FAIL TRUE)
add_test(NAME native_train_smoke COMMAND lkjai-native-train --smoke --steps 2)
add_test(NAME native_dense_cuda_check COMMAND lkjai-native-dense-check)
add_test(
  NAME native_packed_train_contract
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/packed_train_contract.py $<TARGET_FILE:lkjai-native-train> $<TARGET_FILE:lkjai-native-logits-check> $<TARGET_FILE:lkjai-native-packed-cache> $<TARGET_FILE:lkjai-native-inspect>
)
add_test(
  NAME native_smoke_export
  COMMAND sh -c "rm -rf /tmp/lkjai-native-ctest && DATA_DIR=/tmp/lkjai-native-ctest MODEL_NAME=smoke ./lkjai-native-train --smoke --steps 2 && ./lkjai-native-inspect --model-dir /tmp/lkjai-native-ctest/exports/smoke && ./lkjai-native-logits-check --model-dir /tmp/lkjai-native-ctest/exports/smoke --tokens 1,2,3"
)
add_test(
  NAME native_server_chat_contract
  COMMAND sh -c "rm -rf /tmp/lkjai-native-server-test && DATA_DIR=/tmp/lkjai-native-server-test/train MODEL_NAME=smoke ./lkjai-native-train --smoke --steps 2 && python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/server_chat_contract.py $<TARGET_FILE:lkjai-native-server> /tmp/lkjai-native-server-test/models smoke"
)
