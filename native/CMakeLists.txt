cmake_minimum_required(VERSION 3.27)
project(lkjai_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_ARCHITECTURES 86)
  add_library(lkjai_cuda STATIC src/cuda_probe.cu)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(lkjai_cuda PUBLIC CUDA::cudart)
  target_compile_definitions(lkjai_cuda PUBLIC LKJAI_WITH_CUDA=1)
else()
  add_library(lkjai_cuda STATIC src/cuda_probe_stub.cpp)
endif()

add_library(lkjai_native_core STATIC
  src/artifact.cpp
  src/env.cpp
  src/json_min.cpp
)
target_include_directories(lkjai_native_core PUBLIC src)
target_link_libraries(lkjai_native_core PUBLIC lkjai_cuda)

add_executable(lkjai-native-server src/server_main.cpp src/http_server.cpp)
target_link_libraries(lkjai-native-server PRIVATE lkjai_native_core)

add_executable(lkjai-native-train src/train_main.cpp)
target_link_libraries(lkjai-native-train PRIVATE lkjai_native_core)

add_executable(lkjai-native-inspect src/inspect_main.cpp)
target_link_libraries(lkjai-native-inspect PRIVATE lkjai_native_core)

enable_testing()
add_test(NAME native_inspect_missing COMMAND lkjai-native-inspect --model-dir /missing)
set_tests_properties(native_inspect_missing PROPERTIES WILL_FAIL TRUE)
add_test(NAME native_train_smoke COMMAND lkjai-native-train --smoke --steps 2)
