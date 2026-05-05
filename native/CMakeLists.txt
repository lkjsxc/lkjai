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
if(DEFINED LKJAI_GIT_COMMIT_OVERRIDE AND NOT "${LKJAI_GIT_COMMIT_OVERRIDE}" STREQUAL "")
  set(LKJAI_GIT_COMMIT "${LKJAI_GIT_COMMIT_OVERRIDE}")
endif()
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

include(cmake/NativeTargets.cmake)

enable_testing()
include(cmake/NativeTests.cmake)
