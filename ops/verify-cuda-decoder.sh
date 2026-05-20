#!/usr/bin/env bash
set -euo pipefail

cmake -S native -B "${BUILD_DIR:-build/cuda-decoder}" -G Ninja \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
cmake --build "${BUILD_DIR:-build/cuda-decoder}" --parallel
ctest --test-dir "${BUILD_DIR:-build/cuda-decoder}" \
  -R 'decoder_cuda|decoder_acceptance|decoder_route' --output-on-failure
