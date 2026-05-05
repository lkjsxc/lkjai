add_executable(lkjai-native-server src/server_main.cpp src/http_server.cpp)
target_link_libraries(lkjai-native-server PRIVATE lkjai_native_core)

add_executable(lkjai-native-runtime src/runtime_main.cpp src/http_server.cpp)
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

add_executable(lkjai-native-decoder-cuda-block-check
  src/decoder_cuda_block_check.cpp
)
target_link_libraries(lkjai-native-decoder-cuda-block-check
  PRIVATE lkjai_native_core
)

add_executable(lkjai-native-decoder-tokenizer-contract
  tests/decoder_tokenizer_contract.cpp
)
target_link_libraries(lkjai-native-decoder-tokenizer-contract
  PRIVATE lkjai_native_core
)

add_executable(lkjai-native-runtime-contract tests/runtime_contract_check.cpp)
target_link_libraries(lkjai-native-runtime-contract PRIVATE lkjai_native_core)
