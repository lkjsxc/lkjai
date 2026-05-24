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

add_executable(lkjai-native-tokenizer-build src/tokenizer_build_main.cpp)
target_link_libraries(lkjai-native-tokenizer-build PRIVATE lkjai_native_core)

add_executable(lkjai-native-kimi-cli-bridge src/kimi_cli_bridge_main.cpp)

add_executable(lkjai-native-kimi-cli-runner
  src/kimi_cli_runner_main.cpp
  src/kimi_cli_runner.cpp
)
target_link_libraries(lkjai-native-kimi-cli-runner PRIVATE lkjai_native_core)

add_executable(lkjai-native-repo-check
  src/repo_check_main.cpp
  src/repo_check_common.cpp
  src/repo_check_docs.cpp
  src/repo_check_decoder_docs.cpp
  src/repo_check_words.cpp
  src/repo_check_quality.cpp
  src/repo_check_contracts.cpp
  src/repo_check_corpus.cpp
  src/repo_check_secret_defaults.cpp
)
target_link_libraries(lkjai-native-repo-check PRIVATE lkjai_native_core)

include(cmake/NativeContractTargets.cmake)
