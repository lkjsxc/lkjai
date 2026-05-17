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

add_executable(lkjai-native-decoder-slice-block-check
  tests/decoder_slice_block_check.cpp
)
target_link_libraries(lkjai-native-decoder-slice-block-check
  PRIVATE lkjai_native_core
)

add_executable(lkjai-native-decoder-full-forward-check
  tests/decoder_cuda_full_forward_check.cpp
)
target_link_libraries(lkjai-native-decoder-full-forward-check
  PRIVATE lkjai_native_core
)

add_executable(lkjai-native-decoder-train-forward-check
  tests/decoder_cuda_train_forward_check.cpp
)
target_link_libraries(lkjai-native-decoder-train-forward-check
  PRIVATE lkjai_native_core
)

add_executable(lkjai-native-decoder-backward-primitives-check tests/decoder_cuda_backward_primitives_check.cpp)
target_link_libraries(lkjai-native-decoder-backward-primitives-check
  PRIVATE lkjai_native_core
)

add_executable(lkjai-native-decoder-projection-layout-check tests/decoder_cuda_projection_layout_check.cpp)
target_link_libraries(lkjai-native-decoder-projection-layout-check PRIVATE lkjai_native_core)

add_executable(lkjai-native-decoder-rope-backward-check
  tests/decoder_cuda_rope_backward_check.cpp
)
target_link_libraries(lkjai-native-decoder-rope-backward-check
  PRIVATE lkjai_native_core
)

add_executable(lkjai-native-decoder-full-backward-check
  tests/decoder_cuda_full_backward_check.cpp
)
target_link_libraries(lkjai-native-decoder-full-backward-check
  PRIVATE lkjai_native_core
)

add_executable(lkjai-native-decoder-block-weight-change-check
  tests/decoder_cuda_block_weight_change_check.cpp
)
target_link_libraries(lkjai-native-decoder-block-weight-change-check
  PRIVATE lkjai_native_core
)

add_executable(lkjai-native-decoder-decode-alloc-check
  tests/decoder_cuda_decode_alloc_check.cpp
)
target_link_libraries(lkjai-native-decoder-decode-alloc-check
  PRIVATE lkjai_native_core
)

add_executable(lkjai-native-decoder-tokenizer-contract
  tests/decoder_tokenizer_contract.cpp
)
target_link_libraries(lkjai-native-decoder-tokenizer-contract
  PRIVATE lkjai_native_core
)

add_executable(lkjai-native-decoder-artifact-contract
  tests/decoder_artifact_contract.cpp
)
target_link_libraries(lkjai-native-decoder-artifact-contract
  PRIVATE lkjai_native_core
)

add_executable(lkjai-native-decoder-kv-cache-contract
  tests/decoder_kv_cache_contract.cpp
)
target_link_libraries(lkjai-native-decoder-kv-cache-contract
  PRIVATE lkjai_native_core
)

add_executable(lkjai-native-decoder-inference-session-contract
  tests/decoder_inference_session_contract.cpp
)
target_link_libraries(lkjai-native-decoder-inference-session-contract
  PRIVATE lkjai_native_core
)

add_executable(lkjai-native-runtime-contract tests/runtime_contract_check.cpp)
target_link_libraries(lkjai-native-runtime-contract PRIVATE lkjai_native_core)

add_executable(lkjai-native-runtime-action-contract
  tests/runtime_action_contract.cpp
)
target_link_libraries(lkjai-native-runtime-action-contract PRIVATE
  lkjai_native_core
)

add_executable(lkjai-native-runtime-agent-contract
  tests/runtime_agent_contract.cpp
)
target_link_libraries(lkjai-native-runtime-agent-contract PRIVATE
  lkjai_native_core
)

add_executable(lkjai-native-runtime-tools-contract
  tests/runtime_tools_contract.cpp
)
target_link_libraries(lkjai-native-runtime-tools-contract PRIVATE
  lkjai_native_core
)

add_executable(lkjai-native-server-route-contract
  tests/server_route_contract.cpp
)
target_link_libraries(lkjai-native-server-route-contract PRIVATE
  lkjai_native_core
)

add_executable(lkjai-native-server-route-dense-contract
  tests/server_route_dense_contract.cpp
)
target_link_libraries(lkjai-native-server-route-dense-contract PRIVATE
  lkjai_native_core
)

add_executable(lkjai-native-dense-runtime-contract
  tests/dense_runtime_contract.cpp
)
target_link_libraries(lkjai-native-dense-runtime-contract PRIVATE
  lkjai_native_core
)

add_executable(lkjai-native-dense-scheduler-contract
  tests/dense_scheduler_contract.cpp
)
target_link_libraries(lkjai-native-dense-scheduler-contract PRIVATE
  lkjai_native_core
)

add_executable(lkjai-native-dense-train-outputs-contract
  tests/dense_train_outputs_contract.cpp
)
target_link_libraries(lkjai-native-dense-train-outputs-contract PRIVATE
  lkjai_native_core
)

add_executable(lkjai-native-decoder-route-contract
  tests/decoder_route_contract.cpp
)
target_link_libraries(lkjai-native-decoder-route-contract PRIVATE
  lkjai_native_core
)

add_executable(lkjai-native-decoder-acceptance-report-contract
  tests/decoder_acceptance_report_contract.cpp
)
target_link_libraries(lkjai-native-decoder-acceptance-report-contract PRIVATE
  lkjai_native_core
)

add_executable(lkjai-native-static-web-contract
  tests/static_web_contract.cpp
)
target_link_libraries(lkjai-native-static-web-contract PRIVATE
  lkjai_native_core
)

add_executable(lkjai-native-decoder-cuda-attention-plan-check
  tests/decoder_cuda_attention_plan_check.cpp
)
target_link_libraries(lkjai-native-decoder-cuda-attention-plan-check PRIVATE
  lkjai_native_core
)
