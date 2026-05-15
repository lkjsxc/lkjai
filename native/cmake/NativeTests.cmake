set(LKJAI_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_test(NAME native_docs_topology
  COMMAND lkjai-native-repo-check docs-topology --repo ${LKJAI_REPO_ROOT})
add_test(NAME native_docs_links
  COMMAND lkjai-native-repo-check docs-links --repo ${LKJAI_REPO_ROOT})
add_test(NAME native_docs_contract_owners
  COMMAND lkjai-native-repo-check docs-contract-owners --repo ${LKJAI_REPO_ROOT})
add_test(NAME native_docs_wording
  COMMAND lkjai-native-repo-check docs-wording --repo ${LKJAI_REPO_ROOT})
add_test(NAME native_line_limits
  COMMAND lkjai-native-repo-check line-limits --repo ${LKJAI_REPO_ROOT})
add_test(NAME native_repo_readmes
  COMMAND lkjai-native-repo-check repo-readmes --repo ${LKJAI_REPO_ROOT})
add_test(NAME native_no_node
  COMMAND lkjai-native-repo-check no-node --repo ${LKJAI_REPO_ROOT})
add_test(NAME native_native_only
  COMMAND lkjai-native-repo-check native-only --repo ${LKJAI_REPO_ROOT})
add_test(
  NAME native_native_only_source_scan
  COMMAND sh ${LKJAI_REPO_ROOT}/native/tests/native_only_source_scan.sh
    $<TARGET_FILE:lkjai-native-repo-check>
)
add_test(NAME native_stable_identifiers
  COMMAND lkjai-native-repo-check stable-identifiers --repo ${LKJAI_REPO_ROOT})
add_test(NAME native_config_contract
  COMMAND lkjai-native-repo-check config-contract --repo ${LKJAI_REPO_ROOT})
add_test(
  NAME native_config_contract_negatives
  COMMAND sh ${LKJAI_REPO_ROOT}/native/tests/config_contract_negatives.sh
    $<TARGET_FILE:lkjai-native-repo-check>
)
add_test(NAME native_cuda_arch_contract
  COMMAND lkjai-native-repo-check cuda-arch-contract --repo ${LKJAI_REPO_ROOT})
add_test(NAME native_secret_defaults
  COMMAND lkjai-native-repo-check secret-defaults --repo ${LKJAI_REPO_ROOT})

add_test(
  NAME native_corpus_actions
  COMMAND lkjai-native-repo-check corpus-actions --
    ${LKJAI_REPO_ROOT}/corpus/generated/kimi-sft-60m/train/train-000001.jsonl
    ${LKJAI_REPO_ROOT}/corpus/generated/kimi-sft-60m/val/val-000001.jsonl
    ${LKJAI_REPO_ROOT}/corpus/generated/kimi-sft-60m/holdout/holdout-000001.jsonl
)

add_test(NAME native_inspect_missing
  COMMAND lkjai-native-inspect --model-dir /missing)
set_tests_properties(native_inspect_missing PROPERTIES WILL_FAIL TRUE)

add_test(
  NAME native_train_smoke
  COMMAND sh -c "$<TARGET_FILE:lkjai-native-train> --smoke --steps 2 > /tmp/lkjai-native-smoke.json && grep -q '\"dense_cuda_path\":true' /tmp/lkjai-native-smoke.json && grep -q '\"copy_compute_overlap_enabled\":false' /tmp/lkjai-native-smoke.json && grep -q '\"status\":\"success\"' /tmp/lkjai-native-smoke.json"
)
add_test(NAME native_device_tensor_check COMMAND lkjai-native-device-check)
add_test(NAME native_packed_cache_reader_check
  COMMAND lkjai-native-packed-cache-reader-check)
add_test(NAME native_packed_cache_strict_check
  COMMAND lkjai-native-packed-cache-strict-check)
add_test(NAME native_decoder_cuda_rmsnorm_parity
  COMMAND lkjai-native-decoder-cuda-norm-check)
add_test(NAME native_decoder_cuda_block_forward_substrate
  COMMAND lkjai-native-decoder-cuda-block-check)
add_test(NAME native_decoder_slice_block_forward
  COMMAND lkjai-native-decoder-slice-block-check)
add_test(NAME native_decoder_cuda_full_forward
  COMMAND lkjai-native-decoder-full-forward-check)
add_test(NAME native_decoder_cuda_train_forward
  COMMAND lkjai-native-decoder-train-forward-check)
add_test(NAME native_decoder_cuda_backward_primitives
  COMMAND lkjai-native-decoder-backward-primitives-check)
add_test(NAME native_decoder_cuda_rope_backward
  COMMAND lkjai-native-decoder-rope-backward-check)
add_test(NAME native_decoder_tokenizer_contract
  COMMAND lkjai-native-decoder-tokenizer-contract)
add_test(NAME native_decoder_artifact_contract
  COMMAND lkjai-native-decoder-artifact-contract)
add_test(NAME native_decoder_kv_cache_contract
  COMMAND lkjai-native-decoder-kv-cache-contract)
add_test(NAME native_decoder_inference_session_contract
  COMMAND lkjai-native-decoder-inference-session-contract)
add_test(NAME native_runtime_contract COMMAND lkjai-native-runtime-contract)
add_test(NAME native_runtime_action_contract
  COMMAND lkjai-native-runtime-action-contract)
add_test(NAME native_runtime_agent_contract
  COMMAND lkjai-native-runtime-agent-contract)
add_test(NAME native_runtime_tools_contract
  COMMAND lkjai-native-runtime-tools-contract)
add_test(NAME native_server_route_contract
  COMMAND lkjai-native-server-route-contract)
add_test(NAME native_server_route_dense_contract
  COMMAND lkjai-native-server-route-dense-contract)
add_test(NAME native_dense_runtime_contract
  COMMAND lkjai-native-dense-runtime-contract)
add_test(NAME native_dense_scheduler_contract
  COMMAND lkjai-native-dense-scheduler-contract)
add_test(NAME native_dense_train_outputs_contract
  COMMAND lkjai-native-dense-train-outputs-contract)
add_test(NAME native_decoder_route_contract
  COMMAND lkjai-native-decoder-route-contract)
add_test(NAME native_decoder_acceptance_report_contract
  COMMAND lkjai-native-decoder-acceptance-report-contract)
add_test(NAME native_static_web_contract
  COMMAND lkjai-native-static-web-contract)
add_test(NAME native_decoder_cuda_attention_plan
  COMMAND lkjai-native-decoder-cuda-attention-plan-check)

set_tests_properties(native_decoder_tokenizer_contract PROPERTIES
  ENVIRONMENT
    "LKJAI_REPO_ROOT=${LKJAI_REPO_ROOT};LKJAI_TOKENIZER_BUILD=$<TARGET_FILE:lkjai-native-tokenizer-build>")
set_tests_properties(native_decoder_cuda_block_forward_substrate PROPERTIES
  ENVIRONMENT "LKJAI_REPO_ROOT=${LKJAI_REPO_ROOT}")
set_tests_properties(native_decoder_cuda_full_forward PROPERTIES
  ENVIRONMENT "LKJAI_REPO_ROOT=${LKJAI_REPO_ROOT}")
set_tests_properties(native_decoder_cuda_train_forward PROPERTIES
  ENVIRONMENT "LKJAI_REPO_ROOT=${LKJAI_REPO_ROOT}")
set_tests_properties(native_decoder_inference_session_contract PROPERTIES
  ENVIRONMENT "LKJAI_REPO_ROOT=${LKJAI_REPO_ROOT}")
set_tests_properties(native_static_web_contract PROPERTIES
  ENVIRONMENT "LKJAI_REPO_ROOT=${LKJAI_REPO_ROOT}")

add_test(
  NAME native_smoke_export
  COMMAND sh -c "rm -rf /tmp/lkjai-native-ctest && DATA_DIR=/tmp/lkjai-native-ctest MODEL_NAME=smoke ./lkjai-native-train --smoke --steps 2 && ./lkjai-native-inspect --model-dir /tmp/lkjai-native-ctest/exports/smoke && ./lkjai-native-logits-check --model-dir /tmp/lkjai-native-ctest/exports/smoke --tokens 1,2,3"
)
add_test(
  NAME native_decoder_cli_smoke
  COMMAND sh ${LKJAI_REPO_ROOT}/native/tests/decoder_cli_smoke.sh
    $<TARGET_FILE:lkjai-native-train>
    $<TARGET_FILE:lkjai-native-tokenizer-build>
)
