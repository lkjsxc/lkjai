set(LKJAI_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_test(NAME native_docs_topology
  COMMAND lkjai-native-repo-check docs-topology --repo ${LKJAI_REPO_ROOT})
add_test(NAME native_docs_links
  COMMAND lkjai-native-repo-check docs-links --repo ${LKJAI_REPO_ROOT})
add_test(NAME native_line_limits
  COMMAND lkjai-native-repo-check line-limits --repo ${LKJAI_REPO_ROOT})
add_test(NAME native_no_node
  COMMAND lkjai-native-repo-check no-node --repo ${LKJAI_REPO_ROOT})
add_test(NAME native_native_only
  COMMAND lkjai-native-repo-check native-only --repo ${LKJAI_REPO_ROOT})

add_test(
  NAME native_corpus_actions
  COMMAND lkjai-native-repo-check corpus-actions --
    ${LKJAI_REPO_ROOT}/corpus/generated/kimi-sft-60m-v2/train/train-000001.jsonl
    ${LKJAI_REPO_ROOT}/corpus/generated/kimi-sft-60m-v2/val/val-000001.jsonl
    ${LKJAI_REPO_ROOT}/corpus/generated/kimi-sft-60m-v2/holdout/holdout-000001.jsonl
)

add_test(NAME native_inspect_missing
  COMMAND lkjai-native-inspect --model-dir /missing)
set_tests_properties(native_inspect_missing PROPERTIES WILL_FAIL TRUE)

add_test(
  NAME native_train_smoke
  COMMAND sh -c "$<TARGET_FILE:lkjai-native-train> --smoke --steps 2 > /tmp/lkjai-native-smoke.json && grep -q '\"dense_cuda_path\":true' /tmp/lkjai-native-smoke.json && grep -q '\"status\":\"success\"' /tmp/lkjai-native-smoke.json"
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
add_test(NAME native_decoder_tokenizer_contract
  COMMAND lkjai-native-decoder-tokenizer-contract)
add_test(NAME native_runtime_contract COMMAND lkjai-native-runtime-contract)

set_tests_properties(native_decoder_tokenizer_contract PROPERTIES
  ENVIRONMENT "LKJAI_REPO_ROOT=${LKJAI_REPO_ROOT}")
set_tests_properties(native_decoder_cuda_block_forward_substrate PROPERTIES
  ENVIRONMENT "LKJAI_REPO_ROOT=${LKJAI_REPO_ROOT}")

add_test(
  NAME native_smoke_export
  COMMAND sh -c "rm -rf /tmp/lkjai-native-ctest && DATA_DIR=/tmp/lkjai-native-ctest MODEL_NAME=smoke ./lkjai-native-train --smoke --steps 2 && ./lkjai-native-inspect --model-dir /tmp/lkjai-native-ctest/exports/smoke && ./lkjai-native-logits-check --model-dir /tmp/lkjai-native-ctest/exports/smoke --tokens 1,2,3"
)
