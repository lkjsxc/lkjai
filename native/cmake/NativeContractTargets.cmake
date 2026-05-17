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

include(cmake/NativeDecoderContractTargets.cmake)

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

add_executable(lkjai-native-static-web-contract
  tests/static_web_contract.cpp
)
target_link_libraries(lkjai-native-static-web-contract PRIVATE
  lkjai_native_core
)
