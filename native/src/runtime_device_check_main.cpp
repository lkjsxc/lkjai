#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "json_min.hpp"
#include "runtime_device.hpp"

namespace {

bool close(float actual, float expected) {
  return std::fabs(actual - expected) < 0.01f;
}

bool check_tensor(lkjai::DeviceDType dtype, cudaStream_t stream) {
  lkjai::DeviceTensorSpec spec{dtype, {4}};
  lkjai::DeviceTensor tensor(spec);
  tensor.copy_from_host_f32({1.0f, -2.5f, 3.25f, 4.5f}, stream);
  auto roundtrip = tensor.copy_to_host_f32(stream);
  return roundtrip.size() == 4 && close(roundtrip[0], 1.0f) &&
         close(roundtrip[1], -2.5f) && close(roundtrip[2], 3.25f) &&
         close(roundtrip[3], 4.5f);
}

}  // namespace

int main() {
  try {
    lkjai::CudaExecutionContext context;
    lkjai::DeviceWorkspace workspace(context.stream());
    bool ok = check_tensor(lkjai::DeviceDType::f32, context.stream()) &&
              check_tensor(lkjai::DeviceDType::bf16, context.stream()) &&
              workspace.allocate(4096) != nullptr;
    std::cout << "{\"status\":\"" << (ok ? "pass" : "fail")
              << "\",\"dtypes\":[\"f32\",\"bf16\"]"
              << ",\"workspace_bytes\":" << workspace.bytes_reserved()
              << ",\"async_alloc_supported\":"
              << (workspace.async_supported() ? "true" : "false") << "}\n";
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::cout << "{\"status\":\"fail\",\"error\":\""
              << lkjai::json_escape(e.what()) << "\"}\n";
    return 1;
  }
}
