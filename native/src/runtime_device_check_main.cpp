#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "json_min.hpp"
#include "runtime_device.hpp"

int main() {
  try {
    lkjai::CudaExecutionContext context;
    (void)context;
    lkjai::DeviceTensorSpec spec{lkjai::DeviceDType::bf16, {4}};
    lkjai::DeviceTensor tensor(spec);
    tensor.copy_from_host_f32({1.0f, -2.5f, 3.25f, 4.5f});
    auto roundtrip = tensor.copy_to_host_f32();
    bool ok = roundtrip.size() == 4;
    ok = ok && std::fabs(roundtrip[0] - 1.0f) < 0.01f;
    ok = ok && std::fabs(roundtrip[1] + 2.5f) < 0.01f;
    ok = ok && std::fabs(roundtrip[2] - 3.25f) < 0.01f;
    ok = ok && std::fabs(roundtrip[3] - 4.5f) < 0.01f;
    std::cout << "{\"status\":\"" << (ok ? "pass" : "fail")
              << "\",\"dtype\":\"" << lkjai::dtype_name(spec.dtype)
              << "\",\"elements\":" << spec.elements()
              << ",\"bytes\":" << spec.bytes() << "}\n";
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::cout << "{\"status\":\"fail\",\"error\":\""
              << lkjai::json_escape(e.what()) << "\"}\n";
    return 1;
  }
}
