#include <iostream>

#include "dense_cuda.hpp"
#include "json_min.hpp"

int main() {
  auto check = lkjai::run_dense_cuda_check();
  std::cout << "{\"status\":\"" << (check.ok ? "pass" : "fail")
            << "\",\"device\":\"" << lkjai::json_escape(check.device)
            << "\",\"compute_capability\":[" << check.compute_major << ","
            << check.compute_minor << "]"
            << ",\"cuda_runtime_version\":" << check.cuda_runtime_version
            << ",\"cudnn_version\":" << check.cudnn_version
            << ",\"bf16_supported\":"
            << (check.bf16_supported ? "true" : "false")
            << ",\"cublaslt_available\":"
            << (check.cublaslt_available ? "true" : "false")
            << ",\"cudnn_available\":"
            << (check.cudnn_available ? "true" : "false")
            << ",\"sdpa_eligible\":"
            << (check.sdpa_eligible ? "true" : "false")
            << ",\"error\":\"" << lkjai::json_escape(check.error)
            << "\"}\n";
  return check.ok ? 0 : 1;
}
