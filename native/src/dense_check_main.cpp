#include <iostream>

#include "dense_cuda.hpp"
#include "json_min.hpp"

int main() {
  auto check = lkjai::run_dense_cuda_check();
  std::cout << "{\"status\":\"" << (check.ok ? "pass" : "fail")
            << "\",\"device\":\"" << lkjai::json_escape(check.device)
            << "\",\"compute_capability\":[" << check.compute_major << ","
            << check.compute_minor << "]"
            << ",\"cuda_driver_version\":" << check.cuda_driver_version
            << ",\"cuda_runtime_version\":" << check.cuda_runtime_version
            << ",\"cudnn_version\":" << check.cudnn_version
            << ",\"cuda_device_count\":" << check.device_count
            << ",\"cuda_device_index\":" << check.device_index
            << ",\"cuda_total_global_memory\":"
            << static_cast<unsigned long long>(check.total_global_memory)
            << ",\"cuda_sm_count\":" << check.sm_count
            << ",\"cuda_arch_flags\":\""
            << lkjai::json_escape(check.cuda_arch_flags)
            << "\",\"cuda_arch_source\":\""
            << lkjai::json_escape(check.cuda_arch_source) << "\""
            << ",\"bf16_supported\":"
            << (check.bf16_supported ? "true" : "false")
            << ",\"cublaslt_available\":"
            << (check.cublaslt_available ? "true" : "false")
            << ",\"cudnn_available\":"
            << (check.cudnn_available ? "true" : "false")
            << ",\"sdpa_eligible\":"
            << (check.sdpa_eligible ? "true" : "false")
            << ",\"async_alloc_supported\":"
            << (check.async_alloc_supported ? "true" : "false")
            << ",\"loss\":" << check.loss
            << ",\"cpu_loss\":" << check.cpu_loss
            << ",\"max_logit_diff\":" << check.max_logit_diff
            << ",\"max_grad_diff\":" << check.max_grad_diff
            << ",\"max_update_diff\":" << check.max_update_diff
            << ",\"error\":\"" << lkjai::json_escape(check.error)
            << "\"}\n";
  return check.ok ? 0 : 1;
}
