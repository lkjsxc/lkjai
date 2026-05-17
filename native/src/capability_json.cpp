#include "capability_json.hpp"

#include <sstream>

#include "json_min.hpp"

#ifndef LKJAI_CUDA_ARCH_FLAGS
#define LKJAI_CUDA_ARCH_FLAGS "unknown"
#endif

#ifndef LKJAI_CUDA_ARCH_SOURCE
#define LKJAI_CUDA_ARCH_SOURCE "unknown"
#endif

namespace lkjai {

std::string capability_json_fields(const CudaStatus& status) {
  std::ostringstream out;
  out << "\"device\":\"" << json_escape(status.available ? "cuda" : "cpu")
      << "\",\"cuda_available\":" << (status.available ? "true" : "false")
      << ",\"gpu_name\":\"" << json_escape(status.device)
      << "\",\"compute_capability\":[" << status.compute_major << ","
      << status.compute_minor << "]"
      << ",\"cuda_driver_version\":" << status.cuda_driver_version
      << ",\"cuda_runtime_version\":" << status.cuda_runtime_version
      << ",\"cudnn_version\":" << status.cudnn_version
      << ",\"cuda_device_count\":" << status.device_count
      << ",\"cuda_device_index\":" << status.device_index
      << ",\"cuda_total_global_memory\":"
      << static_cast<unsigned long long>(status.total_global_memory)
      << ",\"cuda_sm_count\":" << status.sm_count
      << ",\"cuda_arch_flags\":\"" << json_escape(LKJAI_CUDA_ARCH_FLAGS)
      << "\",\"cuda_arch_source\":\"" << json_escape(LKJAI_CUDA_ARCH_SOURCE)
      << "\""
      << ",\"bf16_supported\":"
      << (status.bf16_supported ? "true" : "false")
      << ",\"cublaslt_available\":"
      << (status.cublaslt_available ? "true" : "false")
      << ",\"cudnn_available\":"
      << (status.cudnn_available ? "true" : "false")
      << ",\"sdpa_eligible\":" << (status.sdpa_eligible ? "true" : "false")
      << ",\"decoder_sdpa_shape_available\":"
      << (status.sdpa_shape_available ? "true" : "false")
      << ",\"decoder_sdpa_forward_supported\":"
      << (status.sdpa_shape_forward_supported ? "true" : "false")
      << ",\"decoder_sdpa_backward_supported\":"
      << (status.sdpa_shape_backward_supported ? "true" : "false")
      << ",\"decoder_sdpa_decode_supported\":"
      << (status.sdpa_shape_decode_supported ? "true" : "false")
      << ",\"decoder_sdpa_cudnn_runtime_version\":"
      << status.cudnn_version
      << ",\"decoder_sdpa_head_dim\":" << status.sdpa_shape_head_dim
      << ",\"decoder_sdpa_layout\":\""
      << json_escape(status.sdpa_shape_layout)
      << "\",\"decoder_sdpa_reason\":\""
      << json_escape(status.sdpa_shape_reason) << "\""
      << ",\"async_alloc_supported\":"
      << (status.async_alloc_supported ? "true" : "false")
      << ",\"warning\":\"" << json_escape(status.warning)
      << "\",\"error\":\"" << json_escape(status.error) << "\"";
  return out.str();
}

std::string capability_json(const CudaStatus& status, bool ok) {
  return "{\"status\":\"" + std::string(ok ? "pass" : "fail") + "\"," +
         capability_json_fields(status) + "}";
}

}  // namespace lkjai
