#include "capability_json.hpp"

#include <sstream>

#include "json_min.hpp"

namespace lkjai {

std::string capability_json_fields(const CudaStatus& status) {
  std::ostringstream out;
  out << "\"device\":\"" << json_escape(status.available ? "cuda" : "cpu")
      << "\",\"cuda_available\":" << (status.available ? "true" : "false")
      << ",\"gpu_name\":\"" << json_escape(status.device)
      << "\",\"compute_capability\":[" << status.compute_major << ","
      << status.compute_minor << "]"
      << ",\"cuda_runtime_version\":" << status.cuda_runtime_version
      << ",\"cudnn_version\":" << status.cudnn_version
      << ",\"bf16_supported\":"
      << (status.bf16_supported ? "true" : "false")
      << ",\"cublaslt_available\":"
      << (status.cublaslt_available ? "true" : "false")
      << ",\"cudnn_available\":"
      << (status.cudnn_available ? "true" : "false")
      << ",\"sdpa_eligible\":" << (status.sdpa_eligible ? "true" : "false")
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
