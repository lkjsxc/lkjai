#include <sstream>

#include "artifact.hpp"
#include "cuda_probe.hpp"
#include "env.hpp"
#include "http_server.hpp"
#include "json_min.hpp"

using lkjai::HttpRequest;
using lkjai::HttpResponse;

namespace {

std::string error_json(std::string_view error) {
  return "{\"error\":\"" + lkjai::json_escape(error) + "\"}";
}

std::string health_json(const lkjai::ArtifactStatus& artifact,
                        const lkjai::CudaStatus& cuda) {
  std::ostringstream out;
  out << "{\"status\":\"ok\",\"loaded\":"
      << (artifact.loaded ? "true" : "false") << ",\"error\":\""
      << lkjai::json_escape(artifact.error) << "\",\"device\":\""
      << lkjai::json_escape(cuda.available ? "cuda" : "cpu")
      << "\",\"cuda_available\":" << (cuda.available ? "true" : "false")
      << ",\"gpu_name\":\"" << lkjai::json_escape(cuda.device)
      << "\",\"warning\":\"" << lkjai::json_escape(cuda.warning) << "\"}";
  return out.str();
}

std::string models_json(const std::string& model, const lkjai::CudaStatus& cuda) {
  std::ostringstream out;
  out << "{\"data\":[{\"id\":\"" << lkjai::json_escape(model)
      << "\",\"object\":\"model\"}],\"device\":\""
      << lkjai::json_escape(cuda.available ? "cuda" : "cpu")
      << "\",\"cuda_available\":" << (cuda.available ? "true" : "false")
      << ",\"gpu_name\":\"" << lkjai::json_escape(cuda.device)
      << "\",\"warning\":\"" << lkjai::json_escape(cuda.warning) << "\"}";
  return out.str();
}

HttpResponse route(const HttpRequest& request,
                   const lkjai::ArtifactStatus& artifact,
                   const lkjai::CudaStatus& cuda) {
  if (request.method == "GET" && request.path == "/healthz") {
    return {200, health_json(artifact, cuda)};
  }
  if (request.method == "GET" && request.path == "/v1/models") {
    if (!artifact.loaded) return {503, error_json(artifact.error)};
    return {200, models_json(artifact.model_name, cuda)};
  }
  if (request.method == "POST" && request.path == "/v1/chat/completions") {
    if (!artifact.loaded) return {503, error_json(artifact.error)};
    return {500, error_json("native decode executor is not implemented")};
  }
  return {404, error_json("not found")};
}

}  // namespace

int main() {
  auto host = lkjai::env_string("INFERENCE_HOST", "0.0.0.0");
  int port = lkjai::env_int("INFERENCE_PORT", 8081);
  auto root = lkjai::env_string("MODEL_ROOT", "/models");
  auto model = lkjai::env_string("MODEL_NAME", "lkjai-scratch-40m");
  auto artifact = lkjai::load_artifact(root, model);
  auto cuda = lkjai::cuda_status();
  return lkjai::serve_http(host, port, [&](const HttpRequest& request) {
    return route(request, artifact, cuda);
  });
}
