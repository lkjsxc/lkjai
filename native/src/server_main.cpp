#include <sstream>

#include "artifact.hpp"
#include "capability_json.hpp"
#include "decoder_decode.hpp"
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
      << (artifact.loaded ? "true" : "false") << ",\"artifact_error\":\""
      << lkjai::json_escape(artifact.error) << "\","
      << lkjai::capability_json_fields(cuda) << "}";
  return out.str();
}

std::string models_json(const std::string& model, const lkjai::CudaStatus& cuda) {
  std::ostringstream out;
  out << "{\"data\":[{\"id\":\"" << lkjai::json_escape(model)
      << "\",\"object\":\"model\"}],"
      << lkjai::capability_json_fields(cuda) << "}";
  return out.str();
}

HttpResponse chat_json(const HttpRequest& request,
                       const lkjai::ArtifactStatus& artifact) {
  auto requested_model = lkjai::json_first_string(request.body, "model");
  if (!requested_model.empty() && requested_model != artifact.model_name) {
    return {404, error_json("requested model is not loaded")};
  }
  if (lkjai::json_string_values(request.body, "content").empty()) {
    return {400, error_json("chat request must include message content")};
  }
  auto manifest = lkjai::read_text(artifact.model_dir / "manifest.json");
  if (lkjai::contains_json_string(manifest, "kind", "transformer")) {
    return {422, error_json("native transformer autoregressive decode is unsupported")};
  }
  if (lkjai::contains_json_string(manifest, "kind", "decoder")) {
    std::string json;
    std::string error;
    if (!lkjai::decoder_chat_json(artifact.model_dir, artifact.model_name,
                                  request.body, &json, &error)) {
      return {500, error_json(error)};
    }
    return {200, json};
  }
  if (lkjai::contains_json_string(manifest, "kind", "dense")) {
    return {422, error_json("native dense autoregressive decode is unsupported")};
  }
  return {422, error_json("native autoregressive decode is unsupported")};
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
    return chat_json(request, artifact);
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
