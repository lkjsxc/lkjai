#include <sstream>

#include "artifact.hpp"
#include "capability_json.hpp"
#include "decoder_decode.hpp"
#include "cuda_probe.hpp"
#include "env.hpp"
#include "http_server.hpp"
#include "json_min.hpp"
#include "runtime_api.hpp"

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

std::string runtime_model_json(const std::string& model,
                               const lkjai::ArtifactStatus& artifact,
                               const lkjai::CudaStatus& cuda) {
  bool ready = artifact.loaded;
  std::ostringstream out;
  out << "{\"model\":\"" << lkjai::json_escape(model)
      << "\",\"api_url\":\"local-native-engine\",\"loaded\":"
      << (ready ? "true" : "false") << ",\"reachable\":"
      << (ready ? "true" : "false") << ",\"message\":\""
      << (ready ? "model loaded" : lkjai::json_escape(artifact.error))
      << "\",\"probe_status\":" << (ready ? 200 : 503) << ","
      << lkjai::capability_json_fields(cuda) << "}";
  return out.str();
}

HttpResponse openai_chat_json(const HttpRequest& request,
                              const lkjai::ArtifactStatus& artifact) {
  auto requested_model = lkjai::json_first_string(request.body, "model");
  if (!requested_model.empty() && requested_model != artifact.model_name) {
    return {404, error_json("requested model is not loaded")};
  }
  auto manifest = lkjai::read_text(artifact.model_dir / "manifest.json");
  if (lkjai::contains_json_string(manifest, "kind", "transformer")) {
    return {422, error_json("native transformer autoregressive decode is unsupported")};
  }
  if (lkjai::contains_json_string(manifest, "kind", "decoder")) {
    std::string json;
    std::string error;
    int status = 500;
    if (!lkjai::decoder_chat_json(artifact.model_dir, artifact.model_name,
                                  request.body, &json, &status, &error)) {
      return {status, error_json(error)};
    }
    return {200, json};
  }
  if (lkjai::contains_json_string(manifest, "kind", "dense")) {
    return {422, error_json("native dense autoregressive decode is unsupported")};
  }
  return {422, error_json("native autoregressive decode is unsupported")};
}

std::string runtime_chat_payload(const std::string& model,
                                 const std::string& message) {
  return "{\"model\":\"" + lkjai::json_escape(model) +
         "\",\"messages\":[{\"role\":\"user\",\"content\":\"" +
         lkjai::json_escape(message) +
         "\"}],\"max_tokens\":512,\"temperature\":0.2}";
}

HttpResponse runtime_chat_json(const lkjai::RuntimeConfig& cfg,
                               const HttpRequest& request,
                               const lkjai::ArtifactStatus& artifact) {
  lkjai::NativeHttpResponse native;
  if (!artifact.loaded) {
    native.status = 503;
    native.body = error_json(artifact.error);
  } else {
    auto message = lkjai::json_first_string(request.body, "message");
    HttpRequest model_request{"POST", "/v1/chat/completions",
                              runtime_chat_payload(cfg.model, message)};
    auto model_response = openai_chat_json(model_request, artifact);
    native.status = model_response.status;
    native.body = model_response.body;
  }
  return lkjai::runtime_chat_with_model_response(cfg, request, native);
}

HttpResponse route(const HttpRequest& request,
                   const lkjai::ArtifactStatus& artifact,
                   const lkjai::CudaStatus& cuda,
                   const lkjai::RuntimeConfig& runtime) {
  if (request.method == "GET" && request.path == "/") {
    return {200, "{\"service\":\"lkjai-native-server\",\"runtime\":\"merged\"}"};
  }
  if (request.method == "GET" && request.path == "/healthz") {
    return {200, health_json(artifact, cuda)};
  }
  if (request.method == "GET" && request.path == "/v1/models") {
    if (!artifact.loaded) return {503, error_json(artifact.error)};
    return {200, models_json(artifact.model_name, cuda)};
  }
  if (request.method == "POST" && request.path == "/v1/chat/completions") {
    if (!artifact.loaded) return {503, error_json(artifact.error)};
    return openai_chat_json(request, artifact);
  }
  if (request.method == "GET" && request.path == "/api/model") {
    return {200, runtime_model_json(runtime.model, artifact, cuda)};
  }
  if (request.method == "POST" && request.path == "/api/chat") {
    return runtime_chat_json(runtime, request, artifact);
  }
  const std::string prefix = "/api/runs/";
  if (request.method == "GET" && request.path.rfind(prefix, 0) == 0) {
    return lkjai::runtime_route(runtime, request);
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
  lkjai::RuntimeConfig runtime{
      host, port, lkjai::env_string("DATA_DIR", "/app/data"),
      "local-native-engine", model};
  return lkjai::serve_http(host, port, [&](const HttpRequest& request) {
    return route(request, artifact, cuda, runtime);
  });
}
