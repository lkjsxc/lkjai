#include "native_server_routes.hpp"

#include <sstream>

#include "capability_json.hpp"
#include "dense_demo.hpp"
#include "decoder_decode.hpp"
#include "json_min.hpp"
#include "native_status_page.hpp"

namespace lkjai {
namespace {

std::string error_json(std::string_view error) {
  return "{\"error\":\"" + json_escape(error) + "\"}";
}

std::string health_json(const ArtifactStatus& artifact,
                        const CudaStatus& cuda) {
  std::ostringstream out;
  out << "{\"status\":\"ok\",\"loaded\":"
      << (artifact.loaded ? "true" : "false") << ",\"artifact_error\":\""
      << json_escape(artifact.error) << "\"," << capability_json_fields(cuda)
      << "}";
  return out.str();
}

std::string models_json(const std::string& model, const CudaStatus& cuda) {
  std::ostringstream out;
  out << "{\"data\":[{\"id\":\"" << json_escape(model)
      << "\",\"object\":\"model\"}]," << capability_json_fields(cuda) << "}";
  return out.str();
}

std::string runtime_model_json(const std::string& model,
                               const ArtifactStatus& artifact,
                               const CudaStatus& cuda) {
  bool ready = artifact.loaded;
  std::ostringstream out;
  out << "{\"model\":\"" << json_escape(model)
      << "\",\"api_url\":\"local-native-engine\",\"loaded\":"
      << (ready ? "true" : "false") << ",\"reachable\":"
      << (ready ? "true" : "false") << ",\"message\":\""
      << (ready ? "model loaded" : json_escape(artifact.error))
      << "\",\"probe_status\":" << (ready ? 200 : 503) << ","
      << capability_json_fields(cuda) << "}";
  return out.str();
}

HttpResponse openai_chat_json(const HttpRequest& request,
                              const ArtifactStatus& artifact) {
  auto requested_model = json_first_string(request.body, "model");
  if (!requested_model.empty() && requested_model != artifact.model_name) {
    return {404, error_json("requested model is not loaded")};
  }
  auto manifest = read_text(artifact.model_dir / "manifest.json");
  if (contains_json_string(manifest, "kind", "transformer")) {
    return {422, error_json("native transformer autoregressive decode is unsupported")};
  }
  if (contains_json_string(manifest, "kind", "decoder")) {
    std::string json;
    std::string error;
    int status = 500;
    if (!decoder_chat_json(artifact.model_dir, artifact.model_name,
                           request.body, &json, &status, &error)) {
      return {status, error_json(error)};
    }
    return {200, json};
  }
  if (contains_json_string(manifest, "kind", "dense")) {
    return {422, error_json("native dense autoregressive decode is unsupported")};
  }
  return {422, error_json("native autoregressive decode is unsupported")};
}

std::string runtime_chat_payload(const std::string& model,
                                 const std::string& message) {
  return "{\"model\":\"" + json_escape(model) +
         "\",\"messages\":[{\"role\":\"user\",\"content\":\"" +
         json_escape(message) +
         "\"}],\"max_tokens\":512,\"temperature\":0.2}";
}

HttpResponse runtime_chat_json(const RuntimeConfig& cfg,
                               const HttpRequest& request,
                               const ArtifactStatus& artifact) {
  NativeHttpResponse native;
  if (!artifact.loaded) {
    native.status = 503;
    native.body = error_json(artifact.error);
  } else {
    auto message = json_first_string(request.body, "message");
    HttpRequest model_request{"POST", "/v1/chat/completions",
                              runtime_chat_payload(cfg.model, message)};
    auto model_response = openai_chat_json(model_request, artifact);
    native.status = model_response.status;
    native.body = model_response.body;
  }
  return runtime_chat_with_model_response(cfg, request, native);
}

}  // namespace

HttpResponse native_server_route(const HttpRequest& request,
                                 const ArtifactStatus& artifact,
                                 const CudaStatus& cuda,
                                 const RuntimeConfig& runtime) {
  if (request.method == "GET" && request.path == "/") {
    return {200, std::string(native_status_page_html()),
            "text/html; charset=utf-8"};
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
  if (request.method == "GET" && request.path == "/api/config") {
    return {200, runtime_config_status_json(runtime)};
  }
  if (request.method == "GET" && request.path == "/api/dense/status") {
    return dense_demo_status_response(artifact);
  }
  if (request.method == "POST" && request.path == "/api/dense/next-token") {
    return dense_demo_next_token_response(artifact, request);
  }
  if (request.method == "POST" && request.path == "/api/chat") {
    return runtime_chat_json(runtime, request, artifact);
  }
  const std::string prefix = "/api/runs/";
  if (request.method == "GET" && request.path.rfind(prefix, 0) == 0) {
    return runtime_route(runtime, request);
  }
  return {404, error_json("not found")};
}

}  // namespace lkjai
