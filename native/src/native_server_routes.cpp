#include "native_server_routes.hpp"

#include <sstream>

#include "capability_json.hpp"
#include "dense_demo.hpp"
#include "decoder_decode.hpp"
#include "json_min.hpp"
#include "native_status_page.hpp"
#include "runtime_agent.hpp"

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

std::string artifact_kind(const ArtifactStatus& artifact) {
  if (!artifact.loaded) return "none";
  auto manifest = read_text(artifact.model_dir / "manifest.json");
  auto kind = json_first_string(manifest, "kind");
  return kind.empty() ? "unknown" : kind;
}

std::string runtime_model_json(const std::string& model,
                               const ArtifactStatus& artifact,
                               const CudaStatus& cuda,
                               const RuntimeConfig& runtime) {
  bool ready = artifact.loaded;
  auto kind = artifact_kind(artifact);
  std::ostringstream out;
  out << "{\"model\":\"" << json_escape(model)
      << "\",\"api_url\":\"local-native-engine\",\"loaded\":"
      << (ready ? "true" : "false") << ",\"reachable\":"
      << (ready ? "true" : "false") << ",\"message\":\""
      << (ready ? "model loaded" : json_escape(artifact.error))
      << "\",\"probe_status\":" << (ready ? 200 : 503)
      << ",\"artifact_kind\":\"" << json_escape(kind) << "\""
      << ",\"chat_supported\":" << (ready && kind == "decoder" ? "true" : "false")
      << ",\"dense_supported\":" << (ready && kind == "dense" ? "true" : "false")
      << ",\"tool_profile\":\"" << json_escape(runtime.tool_profile) << "\","
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

HttpResponse runtime_chat_json(const RuntimeConfig& cfg,
                               const HttpRequest& request,
                               const ArtifactStatus& artifact) {
  return runtime_chat_with_model_callback(
      cfg, request, [&](const std::string& payload) {
        if (!artifact.loaded) {
          return NativeHttpResponse{503, error_json(artifact.error), ""};
        }
        HttpRequest model_request{"POST", "/v1/chat/completions", payload};
        auto model_response = openai_chat_json(model_request, artifact);
        return NativeHttpResponse{model_response.status, model_response.body, ""};
      });
}

}  // namespace

HttpResponse native_server_route(const HttpRequest& request,
                                 const ArtifactStatus& artifact,
                                 const CudaStatus& cuda,
                                 const RuntimeConfig& runtime,
                                 const DenseDemoRuntime& dense) {
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
    return {200, runtime_model_json(runtime.model, artifact, cuda, runtime)};
  }
  if (request.method == "GET" && request.path == "/api/config") {
    return {200, runtime_config_status_json(runtime)};
  }
  if (request.method == "GET" && request.path == "/api/dense/status") {
    return dense_demo_status_response(dense);
  }
  if (request.method == "POST" && request.path == "/api/dense/next-token") {
    return dense_demo_next_token_response(dense, request);
  }
  if (request.method == "POST" && request.path == "/api/chat") {
    return runtime_chat_json(runtime, request, artifact);
  }
  const std::string prefix = "/api/runs";
  if (request.method == "GET" && request.path.rfind(prefix, 0) == 0) {
    return runtime_route(runtime, request);
  }
  return {404, error_json("not found")};
}

HttpResponse native_server_route(const HttpRequest& request,
                                 const ArtifactStatus& artifact,
                                 const CudaStatus& cuda,
                                 const RuntimeConfig& runtime) {
  auto dense = load_dense_demo_runtime(artifact, runtime.data_dir);
  return native_server_route(request, artifact, cuda, runtime, dense);
}

}  // namespace lkjai
