#include "native_server_routes.hpp"

#include <sstream>

#include "capability_json.hpp"
#include "decoder_decode.hpp"
#include "json_min.hpp"

namespace lkjai {
namespace {

std::string error_json(std::string_view error) {
  return "{\"error\":\"" + json_escape(error) + "\"}";
}

std::string health_json(const ArtifactStatus& artifact,
                        const CudaStatus& cuda) {
  auto kind = artifact.kind.empty()
                  ? json_first_string(read_text(artifact.model_dir / "manifest.json"),
                                      "kind")
                  : artifact.kind;
  auto decoder = decoder_route_capability(artifact.model_dir, kind);
  bool degraded = !artifact.loaded || !cuda.available;
  auto degraded_reason =
      !artifact.loaded
          ? artifact.error
          : (!cuda.available ? "cuda unavailable" : decoder.degraded_reason);
  if (!decoder.degraded_reason.empty()) degraded = true;
  std::ostringstream out;
  out << "{\"status\":\"ok\",\"loaded\":"
      << (artifact.loaded ? "true" : "false") << ",\"artifact_error\":\""
      << json_escape(artifact.error) << "\",\"artifact_kind\":\""
      << json_escape(kind) << "\",\"chat_supported\":"
      << (kind == "decoder" ? "true" : "false")
      << ",\"decode_supported\":" << (kind == "decoder" ? "true" : "false")
      << ",\"decoder_artifact_loadable\":"
      << (artifact.loaded && decoder.decoder_artifact ? "true" : "false")
      << ",\"decoder_partial_decode_supported\":"
      << (decoder.decode_supported ? "true" : "false")
      << ",\"decoder_accepted_decode_supported\":"
      << (decoder.decode_accepted ? "true" : "false")
      << ",\"decoder_train_report_accepted\":"
      << (decoder.train_report_accepted ? "true" : "false")
      << ",\"decoder_acceptance_sidecar_present\":"
      << (decoder.acceptance_sidecar_present ? "true" : "false")
      << ",\"decode_backend\":\"" << json_escape(decoder.decode_backend)
      << "\",\"kv_cache_backend\":\"" << json_escape(decoder.kv_cache_backend)
      << "\",\"attention_backend\":\"" << json_escape(decoder.attention_backend)
      << ",\"degraded\":" << (degraded ? "true" : "false")
      << ",\"degraded_reason\":\""
      << json_escape(degraded_reason)
      << "\"," << capability_json_fields(cuda)
      << "}";
  return out.str();
}

std::string models_json(const ArtifactStatus& artifact,
                        const CudaStatus& cuda) {
  auto kind = artifact.kind.empty()
                  ? json_first_string(read_text(artifact.model_dir / "manifest.json"),
                                      "kind")
                  : artifact.kind;
  auto decoder = decoder_route_capability(artifact.model_dir, kind);
  bool chat = kind == "decoder";
  bool degraded = !cuda.available || !decoder.degraded_reason.empty();
  auto degraded_reason =
      !cuda.available ? "cuda unavailable" : decoder.degraded_reason;
  std::ostringstream out;
  out << "{\"data\":[{\"id\":\"" << json_escape(artifact.model_name)
      << "\",\"object\":\"model\",\"artifact_kind\":\"" << json_escape(kind)
      << "\",\"chat_supported\":" << (chat ? "true" : "false")
      << ",\"decode_supported\":" << (chat ? "true" : "false")
      << ",\"decoder_partial_decode_supported\":"
      << (decoder.decode_supported ? "true" : "false")
      << ",\"decoder_accepted_decode_supported\":"
      << (decoder.decode_accepted ? "true" : "false")
      << ",\"decode_backend\":\"" << json_escape(decoder.decode_backend)
      << "\",\"kv_cache_backend\":\"" << json_escape(decoder.kv_cache_backend)
      << "}],\"artifact_kind\":\"" << json_escape(kind)
      << "\",\"chat_supported\":" << (chat ? "true" : "false")
      << ",\"decode_supported\":" << (chat ? "true" : "false")
      << ",\"decoder_artifact_loadable\":"
      << (artifact.loaded && decoder.decoder_artifact ? "true" : "false")
      << ",\"decoder_partial_decode_supported\":"
      << (decoder.decode_supported ? "true" : "false")
      << ",\"decoder_accepted_decode_supported\":"
      << (decoder.decode_accepted ? "true" : "false")
      << ",\"decoder_train_report_accepted\":"
      << (decoder.train_report_accepted ? "true" : "false")
      << ",\"decoder_acceptance_sidecar_present\":"
      << (decoder.acceptance_sidecar_present ? "true" : "false")
      << ",\"decode_backend\":\"" << json_escape(decoder.decode_backend)
      << "\",\"kv_cache_backend\":\"" << json_escape(decoder.kv_cache_backend)
      << "\",\"attention_backend\":\"" << json_escape(decoder.attention_backend)
      << ",\"degraded\":" << (degraded ? "true" : "false")
      << ",\"degraded_reason\":\""
      << json_escape(degraded_reason) << "\","
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

}  // namespace

HttpResponse native_server_route(const HttpRequest& request,
                                 const ArtifactStatus& artifact,
                                 const CudaStatus& cuda,
                                 const RuntimeConfig& runtime,
                                 const DenseDemoRuntime& dense) {
  (void)runtime;
  (void)dense;
  if (request.method == "OPTIONS") return {204, ""};
  if (request.method == "GET" && request.path == "/healthz") {
    return {200, health_json(artifact, cuda)};
  }
  if (request.method == "GET" && request.path == "/v1/models") {
    if (!artifact.loaded) return {503, error_json(artifact.error)};
    return {200, models_json(artifact, cuda)};
  }
  if (request.method == "POST" && request.path == "/v1/chat/completions") {
    if (!artifact.loaded) return {503, error_json(artifact.error)};
    return openai_chat_json(request, artifact);
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
