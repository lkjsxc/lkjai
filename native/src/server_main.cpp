#include <sstream>

#include "artifact.hpp"
#include "cuda_probe.hpp"
#include "dense_model.hpp"
#include "env.hpp"
#include "http_server.hpp"
#include "json_min.hpp"

using lkjai::HttpRequest;
using lkjai::HttpResponse;

namespace {

std::string error_json(std::string_view error) {
  return "{\"error\":\"" + lkjai::json_escape(error) + "\"}";
}

std::string prompt_seed(const HttpRequest& request) {
  auto contents = lkjai::json_string_values(request.body, "content");
  std::string prompt;
  for (const auto& content : contents) {
    if (!prompt.empty()) prompt += "\n";
    prompt += content;
  }
  constexpr size_t kPromptTail = 4096;
  if (prompt.size() > kPromptTail) {
    prompt = prompt.substr(prompt.size() - kPromptTail);
  }
  return prompt + "\n<assistant_action>\n<action>\n<reasoning>";
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
      << "\",\"compute_capability\":[" << cuda.compute_major << ","
      << cuda.compute_minor << "]"
      << ",\"bf16_supported\":"
      << (cuda.bf16_supported ? "true" : "false")
      << ",\"warning\":\"" << lkjai::json_escape(cuda.warning) << "\"}";
  return out.str();
}

std::string models_json(const std::string& model, const lkjai::CudaStatus& cuda) {
  std::ostringstream out;
  out << "{\"data\":[{\"id\":\"" << lkjai::json_escape(model)
      << "\",\"object\":\"model\"}],\"device\":\""
      << lkjai::json_escape(cuda.available ? "cuda" : "cpu")
      << "\",\"cuda_available\":" << (cuda.available ? "true" : "false")
      << ",\"gpu_name\":\"" << lkjai::json_escape(cuda.device)
      << "\",\"compute_capability\":[" << cuda.compute_major << ","
      << cuda.compute_minor << "]"
      << ",\"bf16_supported\":"
      << (cuda.bf16_supported ? "true" : "false")
      << ",\"warning\":\"" << lkjai::json_escape(cuda.warning) << "\"}";
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
  auto max_chars = lkjai::json_int_value(request.body, "max_tokens", 512);
  if (max_chars < 1) max_chars = 1;
  if (max_chars > 4096) max_chars = 4096;
  auto decoded = lkjai::dense_generate_action(
      artifact.model_dir, prompt_seed(request), max_chars);
  auto start = decoded.rfind("<action>");
  auto end = start == std::string::npos ? std::string::npos
                                        : decoded.find("</action>", start);
  if (decoded.empty() || start == std::string::npos ||
      end == std::string::npos || end < start) {
    return {422, error_json("native decode did not produce a complete action")};
  }
  auto text = decoded.substr(start, end + std::string("</action>").size() - start);
  return {200, "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"" +
                   lkjai::json_escape(text) + "\"}}]}"};
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
