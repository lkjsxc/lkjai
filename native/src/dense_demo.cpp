#include "dense_demo.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

#include "dense_cuda_internal.hpp"
#include "json_min.hpp"

namespace lkjai {
namespace {

std::string error_json(std::string_view error) {
  return "{\"error\":\"" + json_escape(error) + "\"}";
}

bool parse_tokens(std::string_view body, std::vector<int>* tokens,
                  std::string* error) {
  auto key = body.find("\"tokens\"");
  if (key == std::string_view::npos) {
    *error = "tokens array is required";
    return false;
  }
  auto open = body.find('[', key);
  auto close = body.find(']', open);
  if (open == std::string_view::npos || close == std::string_view::npos) {
    *error = "tokens must be an array";
    return false;
  }
  size_t pos = open + 1;
  while (pos < close) {
    while (pos < close && (body[pos] == ' ' || body[pos] == '\n' ||
                           body[pos] == '\t' || body[pos] == ',')) {
      ++pos;
    }
    if (pos >= close) break;
    size_t end = pos;
    if (body[end] == '-') ++end;
    while (end < close && body[end] >= '0' && body[end] <= '9') ++end;
    if (end == pos || (body[pos] == '-' && end == pos + 1)) {
      *error = "tokens must contain integer ids";
      return false;
    }
    try {
      tokens->push_back(std::stoi(std::string(body.substr(pos, end - pos))));
    } catch (...) {
      *error = "tokens contain an invalid integer";
      return false;
    }
    pos = end;
  }
  if (tokens->empty()) {
    *error = "tokens array must not be empty";
    return false;
  }
  return true;
}

std::string csv_tokens(const std::vector<int>& tokens) {
  std::ostringstream out;
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (i) out << ",";
    out << tokens[i];
  }
  return out.str();
}

std::vector<int> top_indices(const std::vector<float>& logits, int top_k) {
  std::vector<int> ids(logits.size());
  for (int i = 0; i < static_cast<int>(ids.size()); ++i) ids[i] = i;
  std::partial_sort(ids.begin(), ids.begin() + top_k, ids.end(),
                    [&](int a, int b) { return logits[a] > logits[b]; });
  ids.resize(top_k);
  return ids;
}

double softmax_denominator(const std::vector<float>& logits, float max_logit) {
  double denom = 0.0;
  for (float value : logits) denom += std::exp(value - max_logit);
  return denom;
}

std::string next_token_json(const DenseConfig& cfg,
                            const DenseDemoRuntime& runtime,
                            const std::vector<float>& logits, int top_k) {
  auto best = std::max_element(logits.begin(), logits.end());
  float max_logit = *best;
  int top_token = static_cast<int>(std::distance(logits.begin(), best));
  double denom = softmax_denominator(logits, max_logit);
  auto ids = top_indices(logits, top_k);
  std::ostringstream out;
  out << "{\"status\":\"success\",\"model_kind\":\"dense\""
      << ",\"decode_supported\":false"
      << ",\"vocab_size\":" << cfg.vocab_size
      << ",\"checksum\":\"" << dense_checksum_floats(logits) << "\""
      << ",\"weights_checksum\":\"" << json_escape(runtime.weights_checksum)
      << "\",\"config_checksum\":\"" << json_escape(runtime.config_checksum)
      << "\",\"optimizer_steps\":" << runtime.optimizer_steps
      << ",\"loss\":" << runtime.loss
      << ",\"parameter_count\":" << runtime.parameter_count
      << ",\"train_report_path\":\"" << json_escape(runtime.train_report_path)
      << "\",\"train_report_digest\":\""
      << json_escape(runtime.train_report_digest) << "\""
      << ",\"top_token\":" << top_token << ",\"top_k\":[";
  for (size_t i = 0; i < ids.size(); ++i) {
    int id = ids[i];
    if (i) out << ",";
    out << "{\"id\":" << id << ",\"logit\":" << logits[id]
        << ",\"prob\":" << (denom > 0.0 ? std::exp(logits[id] - max_logit) / denom : 0.0)
        << "}";
  }
  out << "]}";
  return out.str();
}

}  // namespace

HttpResponse dense_demo_status_response(const ArtifactStatus& artifact) {
  return dense_demo_status_response(load_dense_demo_runtime(artifact, {}));
}

HttpResponse dense_demo_status_response(const DenseDemoRuntime& runtime) {
  bool supported = runtime.dense_supported;
  std::ostringstream out;
  out << "{\"status\":\"" << (supported ? "ready" : "degraded") << "\""
      << ",\"loaded\":" << (runtime.artifact.loaded ? "true" : "false")
      << ",\"dense_supported\":" << (supported ? "true" : "false")
      << ",\"decode_supported\":false"
      << ",\"model\":\"" << json_escape(runtime.artifact.model_name) << "\""
      << ",\"model_kind\":\""
      << json_escape(json_first_string(runtime.manifest, "kind"))
      << "\",\"artifact_error\":\""
      << json_escape(runtime.error.empty() ? runtime.artifact.error
                                           : runtime.error)
      << "\"";
  if (supported) {
    out << ",\"vocab_size\":" << runtime.config.vocab_size
        << ",\"context\":" << runtime.config.context
        << ",\"hidden_size\":" << runtime.config.hidden_size
        << ",\"weights_checksum\":\"" << json_escape(runtime.weights_checksum)
        << "\",\"config_checksum\":\"" << json_escape(runtime.config_checksum)
        << "\",\"optimizer_steps\":" << runtime.optimizer_steps
        << ",\"loss\":" << runtime.loss
        << ",\"parameter_count\":" << runtime.parameter_count
        << ",\"train_report_path\":\""
        << json_escape(runtime.train_report_path)
        << "\",\"train_report_digest\":\""
        << json_escape(runtime.train_report_digest) << "\"";
  }
  out << "}";
  return {200, out.str()};
}

HttpResponse dense_demo_next_token_response(const ArtifactStatus& artifact,
                                            const HttpRequest& request) {
  return dense_demo_next_token_response(load_dense_demo_runtime(artifact, {}),
                                        request);
}

HttpResponse dense_demo_next_token_response(const DenseDemoRuntime& runtime,
                                            const HttpRequest& request) {
  if (!runtime.dense_supported) {
    return {422, error_json("loaded artifact does not support dense logits")};
  }
  std::vector<int> tokens;
  std::string error;
  if (!parse_tokens(request.body, &tokens, &error)) return {400, error_json(error)};
  DenseConfig cfg = runtime.config;
  int top_k = json_int_value(request.body, "top_k", 8);
  if (top_k <= 0) return {400, error_json("top_k must be positive")};
  top_k = std::min(top_k, cfg.vocab_size);
  std::vector<float> logits;
  if (!dense_logits_for_tokens(cfg, runtime.embeddings, runtime.head,
                               csv_tokens(tokens), &logits, &error)) {
    return {400, error_json(error)};
  }
  return {200, next_token_json(cfg, runtime, logits, top_k)};
}

}  // namespace lkjai
