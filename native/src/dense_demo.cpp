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

bool dense_loaded(const ArtifactStatus& artifact, std::string* manifest) {
  if (!artifact.loaded) return false;
  *manifest = read_text(artifact.model_dir / "manifest.json");
  return contains_json_string(*manifest, "kind", "dense");
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
  std::string manifest;
  bool supported = dense_loaded(artifact, &manifest);
  std::ostringstream out;
  out << "{\"status\":\"" << (supported ? "ready" : "degraded") << "\""
      << ",\"loaded\":" << (artifact.loaded ? "true" : "false")
      << ",\"dense_supported\":" << (supported ? "true" : "false")
      << ",\"decode_supported\":false"
      << ",\"model\":\"" << json_escape(artifact.model_name) << "\""
      << ",\"model_kind\":\"" << json_escape(json_first_string(manifest, "kind"))
      << "\",\"artifact_error\":\"" << json_escape(artifact.error) << "\"";
  if (supported) {
    DenseConfig cfg = dense_config_from_artifact(artifact.model_dir);
    out << ",\"vocab_size\":" << cfg.vocab_size
        << ",\"context\":" << cfg.context
        << ",\"hidden_size\":" << cfg.hidden_size;
  }
  out << "}";
  return {200, out.str()};
}

HttpResponse dense_demo_next_token_response(const ArtifactStatus& artifact,
                                            const HttpRequest& request) {
  std::string manifest;
  if (!dense_loaded(artifact, &manifest)) {
    return {422, error_json("loaded artifact does not support dense logits")};
  }
  std::vector<int> tokens;
  std::string error;
  if (!parse_tokens(request.body, &tokens, &error)) return {400, error_json(error)};
  DenseConfig cfg = dense_config_from_artifact(artifact.model_dir);
  int top_k = json_int_value(request.body, "top_k", 8);
  if (top_k <= 0) return {400, error_json("top_k must be positive")};
  top_k = std::min(top_k, cfg.vocab_size);
  auto emb = read_dense_tensor(artifact.model_dir, "tok_embeddings", &error);
  if (!error.empty()) return {500, error_json(error)};
  auto head = read_dense_tensor(artifact.model_dir, "lm_head", &error);
  if (!error.empty()) return {500, error_json(error)};
  std::vector<float> logits;
  if (!dense_logits_for_tokens(cfg, emb, head, csv_tokens(tokens), &logits,
                               &error)) {
    return {400, error_json(error)};
  }
  return {200, next_token_json(cfg, logits, top_k)};
}

}  // namespace lkjai
