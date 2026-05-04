#include "transformer_train.hpp"

#include "json_min.hpp"

namespace lkjai {
namespace {

bool json_bool(std::string_view text, std::string_view key, bool fallback) {
  auto pos = text.find("\"" + std::string(key) + "\"");
  if (pos == std::string_view::npos) return fallback;
  pos = text.find(':', pos);
  if (pos == std::string_view::npos) return fallback;
  auto rest = text.substr(pos + 1);
  auto t = rest.find("true");
  auto f = rest.find("false");
  if (t != std::string_view::npos && (f == std::string_view::npos || t < f)) {
    return true;
  }
  if (f != std::string_view::npos) return false;
  return fallback;
}

float json_float(std::string_view text, std::string_view key, float fallback) {
  auto pos = text.find("\"" + std::string(key) + "\"");
  if (pos == std::string_view::npos) return fallback;
  pos = text.find(':', pos);
  if (pos == std::string_view::npos) return fallback;
  try {
    return std::stof(std::string(text.substr(pos + 1)));
  } catch (...) {
    return fallback;
  }
}

}  // namespace

bool load_transformer_config(const std::filesystem::path& path,
                             TransformerConfig* config, std::string* error) {
  auto text = read_text(path);
  if (text.empty()) {
    *error = "empty or missing transformer config: " + path.string();
    return false;
  }
  config->model = json_first_string(text, "model");
  if (config->model.empty()) config->model = "native-debug-bf16";
  config->kind = json_first_string(text, "model_kind");
  if (config->kind.empty()) config->kind = "transformer";
  if (config->kind != "transformer" && config->kind != "decoder") {
    *error = "native transformer config model_kind must be transformer or decoder";
    return false;
  }
  config->dtype = json_first_string(text, "dtype");
  if (config->dtype != "bf16") {
    *error = "native transformer config dtype must be bf16";
    return false;
  }
  config->vocab_size = json_int_value(text, "vocab_size", config->vocab_size);
  config->context = json_int_value(text, "context", config->context);
  config->layers = json_int_value(text, "layers", config->layers);
  config->hidden_size = json_int_value(text, "hidden_size", config->hidden_size);
  config->heads = json_int_value(text, "heads", config->heads);
  config->kv_heads = json_int_value(text, "kv_heads", config->kv_heads);
  config->head_dim = json_int_value(text, "head_dim", config->head_dim);
  config->ffn_size = json_int_value(text, "ffn_size", config->ffn_size);
  config->activation = json_first_string(text, "activation");
  if (config->activation.empty()) config->activation = "swiglu";
  config->rope_theta = json_float(text, "rope_theta", config->rope_theta);
  config->rms_norm_eps = json_float(text, "rms_norm_eps", config->rms_norm_eps);
  config->tie_embeddings = json_bool(text, "tie_embeddings", false);
  config->seed = json_int_value(text, "seed", config->seed);
  if (config->vocab_size <= 0 || config->hidden_size <= 0 ||
      config->layers <= 0 || config->context <= 1 || config->heads <= 0 ||
      config->kv_heads <= 0 || config->ffn_size <= 0) {
    *error = "native transformer config has invalid tensor dimensions";
    return false;
  }
  if (config->heads * config->head_dim != config->hidden_size) {
    *error = "heads * head_dim must equal hidden_size";
    return false;
  }
  if (config->heads % config->kv_heads != 0) {
    *error = "heads must be divisible by kv_heads";
    return false;
  }
  if (config->activation != "swiglu") {
    *error = "native transformer activation must be swiglu";
    return false;
  }
  return true;
}

}  // namespace lkjai
