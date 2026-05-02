#include "training_config.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string_view>
#include <vector>

#include "json_min.hpp"

namespace lkjai {
namespace {

bool known_key(std::string_view key) {
  static const std::vector<std::string_view> known = {
      "format", "name", "description", "preset", "model_name",
      "native_config", "packed_cache_dir", "objective", "sequence_len",
      "learning_rate", "warmup_steps", "batch_size",
      "gradient_accumulation", "max_optimizer_steps",
      "save_latest_every_optimizer_steps", "seed"};
  return std::find(known.begin(), known.end(), key) != known.end();
}

std::vector<std::string> top_keys(std::string_view text) {
  std::vector<std::string> out;
  for (size_t i = 0; i < text.size(); ++i) {
    if (text[i] != '"') continue;
    std::string key;
    for (++i; i < text.size() && text[i] != '"'; ++i) key.push_back(text[i]);
    size_t j = i + 1;
    while (j < text.size() && std::isspace(static_cast<unsigned char>(text[j]))) {
      ++j;
    }
    if (j < text.size() && text[j] == ':') out.push_back(key);
  }
  return out;
}

bool has_key(std::string_view text, std::string_view key) {
  return text.find("\"" + std::string(key) + "\"") != std::string_view::npos;
}

double json_double_value(std::string_view text, std::string_view key,
                         double fallback) {
  const auto needle = "\"" + std::string(key) + "\"";
  auto pos = text.find(needle);
  if (pos == std::string_view::npos) return fallback;
  pos = text.find(':', pos + needle.size());
  if (pos == std::string_view::npos) return fallback;
  try {
    return std::stod(std::string(text.substr(pos + 1)));
  } catch (...) {
    return fallback;
  }
}

bool reject_unknown_keys(std::string_view text, std::string* error) {
  for (const auto& key : top_keys(text)) {
    if (known_key(key)) continue;
    *error = "unsupported TRAIN_CONFIG key: " + key;
    return false;
  }
  return true;
}

}  // namespace

bool apply_training_config(const std::filesystem::path& path,
                           DenseTrainOptions* opt, std::string* error) {
  auto text = read_text(path);
  if (text.empty()) {
    *error = "empty or missing TRAIN_CONFIG: " + path.string();
    return false;
  }
  if (!reject_unknown_keys(text, error)) return false;
  if (!contains_json_string(text, "format", "lkjai-train-config-v1")) {
    *error = "TRAIN_CONFIG format must be lkjai-train-config-v1";
    return false;
  }
  auto objective = json_first_string(text, "objective");
  if (!objective.empty() && objective != "causal_lm_full") {
    *error = "unsupported TRAIN_CONFIG objective: " + objective;
    return false;
  }
  opt->train_config_path = path;
  auto model_name = json_first_string(text, "model_name");
  if (!model_name.empty()) opt->model_name = model_name;
  auto native_config = json_first_string(text, "native_config");
  if (!native_config.empty()) opt->config_path = native_config;
  auto packed_cache = json_first_string(text, "packed_cache_dir");
  if (!packed_cache.empty()) opt->packed_cache = packed_cache;
  opt->seq_len = json_int_value(text, "sequence_len", opt->seq_len);
  opt->batch_size = json_int_value(text, "batch_size", opt->batch_size);
  opt->grad_accum =
      json_int_value(text, "gradient_accumulation", opt->grad_accum);
  opt->max_steps =
      json_int_value(text, "max_optimizer_steps", opt->max_steps);
  opt->warmup_steps = json_int_value(text, "warmup_steps", opt->warmup_steps);
  opt->checkpoint_interval = json_int_value(
      text, "save_latest_every_optimizer_steps", opt->checkpoint_interval);
  opt->seed = json_int_value(text, "seed", opt->seed);
  if (has_key(text, "learning_rate")) {
    opt->lr = static_cast<float>(json_double_value(text, "learning_rate", opt->lr));
  }
  return true;
}

}  // namespace lkjai
