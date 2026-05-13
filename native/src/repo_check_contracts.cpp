#include "repo_check.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <string_view>
#include <vector>

#include "json_min.hpp"
#include "transformer_train.hpp"

namespace lkjai {
namespace {
std::string read(const std::filesystem::path& path) {
  std::ifstream file(path);
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

std::vector<std::string> keys(std::string_view text) {
  std::vector<std::string> out;
  int depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (size_t i = 0; i < text.size(); ++i) {
    char ch = text[i];
    if (in_string) {
      if (escaped) escaped = false;
      else if (ch == '\\') escaped = true;
      else if (ch == '"') in_string = false;
      continue;
    }
    if (ch == '{') ++depth;
    else if (ch == '}') --depth;
    else if (ch == '"' && depth == 1) {
      std::string key;
      for (++i; i < text.size() && text[i] != '"'; ++i) key.push_back(text[i]);
      size_t j = i + 1;
      while (j < text.size() && std::isspace(static_cast<unsigned char>(text[j]))) ++j;
      if (j < text.size() && text[j] == ':') out.push_back(key);
    } else if (ch == '"') {
      in_string = true;
    }
  }
  return out;
}

bool known(std::string_view key, const std::vector<std::string_view>& allowed) {
  return std::find(allowed.begin(), allowed.end(), key) != allowed.end();
}

bool check_keys(const std::filesystem::path& path,
                const std::vector<std::string_view>& allowed,
                RepoCheckResult* result) {
  bool ok = true;
  for (const auto& key : keys(read(path))) {
    if (known(key, allowed)) continue;
    result->fail(path.string() + " has unsupported key " + key);
    ok = false;
  }
  return ok;
}

void check_native_config(const std::filesystem::path& path,
                         RepoCheckResult* result) {
  static const std::vector<std::string_view> allowed = {
      "model", "model_kind", "dtype", "vocab_size", "context", "layers",
      "hidden_size", "heads", "kv_heads", "head_dim", "ffn_size",
      "activation", "rope_theta", "rms_norm_eps", "tie_embeddings", "seed"};
  if (!check_keys(path, allowed, result)) return;
  TransformerConfig cfg;
  std::string error;
  if (!load_transformer_config(path, &cfg, &error)) {
    result->fail(path.string() + " invalid native config: " + error);
    return;
  }
  if (cfg.head_dim % 8 != 0)
    result->fail(path.string() + " head_dim must be a multiple of 8");
  if (cfg.context > 4096)
    result->fail(path.string() + " context exceeds current native bound");
}

void check_training_config(const std::filesystem::path& repo,
                           const std::filesystem::path& path,
                           RepoCheckResult* result) {
  static const std::vector<std::string_view> allowed = {
      "format", "name", "description", "preset", "model_name", "model_kind",
      "native_config", "packed_cache_dir", "tokenizer", "objective",
      "sequence_len", "learning_rate", "lr_schedule",
      "min_learning_rate_fraction", "warmup_steps", "batch_size",
      "gradient_accumulation", "max_optimizer_steps",
      "save_latest_every_optimizer_steps", "target_seconds", "seed"};
  auto body = read(path);
  if (!check_keys(path, allowed, result)) return;
  if (!contains_json_string(body, "format", "lkjai-train-config"))
    result->fail(path.string() + " missing lkjai-train-config format");
  auto native = json_first_string(body, "native_config");
  if (native.empty()) {
    result->fail(path.string() + " missing native_config");
  } else if (native[0] == '/' || native.find("..") != std::string::npos ||
             !std::filesystem::is_regular_file(repo / native)) {
    result->fail(path.string() + " native_config must be repo-local");
  }
  auto objective = json_first_string(body, "objective");
  if (objective != "causal_lm_full")
    result->fail(path.string() + " unsupported objective " + objective);
  auto model_kind = json_first_string(body, "model_kind");
  auto schedule = json_first_string(body, "lr_schedule");
  if (!schedule.empty() && schedule != "warmup_constant" &&
      schedule != "warmup_cosine")
    result->fail(path.string() + " unsupported lr_schedule " + schedule);
  auto name = json_first_string(body, "name");
  bool profile = path.filename().string().find("profile") != std::string::npos ||
                 name.find("profile") != std::string::npos;
  if (profile && native == "configs/native/decoder_40m_bf16_3070.json")
    result->fail(path.string() + " profile config uses acceptance decoder native config");
  if (model_kind == "decoder" && !native.empty() && native[0] != '/') {
    TransformerConfig cfg;
    std::string error;
    if (!load_transformer_config(repo / native, &cfg, &error)) {
      result->fail(path.string() + " invalid decoder native config: " + error);
    } else {
      if (cfg.kind != "decoder")
        result->fail(path.string() + " decoder training must point to decoder native config");
      if (!cfg.tie_embeddings)
        result->fail(path.string() + " decoder acceptance config must stay tied");
      int seq = json_int_value(body, "sequence_len", 0);
      if (seq <= 0 || seq > cfg.context)
        result->fail(path.string() + " sequence_len exceeds decoder context");
    }
    if (json_int_value(body, "target_seconds", 0) <= 0)
      result->fail(path.string() + " decoder training target_seconds must be positive");
  }
}

bool contains(const std::filesystem::path& path, std::string_view needle) {
  return read(path).find(needle) != std::string::npos;
}

void require_contains(const std::filesystem::path& path, std::string_view needle,
                      RepoCheckResult* result) {
  if (!contains(path, needle))
    result->fail(path.string() + " missing " + std::string(needle));
}

void check_decoder_acceptance_config(const std::filesystem::path& repo,
                                     RepoCheckResult* result) {
  auto native = repo / "configs/native/decoder_40m_bf16_3070.json";
  auto train = repo / "configs/training/decoder_2h_40m_3070.json";
  if (!std::filesystem::is_regular_file(native))
    result->fail("missing tied 40M decoder native config");
  if (!std::filesystem::is_regular_file(train))
    result->fail("missing tied 40M decoder training config");
  require_contains(native, "\"tie_embeddings\": true", result);
  require_contains(train, "decoder_2h_40m_3070", result);
  require_contains(train, "\"model_kind\": \"decoder\"", result);
  require_contains(train, "\"target_seconds\": 7200", result);
  require_contains(train, "configs/native/decoder_40m_bf16_3070.json", result);
}
void check_decoder_train_compose_contract(const std::filesystem::path& repo,
                                          RepoCheckResult* result) {
  auto compose = repo / "compose.yaml";
  auto train = repo / "configs/training/decoder_2h_40m_3070.json";
  for (auto needle : {"MODEL_NAME: ${TRAIN_MODEL_NAME:-decoder-2h-40m-3070}",
                      "TRAIN_CONFIG: ${TRAIN_CONFIG:-/workspace/configs/training/decoder_2h_40m_3070.json}",
                      "TRAIN_NATIVE_CONFIG: ${TRAIN_NATIVE_CONFIG:-/workspace/configs/native/decoder_40m_bf16_3070.json}",
                      "command: [\"--train\", \"--mode\", \"decoder\"]"})
    require_contains(compose, needle, result);
  for (auto needle : {"\"target_seconds\": 7200", "\"save_latest_every_optimizer_steps\": 512",
                      "\"model_name\": \"decoder-2h-40m-3070\""})
    require_contains(train, needle, result);
}
}  // namespace
int check_config_contract(const std::filesystem::path& repo) {
  RepoCheckResult result;
  for (const auto& entry : std::filesystem::directory_iterator(repo / "configs/native")) {
    if (entry.path().extension() == ".json") check_native_config(entry.path(), &result);
  }
  for (const auto& entry : std::filesystem::directory_iterator(repo / "configs/training")) {
    if (entry.path().extension() == ".json")
      check_training_config(repo, entry.path(), &result);
  }
  check_decoder_acceptance_config(repo, &result);
  check_decoder_train_compose_contract(repo, &result);
  return result.errors == 0 ? 0 : 1;
}
int check_cuda_arch_contract(const std::filesystem::path& repo) {
  RepoCheckResult result;
  auto cmake = repo / "native" / "CMakeLists.txt";
  for (auto required : {"86-real", "86-virtual", "89-real", "89-virtual",
                        "90-real", "90-virtual", "120-real", "120-virtual",
                        "LKJAI_CUDA_ARCHS", "CMAKE_CUDA_ARCHITECTURES"}) {
    require_contains(cmake, required, &result);
  }
  require_contains(repo / "compose.yaml", "LKJAI_CUDA_ARCHS", &result);
  require_contains(repo / "ops/docker/Dockerfile.native", "LKJAI_CUDA_ARCHS", &result);
  require_contains(repo / "ops/docker/Dockerfile.verify", "LKJAI_CUDA_ARCHS", &result);
  return result.errors == 0 ? 0 : 1;
}
}  // namespace lkjai
