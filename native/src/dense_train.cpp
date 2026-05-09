#include "dense_train.hpp"

#include <chrono>
#include <cmath>

#include "dense_cuda.hpp"
#include "dense_train_internal.hpp"
#include "json_min.hpp"
#include "packed_cache.hpp"

namespace lkjai {
namespace {

int resume_step(const std::filesystem::path& dir) {
  if (dir.empty()) return 0;
  return json_int_value(read_text(dir / "trainer_state.json"),
                        "optimizer_steps", 0);
}

}  // namespace

bool load_dense_config(const std::filesystem::path& path, DenseConfig* config,
                       std::string* error) {
  auto text = read_text(path);
  if (text.empty()) {
    *error = "empty or missing dense config: " + path.string();
    return false;
  }
  config->model = json_first_string(text, "model");
  if (config->model.empty()) config->model = "dense-debug-bf16";
  if (!contains_json_string(text, "dtype", "bf16")) {
    *error = "native dense config dtype must be bf16";
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
  config->seed = json_int_value(text, "seed", config->seed);
  if (config->vocab_size <= 0 || config->hidden_size <= 0 ||
      config->layers <= 0 || config->context <= 1) {
    *error = "native dense config has invalid tensor dimensions";
    return false;
  }
  if (config->head_dim % 8 != 0) {
    *error = "head_dim must be a multiple of 8";
    return false;
  }
  if (config->heads * config->head_dim != config->hidden_size) {
    *error = "heads * head_dim must equal hidden_size";
    return false;
  }
  return true;
}

bool run_dense_training(const DenseTrainOptions& opt, DenseTrainReport* report,
                        std::string* error) {
  return run_dense_cuda_training(opt, report, error);
}

}  // namespace lkjai
