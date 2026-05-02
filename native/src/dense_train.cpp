#include "dense_train.hpp"

#include <chrono>
#include <cmath>

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
  if (config->heads * config->head_dim != config->hidden_size) {
    *error = "heads * head_dim must equal hidden_size";
    return false;
  }
  return true;
}

bool run_dense_training(const DenseTrainOptions& opt, DenseTrainReport* report,
                        std::string* error) {
  DenseConfig cfg;
  if (!load_dense_config(opt.config_path, &cfg, error)) return false;
  auto cache = inspect_packed_cache(opt.packed_cache);
  if (!cache.ok) {
    *error = cache.error;
    return false;
  }
  if (cache.vocab_size > cfg.vocab_size) {
    *error = "packed cache vocab_size exceeds dense config vocab_size";
    return false;
  }
  DenseTrainState state;
  init_dense_state(cfg, &state);
  report->start_step = resume_step(opt.resume_dir);
  int seq_len = opt.seq_len > 0 ? opt.seq_len : cfg.context;
  if (seq_len > cfg.context) {
    *error = "requested seq_len exceeds dense config context";
    return false;
  }
  auto before = state.emb.front();
  auto started = std::chrono::steady_clock::now();
  for (int local = 1; local <= opt.max_steps; ++local) {
    PackedBatch batch;
    auto first = (report->start_step + local - 1) * opt.batch_size;
    if (!load_packed_batch(opt.packed_cache, first, opt.batch_size, seq_len,
                           &batch, error)) {
      return false;
    }
    report->loss = dense_forward_backward(batch, &state);
    if (!std::isfinite(report->loss)) {
      *error = "dense training produced non-finite loss";
      return false;
    }
    int step = report->start_step + local;
    float lr = opt.lr;
    if (opt.warmup_steps > 0 && step <= opt.warmup_steps) {
      lr *= static_cast<float>(step) / static_cast<float>(opt.warmup_steps);
    }
    dense_adamw(&state.emb, &state.m_emb, &state.v_emb, state.grad_emb, lr, step);
    dense_adamw(&state.head, &state.m_head, &state.v_head, state.grad_head,
                lr, step);
    if (opt.checkpoint_interval > 0 && step % opt.checkpoint_interval == 0 &&
        !write_dense_train_artifact(opt.out_dir / "checkpoints" / "latest",
                                    state, step, report->loss, true,
                                    &report->logits_checksum)) {
      *error = "failed to write latest checkpoint";
      return false;
    }
    report->steps = step;
  }
  report->weight_changed = std::fabs(state.emb.front() - before) > 0.0f;
  auto export_dir = opt.out_dir / "exports" / opt.model_name;
  auto served_dir = opt.out_dir.parent_path() / "models" / opt.model_name;
  bool ok = write_dense_train_artifact(opt.out_dir / "checkpoints" / "latest",
                                       state, report->steps, report->loss, true,
                                       &report->logits_checksum) &&
            write_dense_train_artifact(opt.out_dir / "checkpoints" / "final",
                                       state, report->steps, report->loss, true,
                                       &report->logits_checksum) &&
            write_dense_train_artifact(export_dir, state, report->steps,
                                       report->loss, false,
                                       &report->logits_checksum) &&
            write_dense_train_artifact(served_dir, state, report->steps,
                                       report->loss, false,
                                       &report->logits_checksum);
  if (ok && !opt.export_artifact.empty()) {
    ok = write_dense_train_artifact(opt.export_artifact, state, report->steps,
                                    report->loss, false,
                                    &report->logits_checksum);
  }
  if (!ok) {
    *error = "failed to write dense artifact";
    return false;
  }
  report->elapsed_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  return true;
}

}  // namespace lkjai
