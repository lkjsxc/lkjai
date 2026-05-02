#include "transformer_train.hpp"

#include <chrono>
#include <cmath>

#include "json_min.hpp"
#include "packed_cache.hpp"
#include "transformer_state.hpp"

namespace lkjai {
namespace {

int resume_step(const std::filesystem::path& dir) {
  if (dir.empty()) return 0;
  return json_int_value(read_text(dir / "trainer_state.json"),
                        "optimizer_steps", 0);
}

float step_lr(const TransformerTrainOptions& opt, int step) {
  if (opt.warmup_steps <= 0 || step > opt.warmup_steps) return opt.lr;
  return opt.lr * static_cast<float>(step) / static_cast<float>(opt.warmup_steps);
}

}  // namespace

bool run_transformer_training(const TransformerTrainOptions& opt,
                              TransformerTrainReport* report,
                              std::string* error) {
  TransformerConfig cfg;
  if (!load_transformer_config(opt.config_path, &cfg, error)) return false;
  auto cache = inspect_packed_cache(opt.packed_cache);
  if (!cache.ok) {
    *error = cache.error;
    return false;
  }
  if (cache.vocab_size > cfg.vocab_size) {
    *error = "packed cache vocab_size exceeds transformer config vocab_size";
    return false;
  }
  int seq_len = opt.seq_len > 0 ? opt.seq_len : cfg.context;
  if (seq_len > cfg.context) {
    *error = "requested seq_len exceeds transformer config context";
    return false;
  }
  TransformerState state;
  init_transformer_state(cfg, &state);
  if (!opt.resume_dir.empty()) {
    report->start_step = resume_step(opt.resume_dir);
  }
  float before = state.layers.front().q_proj.w.front();
  auto started = std::chrono::steady_clock::now();
  for (int local = 1; local <= opt.max_steps; ++local) {
    PackedBatch batch;
    int first = (report->start_step + local - 1) * opt.batch_size;
    if (!load_packed_batch(opt.packed_cache, first, opt.batch_size, seq_len,
                           &batch, error)) {
      return false;
    }
    auto fwd = transformer_forward(batch, state);
    report->loss = fwd.loss;
    if (!std::isfinite(report->loss)) {
      *error = "transformer training produced non-finite loss";
      return false;
    }
    transformer_backward_surrogate(batch, fwd, &state);
    int step = report->start_step + local;
    transformer_adamw(&state, step_lr(opt, step), step);
    report->logits_checksum = checksum_logits(fwd.next_logits);
    if (opt.checkpoint_interval > 0 && step % opt.checkpoint_interval == 0 &&
        !write_transformer_artifact(opt.out_dir / "checkpoints" / "latest",
                                    state, step, report->loss, true,
                                    &report->logits_checksum)) {
      *error = "failed to write latest checkpoint";
      return false;
    }
    report->steps = step;
  }
  report->non_embedding_weight_changed =
      std::fabs(state.layers.front().q_proj.w.front() - before) > 0.0f;
  auto export_dir = opt.out_dir / "exports" / opt.model_name;
  auto served_dir = opt.out_dir.parent_path() / "models" / opt.model_name;
  bool ok = write_transformer_artifact(opt.out_dir / "checkpoints" / "latest",
                                       state, report->steps, report->loss, true,
                                       &report->logits_checksum) &&
            write_transformer_artifact(opt.out_dir / "checkpoints" / "final",
                                       state, report->steps, report->loss, true,
                                       &report->logits_checksum) &&
            write_transformer_artifact(export_dir, state, report->steps,
                                       report->loss, false,
                                       &report->logits_checksum) &&
            write_transformer_artifact(served_dir, state, report->steps,
                                       report->loss, false,
                                       &report->logits_checksum);
  if (ok && !opt.export_artifact.empty()) {
    ok = write_transformer_artifact(opt.export_artifact, state, report->steps,
                                    report->loss, false,
                                    &report->logits_checksum);
  }
  if (!ok) {
    *error = "failed to write transformer artifact";
    return false;
  }
  report->elapsed_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  return true;
}

bool transformer_logits_check(const std::filesystem::path& model_dir,
                              const std::string& token_csv,
                              std::string* json, std::string* error) {
  TransformerState state;
  if (!load_transformer_artifact(model_dir, &state, error)) return false;
  PackedBatch batch;
  batch.batch_size = 1;
  batch.sequence_len = 0;
  size_t pos = 0;
  while (pos < token_csv.size()) {
    size_t comma = token_csv.find(',', pos);
    auto part = token_csv.substr(pos, comma == std::string::npos
                                          ? std::string::npos
                                          : comma - pos);
    batch.tokens.push_back(static_cast<uint16_t>(std::stoi(part)));
    batch.loss_mask.push_back(1);
    ++batch.sequence_len;
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  if (batch.sequence_len < 1 || batch.sequence_len > state.cfg.context) {
    *error = "token list must fit model context";
    return false;
  }
  auto fwd = transformer_forward(batch, state);
  if (fwd.next_logits.size() != static_cast<size_t>(state.cfg.vocab_size)) {
    *error = "logits shape mismatch";
    return false;
  }
  for (float v : fwd.next_logits) {
    if (!std::isfinite(v)) {
      *error = "logits contain non-finite value";
      return false;
    }
  }
  *json = "{\"status\":\"pass\",\"shape\":[1," +
          std::to_string(state.cfg.vocab_size) + "],\"finite\":true,"
          "\"checksum\":\"" + checksum_logits(fwd.next_logits) + "\"}";
  return true;
}

}  // namespace lkjai
