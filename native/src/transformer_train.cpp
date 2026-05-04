#include "transformer_train.hpp"

#include <chrono>
#include <cmath>

#include "cuda_probe.hpp"
#include "json_min.hpp"
#include "packed_cache.hpp"
#include "transformer_state.hpp"

namespace lkjai {
namespace {

float step_lr(const TransformerTrainOptions& opt, int step) {
  if (opt.warmup_steps <= 0 || step > opt.warmup_steps) return opt.lr;
  return opt.lr * static_cast<float>(step) / static_cast<float>(opt.warmup_steps);
}

double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

}  // namespace

bool run_transformer_training(const TransformerTrainOptions& opt,
                              TransformerTrainReport* report,
                              std::string* error) {
  if (opt.model_kind == "decoder") {
    return run_decoder_cuda_slice_training(opt, report, error);
  }
  TransformerConfig cfg;
  if (!load_transformer_config(opt.config_path, &cfg, error)) return false;
  auto cuda = cuda_status();
  if (!cuda_required_ok(cuda)) {
    *error = "CUDA BF16/cuBLASLt capability unavailable: " +
             (cuda.error.empty() ? cuda.warning : cuda.error);
    return false;
  }
  if (!transformer_cuda_step_probe(error)) return false;
  if (opt.model_kind != "transformer" && opt.model_kind != "decoder") {
    *error = "transformer trainer model_kind must be transformer or decoder";
    return false;
  }
  cfg.kind = opt.model_kind;
  if (opt.seed >= 0) cfg.seed = opt.seed;
  if (cfg.tie_embeddings) {
    *error = "transformer mode requires tie_embeddings=false";
    return false;
  }
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
  if (opt.batch_size <= 0 || opt.grad_accum <= 0 || opt.max_steps <= 0) {
    *error = "batch_size, grad_accum, and max_steps must be positive";
    return false;
  }
  if (seq_len != cache.sequence_len) {
    *error = "requested seq_len must match packed cache sequence_len";
    return false;
  }
  TransformerState state;
  report->train_config_path = opt.train_config_path;
  report->run_purpose = opt.run_purpose;
  report->config_path = opt.config_path; report->model_kind = opt.model_kind;
  report->packed_cache = opt.packed_cache; report->batch_size = opt.batch_size;
  report->seq_len = seq_len; report->grad_accum = opt.grad_accum;
  report->layers = cfg.layers; report->heads = cfg.heads;
  report->kv_heads = cfg.kv_heads; report->hidden_size = cfg.hidden_size;
  report->head_dim = cfg.head_dim; report->ffn_size = cfg.ffn_size;
  report->context = cfg.context; report->target_seconds = opt.target_seconds;
  report->checkpoint_dir = opt.out_dir / "checkpoints" / "latest";
  report->export_dir = opt.out_dir / "exports" / opt.model_name;
  report->served_dir = opt.out_dir.parent_path() / "models" / opt.model_name;
  int resume_microsteps = 0;
  if (!opt.resume_dir.empty()) {
    if (!load_transformer_checkpoint(opt.resume_dir, cfg, opt.batch_size,
                                     seq_len, opt.grad_accum, &state,
                                     &report->start_step, &resume_microsteps,
                                     error)) {
      return false;
    }
  } else {
    init_transformer_state(cfg, &state);
  }
  report->microsteps = resume_microsteps;
  report->parameter_count = transformer_parameter_count(state);
  float before = state.layers.front().q_proj.w.front();
  auto started = std::chrono::steady_clock::now();
  for (int local = 1; local <= opt.max_steps; ++local) {
    if (opt.target_seconds > 0 &&
        seconds_since(started) >= static_cast<double>(opt.target_seconds)) {
      report->deadline_hit = true;
      report->stop_reason = "wall_clock_deadline";
      break;
    }
    double loss_sum = 0.0;
    ForwardResult fwd;
    for (int micro = 0; micro < opt.grad_accum; ++micro) {
      PackedBatch batch;
      int first = ((report->start_step + local - 1) * opt.grad_accum + micro) *
                  opt.batch_size;
      auto phase = std::chrono::steady_clock::now();
      if (!load_packed_batch(opt.packed_cache, first, opt.batch_size, seq_len,
                             &batch, error)) {
        return false;
      }
      report->batch_load_seconds += seconds_since(phase);
      phase = std::chrono::steady_clock::now();
      fwd = transformer_forward(batch, state);
      report->forward_seconds += seconds_since(phase);
      loss_sum += fwd.loss;
      report->microsteps += 1;
      report->input_tokens += opt.batch_size * seq_len;
      report->loss_tokens += fwd.supervised;
      phase = std::chrono::steady_clock::now();
      transformer_backward(batch, fwd, &state);
      report->backward_seconds += seconds_since(phase);
    }
    report->loss = loss_sum / opt.grad_accum;
    if (local == 1) report->initial_loss = report->loss;
    if (!std::isfinite(report->loss)) {
      *error = "transformer training produced non-finite loss";
      return false;
    }
    int step = report->start_step + local;
    auto phase = std::chrono::steady_clock::now();
    transformer_adamw(&state, step_lr(opt, step), step);
    report->optimizer_seconds += seconds_since(phase);
    report->logits_checksum = checksum_logits(fwd.next_logits);
    if (opt.checkpoint_interval > 0 && step % opt.checkpoint_interval == 0) {
      phase = std::chrono::steady_clock::now();
      if (!write_transformer_artifact(opt.out_dir / "checkpoints" / "latest",
                                      state, step, report->microsteps,
                                      opt.batch_size, seq_len, opt.grad_accum,
                                      report->loss, true,
                                      &report->logits_checksum)) {
        *error = "failed to write latest checkpoint";
        return false;
      }
      report->checkpoint_export_seconds += seconds_since(phase);
    }
    report->steps = step;
  }
  if (!report->deadline_hit) report->stop_reason = "max_steps";
  report->non_embedding_weight_changed =
      std::fabs(state.layers.front().q_proj.w.front() - before) > 0.0f;
  report->trainable_weight_changed = report->non_embedding_weight_changed;
  auto export_dir = opt.out_dir / "exports" / opt.model_name;
  auto served_dir = opt.out_dir.parent_path() / "models" / opt.model_name;
  auto phase = std::chrono::steady_clock::now();
  auto write = [&](const std::filesystem::path& dir, bool checkpoint) {
    return write_transformer_artifact(dir, state, report->steps,
                                      report->microsteps, opt.batch_size,
                                      seq_len, opt.grad_accum, report->loss,
                                      checkpoint, &report->logits_checksum);
  };
  bool ok = write(opt.out_dir / "checkpoints" / "latest", true) &&
            write(opt.out_dir / "checkpoints" / "final", true) &&
            write(export_dir, false) && write(served_dir, false);
  if (ok && !opt.export_artifact.empty()) {
    ok = write_transformer_artifact(opt.export_artifact, state, report->steps,
                                    report->microsteps, opt.batch_size, seq_len,
                                    opt.grad_accum, report->loss, false,
                                    &report->logits_checksum);
  }
  report->checkpoint_export_seconds += seconds_since(phase);
  report->export_seconds = report->checkpoint_export_seconds;
  if (!ok) {
    *error = "failed to write transformer artifact";
    return false;
  }
  report->elapsed_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  std::string logits_json;
  std::string logits_error;
  report->logits_check_passed =
      transformer_logits_check(export_dir, "1,2,3", &logits_json, &logits_error);
  report->logits_check_json = logits_json;
  report->logits_check_checksum =
      report->logits_check_passed ? json_first_string(logits_json, "checksum") : "";
  if (!report->logits_check_passed) {
    *error = "exported transformer BF16 logits check failed: " + logits_error;
    return false;
  }
  return true;
}

}  // namespace lkjai
