#include "dense_cuda.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include "cuda_probe.hpp"
#include "dense_cuda_internal.hpp"
#include "dense_loss_trend.hpp"
#include "dense_weight_change.hpp"
#include "json_min.hpp"
namespace lkjai {
bool run_dense_cuda_training(const DenseTrainOptions& opt, DenseTrainReport* report, std::string* error) {
  auto status = cuda_status();
  if (!cuda_required_ok(status)) {
    *error = "CUDA BF16/cuBLASLt capability unavailable: " +
             (status.error.empty() ? status.warning : status.error);
    return false;
  }
  DenseConfig cfg;
  if (!load_dense_config(opt.config_path, &cfg, error)) return false;
  if (opt.seed >= 0) cfg.seed = opt.seed;
  int seq_len = opt.seq_len > 0 ? opt.seq_len : cfg.context;
  if (opt.batch_size <= 0 || opt.grad_accum <= 0 || opt.max_steps <= 0) {
    *error = "batch_size, grad_accum, and max_steps must be positive";
    return false;
  }
  if (seq_len > cfg.context) {
    *error = "requested seq_len exceeds dense config context";
    return false;
  }
  PackedCacheReader reader;
  if (!reader.open(opt.packed_cache, seq_len, cfg.vocab_size, error)) return false;
  if (!packed_cache_allowed_for_run(reader.status(), opt.run_purpose, error)) return false;
  try {
    report->train_config_path = opt.train_config_path;
    report->run_purpose = opt.run_purpose;
    report->config_path = opt.config_path;
    report->packed_cache = opt.packed_cache;
    report->batch_size = opt.batch_size;
    report->seq_len = seq_len;
    report->grad_accum = opt.grad_accum;
    report->target_seconds = opt.target_seconds;
    report->lr_schedule = opt.lr_schedule;
    report->learning_rate = opt.lr;
    report->min_learning_rate_fraction = opt.min_lr_fraction;
    report->loss_sample_interval = opt.loss_sample_interval;
    report->checkpoint_dir = opt.out_dir / "checkpoints" / "latest";
    report->best_checkpoint_dir = opt.out_dir / "checkpoints" / "best";
    report->export_dir = opt.out_dir / "exports" / opt.model_name;
    report->served_dir = opt.out_dir.parent_path() / "models" / opt.model_name;
    int step_rows = opt.batch_size * seq_len;
    report->dense_step_logits_bytes =
        static_cast<uint64_t>(step_rows) * cfg.vocab_size * sizeof(float);
    report->dense_step_grad_logits_bytes = report->dense_step_logits_bytes;
    report->dense_step_d_hidden_bytes =
        static_cast<uint64_t>(step_rows) * cfg.hidden_size * sizeof(float);
    report->dense_logits_readback_bytes =
        static_cast<uint64_t>(cfg.vocab_size) * sizeof(float);
    DenseTrainState init;
    DenseCheckpointMetadata resume;
    if (opt.resume_dir.empty()) {
      init_dense_state(cfg, &init);
    } else if (!load_dense_checkpoint(opt.resume_dir, cfg, opt.batch_size,
                                      seq_len, opt.grad_accum, &init,
                                      &resume, error)) {
      return false;
    }
    CudaExecutionContext ctx;
    DenseCudaState state(cfg, init, &ctx);
    report->start_step = resume.optimizer_steps;
    report->microsteps = resume.microsteps;
    report->best_loss = std::numeric_limits<double>::infinity();
    DenseTrainState best_host;
    bool have_best_host = false;
    std::vector<float> logits;
    auto started = std::chrono::steady_clock::now();
    for (int local = 1; local <= opt.max_steps; ++local) {
      if (local > 1 && opt.target_seconds > 0 &&
          dense_seconds_since(started) >= static_cast<double>(opt.target_seconds)) {
        report->deadline_hit = true;
        report->stop_reason = "wall_clock_deadline";
        break;
      }
      auto phase = std::chrono::steady_clock::now();
      double loss_sum = 0.0;
      int step = report->start_step + local;
      int capture_slot = -1;
      bool capture_step_logits =
          local == opt.max_steps ||
          (opt.checkpoint_interval > 0 && step % opt.checkpoint_interval == 0);
      for (int micro = 0; micro < opt.grad_accum; ++micro) {
        int slot = micro % 3;
        if (micro >= 3) {
          loss_sum += state.slot_loss(slot);
          state.take_deferred_timings(&report->h2d_seconds, &report->forward_seconds, &report->backward_seconds);
        }
        int first = ((report->start_step + local - 1) * opt.grad_accum +
                     micro) * opt.batch_size;
        phase = std::chrono::steady_clock::now();
        size_t items = static_cast<size_t>(opt.batch_size) * seq_len;
        auto pinned = state.prepare_batch_slot(slot, items, items);
        if (!reader.load_batch_into(static_cast<uint64_t>(first),
                                    opt.batch_size, pinned.tokens,
                                    pinned.mask, error)) return false;
        report->batch_load_seconds += dense_seconds_since(phase);
        bool capture = capture_step_logits && micro == opt.grad_accum - 1;
        state.stage_batch_slot(slot, opt.batch_size, seq_len, nullptr);
        state.forward_backward_slot(slot, capture, nullptr, nullptr,
                                    1.0f / opt.grad_accum, micro == 0);
        if (capture) capture_slot = slot;
        report->microsteps += 1;
        report->input_tokens += opt.batch_size * seq_len;
        report->loss_tokens += dense_supervised_count_raw(
            pinned.mask, opt.batch_size, seq_len);
      }
      int pending = std::min(opt.grad_accum, 3);
      for (int i = opt.grad_accum - pending; i < opt.grad_accum; ++i) {
        loss_sum += state.slot_loss(i % 3);
        state.take_deferred_timings(&report->h2d_seconds, &report->forward_seconds, &report->backward_seconds);
      }
      if (capture_slot >= 0) state.slot_logits(capture_slot, &logits);
      report->loss = loss_sum / opt.grad_accum;
      if (local == 1) report->initial_loss = report->loss;
      if (!std::isfinite(report->loss)) {
        *error = "dense CUDA training produced non-finite loss";
        return false;
      }
      if (report->loss < report->best_loss) {
        report->best_loss = report->loss;
        report->best_loss_step = step;
        best_host = state.copy_to_host();
        have_best_host = true;
      }
      bool should_sample = local == 1 || local == opt.max_steps ||
                           (opt.loss_sample_interval > 0 &&
                            step % opt.loss_sample_interval == 0);
      if (should_sample) {
        report->loss_samples.push_back({step, report->loss});
      }
      phase = std::chrono::steady_clock::now();
      report->final_learning_rate = dense_step_lr(opt, step);
      state.adamw(static_cast<float>(report->final_learning_rate), step);
      report->optimizer_seconds += dense_seconds_since(phase);
      report->steps = step;
      if (!logits.empty()) report->logits_checksum = dense_checksum_floats(logits);
      if (opt.checkpoint_interval > 0 && step % opt.checkpoint_interval == 0) {
        phase = std::chrono::steady_clock::now();
        auto host = state.copy_to_host();
        if (!write_dense_train_artifact_staged(
                opt.out_dir / "checkpoints" / "latest", host, step,
                report->microsteps, opt.batch_size, seq_len, opt.grad_accum,
                report->loss, true, &report->logits_checksum)) return false;
        report->checkpoint_seconds += dense_seconds_since(phase);
      }
    }
    if (!report->deadline_hit) report->stop_reason = "max_steps";
    auto host = state.copy_to_host();
    if (!have_best_host) best_host = host;
    report->weight_change = dense_weight_change_report(init, host);
    report->weight_changed = report->weight_change.status == "pass";
    finalize_dense_loss_trend(report);
    dense_fill_runtime_report(state, report);
    if (!write_dense_train_outputs(opt, host, best_host, seq_len, report, error))
      return false;
    std::string logits_json;
    std::string logits_error;
    report->logits_check_passed =
        dense_cuda_logits_check_against_checkpoint(
            report->export_dir, report->checkpoint_dir, "1,2,3", &logits_json,
            &logits_error);
    report->logits_check_json = logits_json;
    report->logits_check_checksum =
        report->logits_check_passed ? json_first_string(logits_json, "checksum")
                                    : "";
    if (!report->logits_check_passed) {
      *error = "exported BF16 logits reference check failed: " + logits_error;
      return false;
    }
    report->elapsed_seconds = dense_seconds_since(started);
    return true;
  } catch (const std::exception& e) {
    *error = e.what();
    return false;
  }
}
}  // namespace lkjai
