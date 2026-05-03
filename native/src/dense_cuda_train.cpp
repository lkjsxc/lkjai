#include "dense_cuda.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include "cuda_probe.hpp"
#include "dense_cuda_internal.hpp"
#include "dense_loss_trend.hpp"
#include "json_min.hpp"
namespace lkjai {
bool run_dense_cuda_training(const DenseTrainOptions& opt,
                             DenseTrainReport* report, std::string* error) {
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
  try {
    report->train_config_path = opt.train_config_path;
    report->run_purpose = opt.run_purpose;
    report->config_path = opt.config_path;
    report->packed_cache = opt.packed_cache;
    report->batch_size = opt.batch_size;
    report->seq_len = seq_len;
    report->grad_accum = opt.grad_accum;
    report->loss_sample_interval = opt.loss_sample_interval;
    report->checkpoint_dir = opt.out_dir / "checkpoints" / "latest";
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
    float before = init.emb.empty() ? 0.0f : init.emb.front();
    report->best_loss = std::numeric_limits<double>::infinity();
    std::vector<float> logits;
    auto started = std::chrono::steady_clock::now();
    for (int local = 1; local <= opt.max_steps; ++local) {
      auto phase = std::chrono::steady_clock::now();
      double loss_sum = 0.0;
      int step = report->start_step + local;
      int capture_slot = -1;
      bool capture_step_logits =
          local == opt.max_steps ||
          (opt.checkpoint_interval > 0 && step % opt.checkpoint_interval == 0);
      for (int micro = 0; micro < opt.grad_accum; ++micro) {
        int slot = micro % 3;
        if (micro >= 3) loss_sum += state.slot_loss(slot);
        int first = ((report->start_step + local - 1) * opt.grad_accum +
                     micro) * opt.batch_size;
        phase = std::chrono::steady_clock::now();
        size_t items = static_cast<size_t>(opt.batch_size) * seq_len;
        auto pinned = state.prepare_batch_slot(slot, items, items);
        if (!reader.load_batch_into(static_cast<uint64_t>(first),
                                    opt.batch_size, pinned.tokens,
                                    pinned.mask, error)) return false;
        report->batch_load_seconds += dense_seconds_since(phase);
        double h2d = 0.0, fwd = 0.0, bwd = 0.0;
        bool capture = capture_step_logits && micro == opt.grad_accum - 1;
        state.stage_batch_slot(slot, opt.batch_size, seq_len, &h2d);
        state.forward_backward_slot(slot, capture, &fwd, &bwd,
                                    1.0f / opt.grad_accum, micro == 0);
        if (capture) capture_slot = slot;
        report->h2d_seconds += h2d;
        report->forward_seconds += fwd;
        report->backward_seconds += bwd;
        report->microsteps += 1;
        report->input_tokens += opt.batch_size * seq_len;
        report->loss_tokens += dense_supervised_count_raw(
            pinned.mask, opt.batch_size, seq_len);
      }
      int pending = std::min(opt.grad_accum, 3);
      for (int i = opt.grad_accum - pending; i < opt.grad_accum; ++i) {
        loss_sum += state.slot_loss(i % 3);
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
      }
      bool should_sample = local == 1 || local == opt.max_steps ||
                           (opt.loss_sample_interval > 0 &&
                            step % opt.loss_sample_interval == 0);
      if (should_sample) {
        report->loss_samples.push_back({step, report->loss});
      }
      phase = std::chrono::steady_clock::now();
      state.adamw(dense_step_lr(opt, step), step);
      report->optimizer_seconds += dense_seconds_since(phase);
      report->steps = step;
      if (!logits.empty()) report->logits_checksum = dense_checksum_floats(logits);
      if (opt.checkpoint_interval > 0 && step % opt.checkpoint_interval == 0) {
        phase = std::chrono::steady_clock::now();
        auto host = state.copy_to_host();
        if (!write_dense_train_artifact(opt.out_dir / "checkpoints" / "latest",
                                        host, step, report->microsteps,
                                        opt.batch_size, seq_len, opt.grad_accum,
                                        report->loss, true,
                                        &report->logits_checksum)) return false;
        report->checkpoint_seconds += dense_seconds_since(phase);
      }
    }
    auto host = state.copy_to_host();
    report->weight_changed =
        !host.emb.empty() && std::fabs(host.emb.front() - before) > 0.0f;
    finalize_dense_loss_trend(report);
    auto phase = std::chrono::steady_clock::now();
    bool ok = write_dense_train_artifact(opt.out_dir / "checkpoints" / "latest",
                                         host, report->steps, report->microsteps,
                                         opt.batch_size, seq_len, opt.grad_accum,
                                         report->loss, true,
                                         &report->logits_checksum) &&
              write_dense_train_artifact(opt.out_dir / "checkpoints" / "final",
                                         host, report->steps, report->microsteps,
                                         opt.batch_size, seq_len, opt.grad_accum,
                                         report->loss, true,
                                         &report->logits_checksum);
    report->checkpoint_seconds += dense_seconds_since(phase);
    phase = std::chrono::steady_clock::now();
    ok = ok && write_dense_train_artifact(report->export_dir, host, report->steps,
                                          report->microsteps, opt.batch_size,
                                          seq_len, opt.grad_accum,
                                          report->loss, false,
                                          &report->logits_checksum) &&
         write_dense_train_artifact(report->served_dir, host, report->steps,
                                    report->microsteps, opt.batch_size, seq_len,
                                    opt.grad_accum, report->loss,
                                    false, &report->logits_checksum);
    if (ok && !opt.export_artifact.empty()) {
      ok = write_dense_train_artifact(opt.export_artifact, host, report->steps,
                                      report->microsteps, opt.batch_size,
                                      seq_len, opt.grad_accum, report->loss,
                                      false,
                                      &report->logits_checksum);
    }
    report->export_seconds += dense_seconds_since(phase);
    report->cublaslt_workspace_bytes = state.cublaslt_workspace_bytes();
    if (!ok) {
      *error = "failed to write dense artifact";
      return false;
    }
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
