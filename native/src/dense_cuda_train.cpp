#include "dense_cuda.hpp"

#include <cmath>

#include "cuda_probe.hpp"
#include "dense_cuda_internal.hpp"

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
  auto cache = inspect_packed_cache(opt.packed_cache);
  if (!cache.ok) {
    *error = cache.error;
    return false;
  }
  if (cache.vocab_size > cfg.vocab_size) {
    *error = "packed cache vocab_size exceeds dense config vocab_size";
    return false;
  }
  int seq_len = opt.seq_len > 0 ? opt.seq_len : cfg.context;
  if (seq_len > cfg.context) {
    *error = "requested seq_len exceeds dense config context";
    return false;
  }
  try {
    DenseTrainState init;
    init_dense_state(cfg, &init);
    CudaExecutionContext ctx;
    DenseCudaState state(cfg, init, &ctx);
    report->start_step = dense_resume_step(opt.resume_dir);
    float before = init.emb.empty() ? 0.0f : init.emb.front();
    std::vector<float> logits;
    auto started = std::chrono::steady_clock::now();
    for (int local = 1; local <= opt.max_steps; ++local) {
      PackedBatch batch;
      int first = (report->start_step + local - 1) * opt.batch_size;
      auto phase = std::chrono::steady_clock::now();
      if (!load_packed_batch(opt.packed_cache, first, opt.batch_size, seq_len,
                             &batch, error)) return false;
      report->batch_load_seconds += dense_seconds_since(phase);
      double fwd = 0.0, bwd = 0.0;
      report->loss = state.forward_backward(batch, &logits, &fwd, &bwd);
      report->forward_seconds += fwd;
      report->backward_seconds += bwd;
      if (local == 1) report->initial_loss = report->loss;
      if (!std::isfinite(report->loss)) {
        *error = "dense CUDA training produced non-finite loss";
        return false;
      }
      int step = report->start_step + local;
      phase = std::chrono::steady_clock::now();
      state.adamw(dense_step_lr(opt, step), step);
      report->optimizer_seconds += dense_seconds_since(phase);
      report->steps = step;
      report->logits_checksum = dense_checksum_floats(logits);
      if (opt.checkpoint_interval > 0 && step % opt.checkpoint_interval == 0) {
        phase = std::chrono::steady_clock::now();
        auto host = state.copy_to_host();
        if (!write_dense_train_artifact(opt.out_dir / "checkpoints" / "latest",
                                        host, step, report->loss, true,
                                        &report->logits_checksum)) return false;
        report->checkpoint_seconds += dense_seconds_since(phase);
      }
    }
    auto host = state.copy_to_host();
    report->weight_changed =
        !host.emb.empty() && std::fabs(host.emb.front() - before) > 0.0f;
    auto phase = std::chrono::steady_clock::now();
    bool ok = write_dense_train_artifact(opt.out_dir / "checkpoints" / "latest",
                                         host, report->steps, report->loss, true,
                                         &report->logits_checksum) &&
              write_dense_train_artifact(opt.out_dir / "checkpoints" / "final",
                                         host, report->steps, report->loss, true,
                                         &report->logits_checksum);
    report->checkpoint_seconds += dense_seconds_since(phase);
    phase = std::chrono::steady_clock::now();
    auto export_dir = opt.out_dir / "exports" / opt.model_name;
    auto served_dir = opt.out_dir.parent_path() / "models" / opt.model_name;
    ok = ok && write_dense_train_artifact(export_dir, host, report->steps,
                                          report->loss, false,
                                          &report->logits_checksum) &&
         write_dense_train_artifact(served_dir, host, report->steps, report->loss,
                                    false, &report->logits_checksum);
    if (ok && !opt.export_artifact.empty()) {
      ok = write_dense_train_artifact(opt.export_artifact, host, report->steps,
                                      report->loss, false,
                                      &report->logits_checksum);
    }
    report->export_seconds += dense_seconds_since(phase);
    if (!ok) {
      *error = "failed to write dense artifact";
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
