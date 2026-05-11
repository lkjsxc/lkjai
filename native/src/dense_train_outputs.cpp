#include "dense_train_internal.hpp"

#include <chrono>

#include "dense_cuda_internal.hpp"

namespace lkjai {
namespace {

bool write_checkpoint_set(const DenseTrainOptions& opt,
                          const DenseTrainState& final_state,
                          const DenseTrainState& best_state, int seq_len,
                          DenseTrainReport* report) {
  return write_dense_train_artifact_staged(
             opt.out_dir / "checkpoints" / "latest", final_state,
             report->steps, report->microsteps, opt.batch_size, seq_len,
             opt.grad_accum, report->loss, true, &report->logits_checksum) &&
         write_dense_train_artifact_staged(
             opt.out_dir / "checkpoints" / "best", best_state,
             report->best_loss_step, report->microsteps, opt.batch_size,
             seq_len, opt.grad_accum, report->best_loss, true,
             &report->logits_checksum) &&
         write_dense_train_artifact_staged(
             opt.out_dir / "checkpoints" / "final", final_state,
             report->steps, report->microsteps, opt.batch_size, seq_len,
             opt.grad_accum, report->loss, true, &report->logits_checksum);
}

bool write_export_set(const DenseTrainOptions& opt,
                      const DenseTrainState& state, int seq_len,
                      DenseTrainReport* report) {
  bool ok = write_dense_train_artifact_staged(
                report->export_dir, state, report->steps, report->microsteps,
                opt.batch_size, seq_len, opt.grad_accum, report->loss, false,
                &report->logits_checksum) &&
            write_dense_train_artifact_staged(
                report->served_dir, state, report->steps, report->microsteps,
                opt.batch_size, seq_len, opt.grad_accum, report->loss, false,
                &report->logits_checksum);
  if (ok && !opt.export_artifact.empty()) {
    ok = write_dense_train_artifact_staged(
        opt.export_artifact, state, report->steps, report->microsteps,
        opt.batch_size, seq_len, opt.grad_accum, report->loss, false,
        &report->logits_checksum);
  }
  return ok;
}

}  // namespace

bool write_dense_train_outputs(const DenseTrainOptions& opt,
                               const DenseTrainState& final_state,
                               const DenseTrainState& best_state,
                               int seq_len, DenseTrainReport* report,
                               std::string* error) {
  auto phase = std::chrono::steady_clock::now();
  bool ok = write_checkpoint_set(opt, final_state, best_state, seq_len, report);
  report->checkpoint_seconds += dense_seconds_since(phase);
  phase = std::chrono::steady_clock::now();
  ok = ok && write_export_set(opt, final_state, seq_len, report);
  report->export_seconds += dense_seconds_since(phase);
  if (!ok) *error = "failed to write dense artifact";
  return ok;
}

}  // namespace lkjai
