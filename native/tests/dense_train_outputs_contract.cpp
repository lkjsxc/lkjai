#include <filesystem>
#include <iostream>

#include "dense_train_internal.hpp"
#include "train_report_digest.hpp"

int main() {
  auto root = std::filesystem::path("/tmp/lkjai-dense-train-outputs-contract");
  std::filesystem::remove_all(root);
  lkjai::DenseConfig cfg;
  cfg.vocab_size = 16;
  cfg.hidden_size = 8;
  lkjai::DenseTrainState state;
  lkjai::init_dense_state(cfg, &state);
  lkjai::DenseTrainOptions opt;
  opt.out_dir = root / "train";
  opt.model_name = "dense-test";
  lkjai::DenseTrainReport report;
  report.steps = 3;
  report.microsteps = 6;
  report.batch_size = 1;
  report.grad_accum = 2;
  report.loss = 1.0;
  report.best_loss = 0.9;
  report.best_loss_step = 2;
  report.checkpoint_dir = opt.out_dir / "checkpoints" / "latest";
  report.best_checkpoint_dir = opt.out_dir / "checkpoints" / "best";
  report.export_dir = opt.out_dir / "exports" / opt.model_name;
  report.served_dir = root / "models" / opt.model_name;
  std::string error;
  bool ok = lkjai::write_dense_train_outputs(opt, state, state, 4, &report,
                                             &error);
  ok = ok && !lkjai::train_report_manifest_checksum(
                  opt.out_dir / "checkpoints" / "best").empty();
  ok = ok && std::filesystem::is_regular_file(
                 opt.out_dir / "checkpoints" / "final" / "manifest.json");
  if (!ok) std::cerr << error << "\n";
  return ok ? 0 : 1;
}
