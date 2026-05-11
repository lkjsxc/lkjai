#include "transformer_train.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include "cuda_probe.hpp"
#include "decoder_cuda_block.hpp"
#include "decoder_cuda_slice_internal.hpp"
#include "decoder_cuda_state.hpp"
#include "dense_cuda_internal.hpp"
#include "json_min.hpp"
#include "native_tokenizer.hpp"
#include "packed_cache.hpp"
#include "runtime_device.hpp"
#include "transformer_state.hpp"

namespace lkjai {
namespace {

double since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

int decoder_trainable_tensor_count(const TransformerConfig& cfg) {
  return 1 + cfg.layers * 9 + 1 + (cfg.tie_embeddings ? 0 : 1);
}

float lr_at(const TransformerTrainOptions& opt, int step) {
  if (opt.warmup_steps <= 0 || step > opt.warmup_steps) return opt.lr;
  return opt.lr * static_cast<float>(step) / static_cast<float>(opt.warmup_steps);
}

std::filesystem::path default_tokenizer_for_config(
    const std::filesystem::path& config_path) {
  auto abs = std::filesystem::absolute(config_path);
  auto native_dir = abs.parent_path();
  auto configs_dir = native_dir.parent_path();
  auto repo = configs_dir.parent_path();
  return repo / "data" / "train" / "tokenizer" / "tokenizer.json";
}

}  // namespace

bool run_decoder_cuda_training(const TransformerTrainOptions& opt,
                               TransformerTrainReport* report,
                               std::string* error) {
  TransformerConfig cfg;
  if (!load_transformer_config(opt.config_path, &cfg, error)) return false;
  if (opt.model_kind != "decoder") {
    *error = "decoder CUDA slice requires model_kind=decoder";
    return false;
  }
  if (!decoder_validate_layer_shapes(cfg, error)) return false;
  auto status = cuda_status();
  if (!cuda_required_ok(status)) {
    *error = "CUDA BF16/cuBLASLt capability unavailable: " +
             (status.error.empty() ? status.warning : status.error);
    return false;
  }
  DecoderCudaForwardSubstrateReport substrate;
  if (!decoder_cuda_forward_substrate_probe(cfg, &substrate, error)) {
    *error = "decoder CUDA forward substrate probe failed: " + *error;
    return false;
  }
  if (opt.seed >= 0) cfg.seed = opt.seed;
  cfg.kind = "decoder";
  auto tokenizer_path = opt.tokenizer_path.empty()
                            ? default_tokenizer_for_config(opt.config_path)
                            : opt.tokenizer_path;
  NativeTokenizer tokenizer;
  if (!load_native_tokenizer(tokenizer_path, &tokenizer, error)) return false;
  if (!validate_decoder_tokenizer(tokenizer, cfg.vocab_size, error)) return false;
  auto effective = opt;
  effective.tokenizer_path = tokenizer_path;
  int seq_len = opt.seq_len > 0 ? opt.seq_len : cfg.context;
  if (opt.batch_size <= 0 || opt.grad_accum <= 0 || opt.max_steps <= 0) {
    *error = "batch_size, grad_accum, and max_steps must be positive";
    return false;
  }
  if (seq_len > cfg.context) {
    *error = "requested seq_len exceeds decoder config context";
    return false;
  }
  PackedCacheReader reader;
  if (!reader.open(opt.packed_cache, seq_len, cfg.vocab_size, error)) return false;
  if (!packed_cache_allowed_for_run(reader.status(), opt.run_purpose, error)) return false;
  TransformerState host_state;
  init_transformer_state(cfg, &host_state);
  auto before_state = host_state;
  DecoderCudaState cuda(host_state.cfg, host_state);
  report->train_config_path = opt.train_config_path;
  report->run_purpose = opt.run_purpose;
  report->config_path = opt.config_path;
  report->model_kind = "decoder";
  report->packed_cache = opt.packed_cache;
  report->batch_size = opt.batch_size;
  report->seq_len = seq_len;
  report->grad_accum = opt.grad_accum;
  report->layers = cfg.layers;
  report->heads = cfg.heads;
  report->kv_heads = cfg.kv_heads;
  report->hidden_size = cfg.hidden_size;
  report->head_dim = cfg.head_dim;
  report->ffn_size = cfg.ffn_size;
  report->context = cfg.context;
  report->embedding_tying =
      cfg.tie_embeddings ? "tok_embeddings:lm_head" : "none";
  report->trainable_tensor_count = decoder_trainable_tensor_count(cfg);
  report->target_seconds = opt.target_seconds;
  report->checkpoint_dir = opt.out_dir / "checkpoints" / "latest";
  report->export_dir = opt.out_dir / "exports" / opt.model_name;
  report->served_dir = opt.out_dir.parent_path() / "models" / opt.model_name;
  report->parameter_count = transformer_parameter_count(host_state);
  decoder_set_forward_probe(substrate, report);
  auto started = std::chrono::steady_clock::now();
  std::vector<float> logits;
  for (int local = 1; local <= opt.max_steps; ++local) {
    if (opt.target_seconds > 0 &&
        since(started) >= static_cast<double>(opt.target_seconds)) {
      report->deadline_hit = true;
      report->stop_reason = "wall_clock_deadline";
      break;
    }
    double loss_sum = 0.0;
    for (int micro = 0; micro < opt.grad_accum; ++micro) {
      PackedBatch batch;
      int first = ((local - 1) * opt.grad_accum + micro) * opt.batch_size;
      auto phase = std::chrono::steady_clock::now();
      if (!reader.load_batch(first, opt.batch_size, &batch, error)) return false;
      report->batch_load_seconds += since(phase);
      if (local == 1 && micro == 0) {
        DecoderCudaForwardSubstrateReport block;
        if (!decoder_cuda_slice_run_block_forward(host_state, batch, &block,
                                                  error)) {
          *error = "decoder training block forward failed: " + *error;
          return false;
        }
        report->decoder_block_forward_in_training = true;
        report->decoder_block_forward_steps = 1;
        report->workspace_high_water_bytes =
            std::max<uint64_t>(report->workspace_high_water_bytes,
                               block.projection_workspace_bytes);
        decoder_set_forward_probe(block, report);
      }
      bool capture = local == opt.max_steps && micro == opt.grad_accum - 1;
      loss_sum += cuda.forward_backward(
          batch, capture ? &logits : nullptr, &report->h2d_seconds,
          &report->forward_seconds, &report->backward_seconds,
          1.0f / opt.grad_accum, micro == 0);
      report->microsteps += 1;
      report->input_tokens += opt.batch_size * seq_len;
      report->loss_tokens += dense_supervised_count(batch);
    }
    report->steps = local;
    report->loss = loss_sum / opt.grad_accum;
    if (local == 1) report->initial_loss = report->loss;
    if (!std::isfinite(report->loss)) {
      *error = "decoder CUDA slice produced non-finite loss";
      return false;
    }
    auto phase = std::chrono::steady_clock::now();
    cuda.optimizer_step(lr_at(opt, local), local);
    report->optimizer_seconds += since(phase);
    if (!logits.empty()) report->logits_checksum = dense_checksum_floats(logits);
  }
  if (!report->deadline_hit) report->stop_reason = "max_steps";
  auto trained_state = cuda.copy_to_host();
  cuda.record_weight_change(before_state, report);
  auto phase = std::chrono::steady_clock::now();
  if (!decoder_write_all(effective, trained_state, report, seq_len)) {
    *error = "failed to write decoder CUDA slice artifact";
    return false;
  }
  report->checkpoint_export_seconds += since(phase);
  report->export_seconds = report->checkpoint_export_seconds;
  cuda.fill_report(report);
  report->elapsed_seconds = since(started);
  std::string logits_json, logits_error;
  report->logits_check_passed =
      transformer_logits_check(report->export_dir, "1,2,3", &logits_json,
                               &logits_error);
  report->logits_check_json = logits_json;
  report->logits_check_checksum =
      report->logits_check_passed ? json_first_string(logits_json, "checksum") : "";
  if (!report->logits_check_passed) {
    *error = "exported decoder BF16 logits check failed: " + logits_error;
    return false;
  }
  return true;
}

}  // namespace lkjai
