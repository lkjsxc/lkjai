#include "decoder_cuda_state.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "decoder_cuda_block_internal.hpp"
#include "decoder_cuda_norm.hpp"

namespace lkjai {
namespace {

constexpr size_t kWorkspaceBytes = 4 * 1024 * 1024;
constexpr double kLossTolerance = 0.15;
constexpr double kLogitsMaxTolerance = 0.08;
constexpr double kLogitsMeanTolerance = 0.025;

double since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

int last_supervised_row(const PackedBatch& batch) {
  int last = -1;
  for (int row = 0; row < batch.batch_size; ++row) {
    int base = row * batch.sequence_len;
    for (int pos = 0; pos + 1 < batch.sequence_len; ++pos) {
      if (batch.loss_mask[static_cast<size_t>(base + pos + 1)] != 0) {
        last = base + pos;
      }
    }
  }
  return last;
}

void compare_logits(const std::vector<float>& got,
                    const std::vector<float>& want) {
  if (got.size() != want.size()) {
    throw std::runtime_error("decoder CUDA logits parity size mismatch");
  }
  double max_abs = 0.0;
  double mean_abs = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    if (!std::isfinite(got[i])) {
      throw std::runtime_error("decoder CUDA logits contain non-finite value");
    }
    double diff = std::abs(static_cast<double>(got[i]) - want[i]);
    max_abs = std::max(max_abs, diff);
    mean_abs += diff;
  }
  if (!got.empty()) mean_abs /= static_cast<double>(got.size());
  if (max_abs > kLogitsMaxTolerance || mean_abs > kLogitsMeanTolerance) {
    throw std::runtime_error("decoder CUDA logits parity check failed");
  }
}

}  // namespace

double DecoderCudaState::forward_backward(
    const PackedBatch& batch, std::vector<float>* logits, double* h2d_seconds,
    double* forward_seconds, double* backward_seconds, float grad_scale,
    bool reset_grads) {
  const auto& cfg = state_.cfg;
  int rows = batch.batch_size * batch.sequence_len;
  int supervised = dense_supervised_count(batch);
  int capture_row = last_supervised_row(batch);
  ensure_tape_capacity(rows, cfg.vocab_size, cfg.hidden_size, cfg.layers);

  auto phase = std::chrono::steady_clock::now();
  require_cuda(cudaMemcpyAsync(tape_.device_tokens, batch.tokens.data(),
                               batch.tokens.size() * sizeof(uint16_t),
                               cudaMemcpyHostToDevice, ctx_.copy_stream()),
               "decoder train tokens H2D");
  require_cuda(cudaMemcpyAsync(tape_.device_loss_mask, batch.loss_mask.data(),
                               batch.loss_mask.size(), cudaMemcpyHostToDevice,
                               ctx_.copy_stream()),
               "decoder train mask H2D");
  require_cuda(cudaStreamSynchronize(ctx_.copy_stream()),
               "decoder train H2D sync");
  if (h2d_seconds) *h2d_seconds += since(phase);

  auto find = [&](const std::string& name) -> RegistryTensor* {
    for (auto& t : registry_) {
      if (t.name == name) return &t;
    }
    throw std::runtime_error("missing decoder CUDA registry tensor: " + name);
  };

  phase = std::chrono::steady_clock::now();
  require_cuda(cudaMemsetAsync(tape_.loss.data(), 0, sizeof(float),
                               ctx_.stream()),
               "decoder train loss zero");
  dense_launch_gather(tape_.device_tokens, find("tok_embeddings")->shadow.data(),
                      tape_.embeddings.data(), batch.batch_size,
                      batch.sequence_len, cfg.vocab_size, cfg.hidden_size,
                      ctx_.stream());
  DeviceTensor* hidden = &tape_.embeddings;
  for (int i = 0; i < cfg.layers; ++i) {
    DecoderCudaForwardSubstrateReport layer_report;
    DeviceTensor out;
    layer_forwards_[static_cast<size_t>(i)].run(
        *hidden, batch.batch_size, batch.sequence_len, &out, &layer_report);
    if (!layer_report.outputs_finite) {
      throw std::runtime_error("decoder CUDA training layer forward failed");
    }
    tape_.layers[static_cast<size_t>(i)].block_residual = std::move(out);
    hidden = &tape_.layers[static_cast<size_t>(i)].block_residual;
  }
  require_cuda(cudaMemcpyAsync(tape_.final_norm_input.data(), hidden->data(),
                               static_cast<size_t>(rows) * cfg.hidden_size *
                                   sizeof(uint16_t),
                               cudaMemcpyDeviceToDevice, ctx_.stream()),
               "decoder train final norm input copy");
  decoder_launch_rmsnorm_bf16(
      tape_.final_norm_input.data(),
      static_cast<float*>(find("final_norm")->weight.data()),
      tape_.final_norm.data(), rows, cfg.hidden_size, cfg.rms_norm_eps,
      ctx_.stream());
  auto* head = state_.cfg.tie_embeddings ? find("tok_embeddings")
                                         : find("lm_head");
  void* ws = workspace_.allocate(kWorkspaceBytes);
  decoder_cuda_project_bf16(ctx_.cublaslt(), ctx_.stream(),
                            tape_.final_norm.data(), head->shadow.data(),
                            tape_.logits_bf16.data(), rows, cfg.hidden_size,
                            cfg.vocab_size, ws, kWorkspaceBytes);
  dense_launch_bf16_to_f32(tape_.logits_bf16.data(),
                           static_cast<float*>(tape_.logits.data()),
                           rows * cfg.vocab_size, ctx_.stream());
  dense_launch_loss_grad(static_cast<float*>(tape_.logits.data()),
                         tape_.device_tokens, tape_.device_loss_mask,
                         static_cast<float*>(tape_.grad_logits.data()),
                         static_cast<float*>(tape_.loss.data()),
                         batch.batch_size, batch.sequence_len, cfg.vocab_size,
                         supervised, grad_scale, ctx_.stream());
  require_cuda(cudaMemcpyAsync(tape_.host_loss, tape_.loss.data(),
                               sizeof(float), cudaMemcpyDeviceToHost,
                               ctx_.stream()),
               "decoder train loss D2H");
  if (logits && capture_row >= 0) {
    auto* source = static_cast<float*>(tape_.logits.data()) +
                   static_cast<size_t>(capture_row) * cfg.vocab_size;
    require_cuda(cudaMemcpyAsync(tape_.host_logits, source,
                                 static_cast<size_t>(cfg.vocab_size) *
                                     sizeof(float),
                                 cudaMemcpyDeviceToHost, ctx_.stream()),
                 "decoder train logits D2H");
  }
  require_cuda(cudaStreamSynchronize(ctx_.stream()),
               "decoder train forward sync");
  if (forward_seconds) *forward_seconds += since(phase);

  double cuda_loss = static_cast<double>(tape_.host_loss[0]);
  if (logits) {
    if (capture_row >= 0) {
      logits->assign(tape_.host_logits, tape_.host_logits + cfg.vocab_size);
    } else {
      logits->assign(static_cast<size_t>(cfg.vocab_size), 0.0f);
    }
  }

  auto host_fwd = transformer_forward(batch, state_);
  if (!std::isfinite(cuda_loss) ||
      std::abs(cuda_loss - host_fwd.loss) > kLossTolerance) {
    throw std::runtime_error("decoder CUDA loss parity check failed");
  }
  if (logits && capture_row >= 0) compare_logits(*logits, host_fwd.loss_logits);

  phase = std::chrono::steady_clock::now();
  run_device_backward(static_cast<float>(cuda_loss), rows, capture_row,
                      grad_scale, reset_grads);
  require_cuda(cudaStreamSynchronize(ctx_.stream()),
               "decoder train backward sync");
  if (backward_seconds) *backward_seconds += since(phase);
  return cuda_loss;
}

}  // namespace lkjai
