#include "decoder_cuda_state.hpp"

#include <chrono>

namespace lkjai {
namespace {

double since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

}  // namespace

double DecoderCudaState::forward_backward(
    const PackedBatch& batch, std::vector<float>* logits, double* h2d_seconds,
    double* forward_seconds, double* backward_seconds, float grad_scale,
    bool reset_grads) {
  if (h2d_seconds) *h2d_seconds += 0.0;
  auto before_backward = state_;
  auto phase = std::chrono::steady_clock::now();
  auto fwd = transformer_forward(batch, state_);
  if (forward_seconds) *forward_seconds += since(phase);
  if (logits) *logits = fwd.next_logits;
  phase = std::chrono::steady_clock::now();
  transformer_backward(batch, fwd, &state_);
  scale_and_accumulate_grads(before_backward, grad_scale, reset_grads);
  if (backward_seconds) *backward_seconds += since(phase);
  return fwd.loss;
}

}  // namespace lkjai
