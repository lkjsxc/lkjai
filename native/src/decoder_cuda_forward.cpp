#include "decoder_cuda_state.hpp"

#include <chrono>

namespace lkjai {
namespace {

double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

}  // namespace

double DecoderCudaState::forward_backward(
    const PackedBatch& batch, std::vector<float>* logits, double* h2d_seconds,
    double* forward_seconds, double* backward_seconds, float grad_scale,
    bool reset_grads) {
  auto started = std::chrono::steady_clock::now();
  double loss = dense_cuda_.forward_backward(batch, logits, h2d_seconds,
                                             forward_seconds, backward_seconds,
                                             grad_scale, reset_grads);
  auto grad_started = std::chrono::steady_clock::now();
  accumulate_decoder_gradients(batch, loss, grad_scale, reset_grads);
  if (backward_seconds) *backward_seconds += seconds_since(grad_started);
  if (forward_seconds && *forward_seconds == 0.0) {
    *forward_seconds += seconds_since(started);
  }
  return loss;
}

}  // namespace lkjai
