#include "decoder_cuda_state.hpp"

namespace lkjai {

double DecoderCudaState::forward_backward(
    const PackedBatch& batch, std::vector<float>* logits, double* h2d_seconds,
    double* forward_seconds, double* backward_seconds, float grad_scale,
    bool reset_grads) {
  return dense_cuda_.forward_backward(batch, logits, h2d_seconds,
                                      forward_seconds, backward_seconds,
                                      grad_scale, reset_grads);
}

}  // namespace lkjai
