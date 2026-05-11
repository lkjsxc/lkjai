#include "decoder_cuda_state.hpp"

namespace lkjai {

void DecoderCudaState::optimizer_step(float lr, int step) {
  dense_cuda_.adamw(lr, step);
  for (auto& t : registry_) {
    t.grad.copy_from_host_f32(t.accumulated_grad, ctx_.stream());
    dense_launch_adamw(static_cast<float*>(t.weight.data()),
                       static_cast<float*>(t.moment_m.data()),
                       static_cast<float*>(t.moment_v.data()),
                       static_cast<const float*>(t.grad.data()),
                       t.shadow.data(), static_cast<int>(t.accumulated_grad.size()),
                       lr, step, ctx_.stream());
  }
  require_cuda(cudaStreamSynchronize(ctx_.stream()), "decoder registry adamw");
  copy_registry_to_host();
}

}  // namespace lkjai
