#include "decoder_cuda_state.hpp"

namespace lkjai {

void DecoderCudaState::sync_registry_grads_from_host() {
  for (auto& t : registry_) {
    t.grad.copy_from_host_f32(t.param->g, ctx_.stream());
  }
}

void DecoderCudaState::optimizer_step(float lr, int step) {
  for (auto& t : registry_) {
    dense_launch_adamw(static_cast<float*>(t.weight.data()),
                       static_cast<float*>(t.moment_m.data()),
                       static_cast<float*>(t.moment_v.data()),
                       static_cast<float*>(t.grad.data()), t.shadow.data(),
                       static_cast<int>(t.weight.spec().elements()), lr, step,
                       ctx_.stream());
  }
  copy_registry_to_host();
  refresh_layer_forwards();
  require_cuda(cudaStreamSynchronize(ctx_.stream()),
               "decoder full-state adamw sync");
}

}  // namespace lkjai
