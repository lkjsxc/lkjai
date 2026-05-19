#include "decoder_cuda_state.hpp"

namespace lkjai {

void DecoderCudaState::sync_registry_grads_from_host() {
  for (auto& t : registry_) {
    t.grad.copy_from_host_f32(t.param->g, ctx_.stream());
  }
}

void DecoderCudaState::optimizer_step(float lr, int step) {
  optimizer_step_d2h_bytes_ = 0;
  for (auto& t : registry_) {
    dense_launch_adamw(static_cast<float*>(t.weight.data()),
                       static_cast<float*>(t.moment_m.data()),
                       static_cast<float*>(t.moment_v.data()),
                       static_cast<float*>(t.grad.data()), t.shadow.data(),
                       static_cast<int>(t.weight.spec().elements()), lr, step,
                       ctx_.stream());
  }
  refresh_layer_forwards_from_registry();
  require_cuda(cudaStreamSynchronize(ctx_.stream()),
               "decoder full-state adamw device sync");
}

}  // namespace lkjai
