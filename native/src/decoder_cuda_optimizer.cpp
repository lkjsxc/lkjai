#include "decoder_cuda_state.hpp"

namespace lkjai {

void DecoderCudaState::optimizer_step(float lr, int step) {
  transformer_adamw(&state_, lr, step);
  sync_registry_from_host();
  require_cuda(cudaStreamSynchronize(ctx_.stream()),
               "decoder full-state adamw sync");
}

}  // namespace lkjai
