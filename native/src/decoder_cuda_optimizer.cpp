#include "decoder_cuda_state.hpp"

namespace lkjai {

void DecoderCudaState::optimizer_step(float lr, int step) {
  dense_cuda_.adamw(lr, step);
  require_cuda(cudaStreamSynchronize(ctx_.stream()),
               "decoder embedding lm-head adamw");
}

}  // namespace lkjai
