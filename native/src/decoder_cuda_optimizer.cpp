#include "decoder_cuda_state.hpp"

namespace lkjai {

void DecoderCudaState::optimizer_step(float lr, int step) {
  dense_cuda_.adamw(lr, step);
}

}  // namespace lkjai
