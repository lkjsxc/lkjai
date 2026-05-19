#include "decoder_cuda_state.hpp"

#include <cstddef>
#include <string>

#include "decoder_cuda_layer_forward.hpp"

namespace lkjai {

void DecoderCudaState::refresh_layer_forwards_from_registry() {
  for (size_t i = 0; i < layer_forwards_.size(); ++i) {
    auto p = "layers." + std::to_string(i) + ".";
    DecoderCudaLayerDeviceWeights weights;
    weights.attn_norm = &find_registry_tensor(p + "attn_norm")->weight;
    weights.q_proj = &find_registry_tensor(p + "q_proj")->shadow;
    weights.k_proj = &find_registry_tensor(p + "k_proj")->shadow;
    weights.v_proj = &find_registry_tensor(p + "v_proj")->shadow;
    weights.o_proj = &find_registry_tensor(p + "o_proj")->shadow;
    weights.mlp_norm = &find_registry_tensor(p + "mlp_norm")->weight;
    weights.gate_proj = &find_registry_tensor(p + "gate_proj")->shadow;
    weights.up_proj = &find_registry_tensor(p + "up_proj")->shadow;
    weights.down_proj = &find_registry_tensor(p + "down_proj")->shadow;
    layer_forwards_[i].refresh_from_device(weights);
  }
}

}  // namespace lkjai
