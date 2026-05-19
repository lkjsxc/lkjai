#include "decoder_cuda_state.hpp"

#include <stdexcept>

namespace lkjai {
namespace {

DeviceTensor* layer_tensor(DecoderCudaLayerTape* layer,
                           const std::string& name) {
  if (name == "attn_norm_input") return &layer->attn_norm_input;
  if (name == "attn_norm") return &layer->attn_norm;
  if (name == "q_rope") return &layer->q_rope;
  if (name == "k_rope") return &layer->k_rope;
  if (name == "v") return &layer->v;
  if (name == "attention_state") return &layer->attention_state;
  if (name == "o_proj") return &layer->o_proj;
  if (name == "attention_residual") return &layer->attention_residual;
  if (name == "mlp_norm_input") return &layer->mlp_norm_input;
  if (name == "mlp_norm") return &layer->mlp_norm;
  if (name == "gate") return &layer->gate;
  if (name == "up") return &layer->up;
  if (name == "swiglu") return &layer->swiglu;
  if (name == "down") return &layer->down;
  if (name == "block_residual") return &layer->block_residual;
  throw std::runtime_error("unknown decoder CUDA layer tape tensor: " + name);
}

}  // namespace

std::vector<float> DecoderCudaState::debug_last_layer_tape(
    int layer, const std::string& name) {
  if (layer < 0 || layer >= static_cast<int>(tape_.layers.size())) {
    throw std::runtime_error("decoder CUDA layer tape index out of range");
  }
  return layer_tensor(&tape_.layers[static_cast<size_t>(layer)], name)
      ->copy_to_host_f32(ctx_.stream());
}

size_t DecoderCudaState::debug_last_layer_tape_elements(
    int layer, const std::string& name) {
  if (layer < 0 || layer >= static_cast<int>(tape_.layers.size())) {
    throw std::runtime_error("decoder CUDA layer tape index out of range");
  }
  return layer_tensor(&tape_.layers[static_cast<size_t>(layer)], name)
      ->spec()
      .elements();
}

}  // namespace lkjai
