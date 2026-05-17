#pragma once

#include <cstdint>
#include <vector>

#include "runtime_device.hpp"

namespace lkjai {

struct DecoderCudaLayerTape {
  DeviceTensor attn_norm_input;
  DeviceTensor attn_norm;
  DeviceTensor q_rope;
  DeviceTensor k_rope;
  DeviceTensor v;
  DeviceTensor attention_state;
  DeviceTensor o_proj;
  DeviceTensor attention_residual;
  DeviceTensor mlp_norm_input;
  DeviceTensor mlp_norm;
  DeviceTensor gate;
  DeviceTensor up;
  DeviceTensor swiglu;
  DeviceTensor down;
  DeviceTensor block_residual;
};

struct DecoderCudaTape {
  ~DecoderCudaTape();
  DecoderCudaTape() = default;
  DecoderCudaTape(const DecoderCudaTape&) = delete;
  DecoderCudaTape& operator=(const DecoderCudaTape&) = delete;

  uint16_t* device_tokens = nullptr;
  uint8_t* device_loss_mask = nullptr;
  float* host_loss = nullptr;
  float* host_logits = nullptr;
  size_t token_capacity = 0;
  size_t mask_capacity = 0;
  size_t host_logits_capacity = 0;
  int rows_capacity = 0;
  int vocab_capacity = 0;
  int hidden_capacity = 0;
  int layer_capacity = 0;
  DeviceTensor embeddings;
  std::vector<DecoderCudaLayerTape> layers;
  DeviceTensor final_norm_input;
  DeviceTensor final_norm;
  DeviceTensor grad_final_norm;
  DeviceTensor grad_final_norm_input;
  DeviceTensor lm_head_f32;
  DeviceTensor logits_bf16;
  DeviceTensor logits;
  DeviceTensor grad_logits;
  DeviceTensor loss;
};

}  // namespace lkjai
