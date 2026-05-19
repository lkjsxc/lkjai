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
  DeviceTensor grad_block_residual_bf16;
  DeviceTensor grad_down_f32;
  DeviceTensor grad_down_bf16;
  DeviceTensor grad_swiglu_f32;
  DeviceTensor grad_swiglu_bf16;
  DeviceTensor grad_gate_bf16;
  DeviceTensor grad_up_bf16;
  DeviceTensor grad_mlp_norm_gate_f32;
  DeviceTensor grad_mlp_norm_up_f32;
  DeviceTensor grad_mlp_norm_f32;
  DeviceTensor grad_mlp_norm_input_f32;
  DeviceTensor grad_attention_residual_f32;
  DeviceTensor grad_attention_residual_bf16;
  DeviceTensor grad_o_proj_f32;
  DeviceTensor grad_o_proj_bf16;
  DeviceTensor grad_attention_state_f32;
  DeviceTensor grad_attention_state_bf16;
  DeviceTensor grad_q_rope_bf16;
  DeviceTensor grad_k_rope_bf16;
  DeviceTensor grad_v_bf16;
  DeviceTensor grad_q_pre_rope_bf16;
  DeviceTensor grad_k_pre_rope_bf16;
  DeviceTensor grad_attn_norm_q_f32;
  DeviceTensor grad_attn_norm_k_f32;
  DeviceTensor grad_attn_norm_v_f32;
  DeviceTensor grad_attn_norm_f32;
  DeviceTensor grad_attn_norm_input_f32;
  DeviceTensor grad_layer_input_f32;
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
  DeviceTensor grad_embeddings_f32;
  DeviceTensor grad_layer_upstream_bf16;
  DeviceTensor lm_head_f32;
  DeviceTensor logits_bf16;
  DeviceTensor logits;
  DeviceTensor grad_logits;
  DeviceTensor loss;
};

}  // namespace lkjai
