#include "decoder_cuda_state.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

namespace lkjai {
namespace {

constexpr size_t kWorkspaceBytes = 4 * 1024 * 1024;

DeviceTensor bf16(cudaStream_t stream, int rows, int cols) {
  return DeviceTensor({DeviceDType::bf16, {rows, cols}}, stream);
}

DeviceTensor f32(cudaStream_t stream, int rows, int cols) {
  return DeviceTensor({DeviceDType::f32, {rows, cols}}, stream);
}

}  // namespace

DecoderCudaTape::~DecoderCudaTape() {
  if (device_tokens) cudaFree(device_tokens);
  if (device_loss_mask) cudaFree(device_loss_mask);
  if (host_loss) cudaFreeHost(host_loss);
  if (host_logits) cudaFreeHost(host_logits);
}

void DecoderCudaState::refresh_layer_forwards() {
  layer_forwards_.clear();
  layer_forwards_.reserve(state_.layers.size());
  for (const auto& layer : state_.layers) {
    layer_forwards_.emplace_back(state_.cfg, layer, &ctx_, &workspace_,
                                 kWorkspaceBytes);
  }
}

void DecoderCudaState::ensure_tape_capacity(int rows, int vocab, int hidden,
                                            int layers) {
  size_t items = static_cast<size_t>(rows);
  if (items > tape_.token_capacity) {
    if (tape_.device_tokens) cudaFree(tape_.device_tokens);
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&tape_.device_tokens),
                            items * sizeof(uint16_t)),
                 "decoder train device tokens");
    tape_.token_capacity = items;
  }
  if (items > tape_.mask_capacity) {
    if (tape_.device_loss_mask) cudaFree(tape_.device_loss_mask);
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&tape_.device_loss_mask),
                            items),
                 "decoder train device loss mask");
    tape_.mask_capacity = items;
  }
  if (!tape_.host_loss) {
    require_cuda(cudaMallocHost(reinterpret_cast<void**>(&tape_.host_loss),
                                sizeof(float)),
                 "decoder train host loss");
  }
  if (static_cast<size_t>(vocab) > tape_.host_logits_capacity) {
    if (tape_.host_logits) cudaFreeHost(tape_.host_logits);
    require_cuda(cudaMallocHost(reinterpret_cast<void**>(&tape_.host_logits),
                                static_cast<size_t>(vocab) * sizeof(float)),
                 "decoder train host logits");
    tape_.host_logits_capacity = static_cast<size_t>(vocab);
  }
  bool grow = rows > tape_.rows_capacity || vocab > tape_.vocab_capacity ||
              hidden > tape_.hidden_capacity ||
              layers > tape_.layer_capacity;
  if (!grow) return;
  tape_.rows_capacity = std::max(rows, tape_.rows_capacity);
  tape_.vocab_capacity = std::max(vocab, tape_.vocab_capacity);
  tape_.hidden_capacity = std::max(hidden, tape_.hidden_capacity);
  tape_.layer_capacity = std::max(layers, tape_.layer_capacity);
  int r = tape_.rows_capacity;
  int v = tape_.vocab_capacity;
  int h = tape_.hidden_capacity;
  int kv = state_.cfg.kv_heads * state_.cfg.head_dim;
  int ffn = state_.cfg.ffn_size;
  tape_.embeddings = bf16(ctx_.stream(), r, h);
  tape_.layers.resize(static_cast<size_t>(tape_.layer_capacity));
  for (auto& layer : tape_.layers) {
    layer.attn_norm_input = bf16(ctx_.stream(), r, h);
    layer.attn_norm = bf16(ctx_.stream(), r, h);
    layer.q_rope = bf16(ctx_.stream(), r, h);
    layer.k_rope = bf16(ctx_.stream(), r, kv);
    layer.v = bf16(ctx_.stream(), r, kv);
    layer.attention_state = bf16(ctx_.stream(), r, h);
    layer.o_proj = bf16(ctx_.stream(), r, h);
    layer.attention_residual = bf16(ctx_.stream(), r, h);
    layer.mlp_norm_input = bf16(ctx_.stream(), r, h);
    layer.mlp_norm = bf16(ctx_.stream(), r, h);
    layer.gate = bf16(ctx_.stream(), r, ffn);
    layer.up = bf16(ctx_.stream(), r, ffn);
    layer.swiglu = bf16(ctx_.stream(), r, ffn);
    layer.down = bf16(ctx_.stream(), r, h);
    layer.block_residual = bf16(ctx_.stream(), r, h);
  }
  tape_.final_norm_input = bf16(ctx_.stream(), r, h);
  tape_.final_norm = bf16(ctx_.stream(), r, h);
  tape_.grad_final_norm = f32(ctx_.stream(), r, h);
  tape_.grad_final_norm_input = f32(ctx_.stream(), r, h);
  tape_.lm_head_f32 = f32(ctx_.stream(), v, h);
  tape_.logits_bf16 = bf16(ctx_.stream(), r, v);
  tape_.logits = f32(ctx_.stream(), r, v);
  tape_.grad_logits = f32(ctx_.stream(), r, v);
  tape_.loss = f32(ctx_.stream(), 1, 1);
}

std::vector<float> DecoderCudaState::debug_last_grad_logits() {
  return tape_.grad_logits.copy_to_host_f32(ctx_.stream());
}

std::vector<float> DecoderCudaState::debug_last_final_norm_input() {
  return tape_.final_norm_input.copy_to_host_f32(ctx_.stream());
}

std::vector<float> DecoderCudaState::debug_last_final_norm() {
  return tape_.final_norm.copy_to_host_f32(ctx_.stream());
}

std::vector<float> DecoderCudaState::debug_last_grad_final_norm() {
  return tape_.grad_final_norm.copy_to_host_f32(ctx_.stream());
}

std::vector<float> DecoderCudaState::debug_last_grad_final_norm_input() {
  return tape_.grad_final_norm_input.copy_to_host_f32(ctx_.stream());
}

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
