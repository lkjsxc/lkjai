#include "decoder_cuda_state.hpp"

#include <algorithm>
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
  tape_.embeddings = bf16(ctx_.stream(), r, h);
  tape_.layers.resize(static_cast<size_t>(tape_.layer_capacity));
  tape_.final_norm_input = bf16(ctx_.stream(), r, h);
  tape_.final_norm = bf16(ctx_.stream(), r, h);
  tape_.logits_bf16 = bf16(ctx_.stream(), r, v);
  tape_.logits = f32(ctx_.stream(), r, v);
  tape_.grad_logits = f32(ctx_.stream(), r, v);
  tape_.loss = f32(ctx_.stream(), 1, 1);
}

std::vector<float> DecoderCudaState::debug_last_grad_logits() {
  return tape_.grad_logits.copy_to_host_f32(ctx_.stream());
}

}  // namespace lkjai
