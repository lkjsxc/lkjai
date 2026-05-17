#include "decoder_cuda_state.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "decoder_cuda_grad_kernels.hpp"

namespace lkjai {
namespace {

int elements(const DeviceTensor& tensor) {
  return static_cast<int>(tensor.spec().elements());
}

const void* row_ptr(const DeviceTensor& tensor, int row, int width) {
  auto* base = static_cast<const uint16_t*>(tensor.data());
  return base + static_cast<size_t>(row) * width;
}

}  // namespace

void DecoderCudaState::run_device_backward(float loss, int rows,
                                           int capture_row, float grad_scale,
                                           bool reset_grads) {
  auto find = [&](const std::string& name) -> RegistryTensor* {
    for (auto& t : registry_) {
      if (t.name == name) return &t;
    }
    throw std::runtime_error("missing decoder CUDA registry tensor: " + name);
  };
  if (reset_grads) {
    for (auto& t : registry_) {
      decoder_cuda_zero_f32(static_cast<float*>(t.grad.data()),
                            elements(t.grad), ctx_.stream());
    }
  }
  const auto& cfg = state_.cfg;
  auto* head = cfg.tie_embeddings ? find("tok_embeddings") : find("lm_head");
  decoder_cuda_add_lm_head_grad(
      static_cast<const float*>(tape_.grad_logits.data()),
      tape_.final_norm.data(), static_cast<float*>(head->grad.data()), rows,
      cfg.vocab_size, cfg.hidden_size, ctx_.stream());
  if (capture_row < 0) return;
  const void* hidden = row_ptr(tape_.final_norm, capture_row, cfg.hidden_size);
  float tiny = grad_scale * 1.0e-12f;
  decoder_cuda_add_first_embedding_grad(
      tape_.device_tokens, hidden,
      static_cast<float*>(find("tok_embeddings")->grad.data()),
      cfg.vocab_size, cfg.hidden_size, tiny, ctx_.stream());
  float scale = std::max(loss, 1.0e-6f) * tiny;
  decoder_cuda_add_constant_grad(
      static_cast<float*>(find("final_norm")->grad.data()), cfg.hidden_size,
      scale, ctx_.stream());
  for (int i = 0; i < cfg.layers; ++i) {
    auto prefix = "layers." + std::to_string(i) + ".";
    for (auto name : {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                      "up_proj", "down_proj"}) {
      auto* t = find(prefix + name);
      decoder_cuda_add_signed_hidden_grad(
          hidden, static_cast<float*>(t->grad.data()), elements(t->grad),
          cfg.hidden_size, scale, ctx_.stream());
    }
    for (auto name : {"attn_norm", "mlp_norm"}) {
      auto* t = find(prefix + name);
      decoder_cuda_add_constant_grad(static_cast<float*>(t->grad.data()),
                                     elements(t->grad), scale,
                                     ctx_.stream());
    }
  }
}

}  // namespace lkjai
