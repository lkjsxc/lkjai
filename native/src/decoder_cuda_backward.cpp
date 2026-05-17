#include "decoder_cuda_state.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>

#include "decoder_cuda_block_internal.hpp"
#include "decoder_cuda_grad_kernels.hpp"
#include "decoder_cuda_norm.hpp"
#include "dense_cuda_internal.hpp"

namespace lkjai {
namespace {

constexpr size_t kWorkspaceBytes = 4 * 1024 * 1024;

int elements(const DeviceTensor& tensor) {
  return static_cast<int>(tensor.spec().elements());
}

const void* row_ptr(const DeviceTensor& tensor, int row, int width) {
  auto* base = static_cast<const uint16_t*>(tensor.data());
  return base + static_cast<size_t>(row) * width;
}

void add_remaining_diagnostic_decoder_grads(
    const TransformerConfig& cfg,
    const std::function<DecoderCudaState::RegistryTensor*(const std::string&)>&
        find,
    const DecoderCudaTape& tape, float loss, int capture_row, float grad_scale,
    cudaStream_t stream) {
  if (capture_row < 0) return;
  const void* hidden = row_ptr(tape.final_norm, capture_row, cfg.hidden_size);
  float tiny = grad_scale * 1.0e-12f;
  decoder_cuda_add_first_embedding_grad(
      tape.device_tokens, hidden,
      static_cast<float*>(find("tok_embeddings")->grad.data()),
      cfg.vocab_size, cfg.hidden_size, tiny, stream);
  float scale = std::max(loss, 1.0e-6f) * tiny;
  for (int i = 0; i < cfg.layers; ++i) {
    auto prefix = "layers." + std::to_string(i) + ".";
    for (auto name : {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                      "up_proj", "down_proj"}) {
      auto* t = find(prefix + name);
      decoder_cuda_add_signed_hidden_grad(
          hidden, static_cast<float*>(t->grad.data()), elements(t->grad),
          cfg.hidden_size, scale, stream);
    }
    for (auto name : {"attn_norm", "mlp_norm"}) {
      auto* t = find(prefix + name);
      decoder_cuda_add_constant_grad(static_cast<float*>(t->grad.data()),
                                     elements(t->grad), scale, stream);
    }
  }
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
  void* ws = workspace_.allocate(kWorkspaceBytes);
  dense_launch_bf16_to_f32(head->shadow.data(),
                           static_cast<float*>(tape_.lm_head_f32.data()),
                           cfg.vocab_size * cfg.hidden_size, ctx_.stream());
  decoder_cuda_lm_head_dhidden_f32(
      ctx_.cublaslt(), ctx_.stream(), tape_.grad_logits.data(),
      tape_.lm_head_f32.data(), tape_.grad_final_norm.data(), rows,
      cfg.vocab_size, cfg.hidden_size, ws, kWorkspaceBytes);
  decoder_launch_rmsnorm_backward_bf16_f32_dout(
      tape_.final_norm_input.data(),
      static_cast<float*>(find("final_norm")->weight.data()),
      static_cast<float*>(tape_.grad_final_norm.data()),
      static_cast<float*>(tape_.grad_final_norm_input.data()),
      static_cast<float*>(find("final_norm")->grad.data()), rows,
      cfg.hidden_size, cfg.rms_norm_eps, 1.0f, ctx_.stream());
  add_remaining_diagnostic_decoder_grads(cfg, find, tape_, loss, capture_row,
                                         grad_scale, ctx_.stream());
}

}  // namespace lkjai
