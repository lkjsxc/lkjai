#include "decoder_cuda_state.hpp"

#include <cstddef>
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

void f32_to_bf16(const DeviceTensor& src, DeviceTensor* dst, int elements,
                 cudaStream_t stream) {
  decoder_cuda_f32_to_bf16(static_cast<const float*>(src.data()), dst->data(),
                           elements, stream);
}

}  // namespace

DecoderCudaState::RegistryTensor* DecoderCudaState::find_registry_tensor(
    const std::string& name) {
  for (auto& t : registry_) {
    if (t.name == name) return &t;
  }
  throw std::runtime_error("missing decoder CUDA registry tensor: " + name);
}

void DecoderCudaState::run_device_backward(float loss, int batch_size,
                                           int sequence_len, int capture_row,
                                           float grad_scale,
                                           bool reset_grads) {
  (void)loss;
  (void)capture_row;
  (void)grad_scale;
  if (reset_grads) {
    for (auto& t : registry_) {
      decoder_cuda_zero_f32(static_cast<float*>(t.grad.data()),
                            elements(t.grad), ctx_.stream());
    }
  }
  const auto& cfg = state_.cfg;
  int rows = batch_size * sequence_len;
  auto* head = cfg.tie_embeddings ? find_registry_tensor("tok_embeddings")
                                  : find_registry_tensor("lm_head");
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
      static_cast<float*>(find_registry_tensor("final_norm")->weight.data()),
      static_cast<float*>(tape_.grad_final_norm.data()),
      static_cast<float*>(tape_.grad_final_norm_input.data()),
      static_cast<float*>(find_registry_tensor("final_norm")->grad.data()), rows,
      cfg.hidden_size, cfg.rms_norm_eps, 1.0f, ctx_.stream());
  DeviceTensor* upstream = &tape_.grad_final_norm_input;
  int hidden_elems = rows * cfg.hidden_size;
  for (int layer_index = cfg.layers - 1; layer_index >= 0; --layer_index) {
    upstream =
        run_device_layer_backward(layer_index, batch_size, sequence_len,
                                  upstream);
  }
  f32_to_bf16(*upstream, &tape_.grad_layer_upstream_bf16, hidden_elems,
              ctx_.stream());
  decoder_cuda_add_input_embedding_grad(
      tape_.device_tokens, tape_.grad_layer_upstream_bf16.data(),
      static_cast<float*>(find_registry_tensor("tok_embeddings")->grad.data()),
      rows, cfg.vocab_size, cfg.hidden_size, ctx_.stream());
}

}  // namespace lkjai
