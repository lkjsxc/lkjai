#include "decoder_cuda_state.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

#include "decoder_cuda_block_internal.hpp"
#include "decoder_cuda_grad_kernels.hpp"
#include "decoder_cuda_norm.hpp"
#include "dense_cuda_internal.hpp"

namespace lkjai {
namespace {

constexpr size_t kWorkspaceBytes = 4 * 1024 * 1024;

void f32_to_bf16(const DeviceTensor& src, DeviceTensor* dst, int elements,
                 cudaStream_t stream) {
  decoder_cuda_f32_to_bf16(static_cast<const float*>(src.data()), dst->data(),
                           elements, stream);
}

void add_f32(const DeviceTensor& src, DeviceTensor* dst, int elements,
             cudaStream_t stream) {
  decoder_cuda_add_f32(static_cast<const float*>(src.data()),
                       static_cast<float*>(dst->data()), elements, stream);
}

void add_bf16_to_f32(const DeviceTensor& src, DeviceTensor* dst, int elements,
                     cudaStream_t stream) {
  decoder_cuda_add_bf16_to_f32(src.data(), static_cast<float*>(dst->data()),
                               elements, stream);
}

void copy_tensor(const DeviceTensor& src, DeviceTensor* dst, int elements,
                 size_t element_bytes, cudaStream_t stream,
                 const char* label) {
  require_cuda(cudaMemcpyAsync(dst->data(), src.data(),
                               static_cast<size_t>(elements) * element_bytes,
                               cudaMemcpyDeviceToDevice, stream),
               label);
}

}  // namespace

DeviceTensor* DecoderCudaState::run_device_layer_backward(
    int layer_index, int batch_size, int sequence_len, DeviceTensor* upstream) {
  const auto& cfg = state_.cfg;
  int rows = batch_size * sequence_len;
  int hidden_elems = rows * cfg.hidden_size;
  int kv_width = cfg.kv_heads * cfg.head_dim;
  int ffn_elems = rows * cfg.ffn_size;
  auto& l = tape_.layers[static_cast<size_t>(layer_index)];
  const auto& f = layer_forwards_[static_cast<size_t>(layer_index)];
  auto p = "layers." + std::to_string(layer_index) + ".";
  void* ws = workspace_.allocate(kWorkspaceBytes);

  f32_to_bf16(*upstream, &l.grad_block_residual_bf16, hidden_elems,
              ctx_.stream());
  copy_tensor(l.grad_block_residual_bf16, &l.grad_down_bf16, hidden_elems,
              sizeof(uint16_t), ctx_.stream(), "decoder down grad copy");
  decoder_cuda_project_backward_param_layout_bf16(
      ctx_.cublaslt(), ctx_.stream(), l.swiglu.data(), f.wd().data(),
      l.grad_down_bf16.data(), l.grad_swiglu_f32.data(),
      find_registry_tensor(p + "down_proj")->grad.data(), rows, cfg.ffn_size,
      cfg.hidden_size, ws, kWorkspaceBytes, 1.0f);
  f32_to_bf16(l.grad_swiglu_f32, &l.grad_swiglu_bf16, ffn_elems,
              ctx_.stream());
  decoder_launch_swiglu_backward_bf16(
      l.gate.data(), l.up.data(), l.grad_swiglu_bf16.data(),
      l.grad_gate_bf16.data(), l.grad_up_bf16.data(), ffn_elems,
      ctx_.stream());
  decoder_cuda_project_backward_param_layout_bf16(
      ctx_.cublaslt(), ctx_.stream(), l.mlp_norm.data(), f.wg().data(),
      l.grad_gate_bf16.data(), l.grad_mlp_norm_gate_f32.data(),
      find_registry_tensor(p + "gate_proj")->grad.data(), rows,
      cfg.hidden_size, cfg.ffn_size, ws, kWorkspaceBytes, 1.0f);
  decoder_cuda_project_backward_param_layout_bf16(
      ctx_.cublaslt(), ctx_.stream(), l.mlp_norm.data(), f.wu().data(),
      l.grad_up_bf16.data(), l.grad_mlp_norm_up_f32.data(),
      find_registry_tensor(p + "up_proj")->grad.data(), rows, cfg.hidden_size,
      cfg.ffn_size, ws, kWorkspaceBytes, 1.0f);
  copy_tensor(l.grad_mlp_norm_gate_f32, &l.grad_mlp_norm_f32, hidden_elems,
              sizeof(float), ctx_.stream(), "decoder mlp norm grad seed");
  add_f32(l.grad_mlp_norm_up_f32, &l.grad_mlp_norm_f32, hidden_elems,
          ctx_.stream());
  decoder_launch_rmsnorm_backward_bf16_f32_dout(
      l.mlp_norm_input.data(),
      static_cast<float*>(find_registry_tensor(p + "mlp_norm")->weight.data()),
      static_cast<float*>(l.grad_mlp_norm_f32.data()),
      static_cast<float*>(l.grad_mlp_norm_input_f32.data()),
      static_cast<float*>(find_registry_tensor(p + "mlp_norm")->grad.data()),
      rows, cfg.hidden_size, cfg.rms_norm_eps, 1.0f, ctx_.stream());
  copy_tensor(l.grad_mlp_norm_input_f32, &l.grad_attention_residual_f32,
              hidden_elems, sizeof(float), ctx_.stream(),
              "decoder attention residual grad seed");
  add_bf16_to_f32(l.grad_block_residual_bf16, &l.grad_attention_residual_f32,
                  hidden_elems, ctx_.stream());

  f32_to_bf16(l.grad_attention_residual_f32,
              &l.grad_attention_residual_bf16, hidden_elems, ctx_.stream());
  copy_tensor(l.grad_attention_residual_bf16, &l.grad_o_proj_bf16,
              hidden_elems, sizeof(uint16_t), ctx_.stream(),
              "decoder o projection grad copy");
  decoder_cuda_project_backward_param_layout_bf16(
      ctx_.cublaslt(), ctx_.stream(), l.attention_state.data(), f.wo().data(),
      l.grad_o_proj_bf16.data(), l.grad_attention_state_f32.data(),
      find_registry_tensor(p + "o_proj")->grad.data(), rows, cfg.hidden_size,
      cfg.hidden_size, ws, kWorkspaceBytes, 1.0f);
  f32_to_bf16(l.grad_attention_state_f32, &l.grad_attention_state_bf16,
              hidden_elems, ctx_.stream());
  decoder_launch_causal_gqa_attention_backward_bf16(
      l.q_rope.data(), l.k_rope.data(), l.v.data(),
      l.grad_attention_state_bf16.data(), l.grad_q_rope_bf16.data(),
      l.grad_k_rope_bf16.data(), l.grad_v_bf16.data(), batch_size,
      sequence_len, cfg.heads, cfg.kv_heads, cfg.head_dim, ctx_.stream());
  decoder_launch_rope_backward_bf16_at(
      l.grad_q_rope_bf16.data(), l.grad_q_pre_rope_bf16.data(), batch_size,
      sequence_len, cfg.heads, cfg.head_dim, 0, cfg.rope_theta, ctx_.stream());
  decoder_launch_rope_backward_bf16_at(
      l.grad_k_rope_bf16.data(), l.grad_k_pre_rope_bf16.data(), batch_size,
      sequence_len, cfg.kv_heads, cfg.head_dim, 0, cfg.rope_theta,
      ctx_.stream());
  decoder_cuda_project_backward_param_layout_bf16(
      ctx_.cublaslt(), ctx_.stream(), l.attn_norm.data(), f.wq().data(),
      l.grad_q_pre_rope_bf16.data(), l.grad_attn_norm_q_f32.data(),
      find_registry_tensor(p + "q_proj")->grad.data(), rows, cfg.hidden_size,
      cfg.hidden_size, ws, kWorkspaceBytes, 1.0f);
  decoder_cuda_project_backward_param_layout_bf16(
      ctx_.cublaslt(), ctx_.stream(), l.attn_norm.data(), f.wk().data(),
      l.grad_k_pre_rope_bf16.data(), l.grad_attn_norm_k_f32.data(),
      find_registry_tensor(p + "k_proj")->grad.data(), rows, cfg.hidden_size,
      kv_width, ws, kWorkspaceBytes, 1.0f);
  decoder_cuda_project_backward_param_layout_bf16(
      ctx_.cublaslt(), ctx_.stream(), l.attn_norm.data(), f.wv().data(),
      l.grad_v_bf16.data(), l.grad_attn_norm_v_f32.data(),
      find_registry_tensor(p + "v_proj")->grad.data(), rows, cfg.hidden_size,
      kv_width, ws, kWorkspaceBytes, 1.0f);
  copy_tensor(l.grad_attn_norm_q_f32, &l.grad_attn_norm_f32, hidden_elems,
              sizeof(float), ctx_.stream(), "decoder attn norm grad seed");
  add_f32(l.grad_attn_norm_k_f32, &l.grad_attn_norm_f32, hidden_elems,
          ctx_.stream());
  add_f32(l.grad_attn_norm_v_f32, &l.grad_attn_norm_f32, hidden_elems,
          ctx_.stream());
  decoder_launch_rmsnorm_backward_bf16_f32_dout(
      l.attn_norm_input.data(),
      static_cast<float*>(find_registry_tensor(p + "attn_norm")->weight.data()),
      static_cast<float*>(l.grad_attn_norm_f32.data()),
      static_cast<float*>(l.grad_attn_norm_input_f32.data()),
      static_cast<float*>(find_registry_tensor(p + "attn_norm")->grad.data()),
      rows, cfg.hidden_size, cfg.rms_norm_eps, 1.0f, ctx_.stream());
  copy_tensor(l.grad_attn_norm_input_f32, &l.grad_layer_input_f32,
              hidden_elems, sizeof(float), ctx_.stream(),
              "decoder layer input grad seed");
  add_bf16_to_f32(l.grad_attention_residual_bf16, &l.grad_layer_input_f32,
                  hidden_elems, ctx_.stream());
  return &l.grad_layer_input_f32;
}

}  // namespace lkjai
