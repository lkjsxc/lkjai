#include "decoder_cuda_layer_forward.hpp"

#include <stdexcept>

#include "decoder_cuda_weight_sync.hpp"

namespace lkjai {
namespace {

const DeviceTensor& need(const DeviceTensor* tensor, const char* name) {
  if (!tensor) throw std::runtime_error(std::string("missing ") + name);
  return *tensor;
}

void copy_norm(const DeviceTensor& src, DeviceTensor* dst,
               cudaStream_t stream) {
  decoder_cuda_copy_f32_device(src.data(), dst->data(),
                               static_cast<int>(src.spec().elements()),
                               stream);
}

void transpose_proj(const DeviceTensor& src, DeviceTensor* dst, int in,
                    int out, cudaStream_t stream) {
  decoder_cuda_transpose_bf16_device(src.data(), dst->data(), in, out, stream);
}

}  // namespace

void DecoderCudaLayerForward::refresh_from_device(
    const DecoderCudaLayerDeviceWeights& weights) {
  auto stream = ctx_->stream();
  copy_norm(need(weights.attn_norm, "attn_norm"), &attn_w_, stream);
  copy_norm(need(weights.mlp_norm, "mlp_norm"), &mlp_w_, stream);
  transpose_proj(need(weights.q_proj, "q_proj"), &wq_, cfg_.hidden_size,
                 cfg_.hidden_size, stream);
  transpose_proj(need(weights.k_proj, "k_proj"), &wk_, cfg_.hidden_size,
                 kv_width_, stream);
  transpose_proj(need(weights.v_proj, "v_proj"), &wv_, cfg_.hidden_size,
                 kv_width_, stream);
  transpose_proj(need(weights.o_proj, "o_proj"), &wo_, cfg_.hidden_size,
                 cfg_.hidden_size, stream);
  transpose_proj(need(weights.gate_proj, "gate_proj"), &wg_,
                 cfg_.hidden_size, cfg_.ffn_size, stream);
  transpose_proj(need(weights.up_proj, "up_proj"), &wu_, cfg_.hidden_size,
                 cfg_.ffn_size, stream);
  transpose_proj(need(weights.down_proj, "down_proj"), &wd_, cfg_.ffn_size,
                 cfg_.hidden_size, stream);
}

}  // namespace lkjai
