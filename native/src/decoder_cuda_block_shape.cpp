#include "decoder_cuda_block.hpp"

#include <string>

namespace lkjai {

bool decoder_cuda_block_shape(const TransformerConfig& cfg,
                              DecoderCudaBlockShape* shape,
                              std::string* error) {
  if (cfg.kind != "decoder") {
    *error = "decoder CUDA block requires model_kind=decoder";
    return false;
  }
  if (cfg.dtype != "bf16") {
    *error = "decoder CUDA block requires dtype=bf16";
    return false;
  }
  if (cfg.hidden_size <= 0 || cfg.heads <= 0 || cfg.kv_heads <= 0 ||
      cfg.head_dim <= 0 || cfg.ffn_size <= 0 || cfg.context <= 1 ||
      cfg.layers <= 0 || cfg.vocab_size <= 0) {
    *error = "decoder CUDA block config has invalid non-positive dimensions";
    return false;
  }
  if (cfg.heads * cfg.head_dim != cfg.hidden_size) {
    *error = "decoder CUDA block heads * head_dim must equal hidden_size";
    return false;
  }
  if (cfg.heads % cfg.kv_heads != 0) {
    *error = "decoder CUDA block heads must be divisible by kv_heads";
    return false;
  }
  if (cfg.head_dim % 2 != 0) {
    *error = "decoder CUDA block RoPE requires even head_dim";
    return false;
  }
  if (cfg.ffn_size < cfg.hidden_size) {
    *error = "decoder CUDA block ffn_size must be at least hidden_size";
    return false;
  }
  if (cfg.activation != "swiglu") {
    *error = "decoder CUDA block requires swiglu activation";
    return false;
  }
  if (shape) {
    shape->hidden = cfg.hidden_size;
    shape->heads = cfg.heads;
    shape->kv_heads = cfg.kv_heads;
    shape->head_dim = cfg.head_dim;
    shape->q_width = cfg.hidden_size;
    shape->k_width = cfg.kv_heads * cfg.head_dim;
    shape->v_width = cfg.kv_heads * cfg.head_dim;
    shape->o_width = cfg.hidden_size;
    shape->ffn_width = cfg.ffn_size;
    shape->gqa_group_size = cfg.heads / cfg.kv_heads;
  }
  return true;
}

}  // namespace lkjai
