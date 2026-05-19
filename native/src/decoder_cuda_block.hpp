#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "transformer_state.hpp"

namespace lkjai {

struct DecoderCudaBlockShape {
  int hidden = 0;
  int heads = 0;
  int kv_heads = 0;
  int head_dim = 0;
  int q_width = 0;
  int k_width = 0;
  int v_width = 0;
  int o_width = 0;
  int ffn_width = 0;
  int gqa_group_size = 0;
  const char* row_layout = "row_major_bxs_flattened";
};

struct DecoderCudaForwardSubstrateReport {
  DecoderCudaBlockShape shape;
  int probe_batch = 0;
  int probe_seq = 0;
  bool rmsnorm_checked = false;
  bool rope_checked = false;
  bool qkv_projection_checked = false;
  bool attention_checked = false;
  bool o_projection_checked = false;
  bool attention_residual_checked = false;
  bool mlp_norm_checked = false;
  bool swiglu_checked = false;
  bool down_projection_checked = false;
  bool block_residual_checked = false;
  bool outputs_finite = false;
  std::vector<float> output_hidden;
  int output_rows = 0;
  int output_hidden_size = 0;
  uint64_t projection_workspace_bytes = 0;
};

struct DecoderCudaFullForwardReport {
  int layers = 0;
  int batch = 0;
  int sequence = 0;
  bool layers_checked = false;
  bool final_norm_checked = false;
  bool logits_checked = false;
  bool hidden_close = false;
  bool logits_close = false;
  bool outputs_finite = false;
  double hidden_max_abs = 0.0;
  double hidden_mean_abs = 0.0;
  double logits_max_abs = 0.0;
  double logits_mean_abs = 0.0;
  uint64_t workspace_bytes = 0;
};

bool decoder_cuda_block_shape(const TransformerConfig& cfg,
                              DecoderCudaBlockShape* shape,
                              std::string* error);

bool decoder_cuda_forward_substrate_probe(
    const TransformerConfig& cfg, DecoderCudaForwardSubstrateReport* report,
    std::string* error);
bool decoder_cuda_full_forward_probe(const TransformerState& state,
                                     const PackedBatch& batch,
                                     DecoderCudaFullForwardReport* report,
                                     std::string* error);

void decoder_launch_rope_bf16(void* tensor_bf16, int batch, int seq, int heads,
                              int head_dim, float theta,
                              cudaStream_t stream);
void decoder_launch_rope_bf16_at(void* tensor_bf16, int batch, int seq,
                                 int heads, int head_dim, int position_offset,
                                 float theta, cudaStream_t stream);
void decoder_launch_rope_backward_bf16_at(const void* d_output_bf16,
                                          void* d_input_bf16, int batch,
                                          int seq, int heads, int head_dim,
                                          int position_offset, float theta,
                                          cudaStream_t stream);

void decoder_launch_swiglu_bf16(const void* gate_bf16, const void* up_bf16,
                                void* out_bf16, int elements,
                                cudaStream_t stream);
void decoder_launch_swiglu_backward_bf16(
    const void* gate_bf16, const void* up_bf16, const void* d_out_bf16,
    void* d_gate_bf16, void* d_up_bf16, int elements, cudaStream_t stream);
void decoder_launch_causal_gqa_attention_bf16(
    const void* q_bf16, const void* k_bf16, const void* v_bf16, void* out_bf16,
    int batch, int seq, int heads, int kv_heads, int head_dim,
    cudaStream_t stream);
void decoder_launch_causal_gqa_attention_backward_bf16(
    const void* q_bf16, const void* k_bf16, const void* v_bf16,
    const void* d_out_bf16, void* d_q_bf16, void* d_k_bf16, void* d_v_bf16,
    int batch, int seq, int heads, int kv_heads, int head_dim,
    cudaStream_t stream);
void decoder_launch_cached_gqa_attention_bf16(
    const void* q_bf16, const void* key_cache_bf16,
    const void* value_cache_bf16, void* out_bf16, int layer,
    int start_batch, int position, int cache_batch, int context, int batch,
    int heads, int kv_heads, int head_dim, cudaStream_t stream);

}  // namespace lkjai
