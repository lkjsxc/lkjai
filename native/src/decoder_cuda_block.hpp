#pragma once

#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "transformer_train.hpp"

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
  bool o_projection_checked = false;
  bool swiglu_checked = false;
  bool outputs_finite = false;
  uint64_t projection_workspace_bytes = 0;
};

bool decoder_cuda_block_shape(const TransformerConfig& cfg,
                              DecoderCudaBlockShape* shape,
                              std::string* error);

bool decoder_cuda_forward_substrate_probe(
    const TransformerConfig& cfg, DecoderCudaForwardSubstrateReport* report,
    std::string* error);

void decoder_launch_rope_bf16(void* tensor_bf16, int batch, int seq, int heads,
                              int head_dim, float theta,
                              cudaStream_t stream);

void decoder_launch_swiglu_bf16(const void* gate_bf16, const void* up_bf16,
                                void* out_bf16, int elements,
                                cudaStream_t stream);
void decoder_launch_causal_gqa_attention_bf16(
    const void* q_bf16, const void* k_bf16, const void* v_bf16, void* out_bf16,
    int batch, int seq, int heads, int kv_heads, int head_dim,
    cudaStream_t stream);

}  // namespace lkjai
