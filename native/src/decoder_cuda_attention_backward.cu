#include "decoder_cuda_block.hpp"

#include <stdexcept>

#include <cuda_bf16.h>

#include "runtime_device.hpp"

namespace lkjai {
namespace {

__device__ float score(const __nv_bfloat16* q, const __nv_bfloat16* k, int b,
                       int t, int s, int h, int kv_h, int seq, int heads,
                       int kv_heads, int dim) {
  size_t qb = ((static_cast<size_t>(b) * seq + t) * heads + h) * dim;
  size_t kb = ((static_cast<size_t>(b) * seq + s) * kv_heads + kv_h) * dim;
  float out = 0.0f;
  for (int d = 0; d < dim; ++d)
    out += __bfloat162float(q[qb + d]) * __bfloat162float(k[kb + d]);
  return out * rsqrtf(static_cast<float>(dim));
}

__device__ void stats(const __nv_bfloat16* q, const __nv_bfloat16* k, int b,
                      int t, int h, int kv_h, int seq, int heads,
                      int kv_heads, int dim, float* max_s, float* denom) {
  *max_s = -INFINITY;
  for (int s = 0; s <= t; ++s)
    *max_s = fmaxf(*max_s,
                   score(q, k, b, t, s, h, kv_h, seq, heads, kv_heads, dim));
  *denom = 0.0f;
  for (int s = 0; s <= t; ++s)
    *denom += expf(score(q, k, b, t, s, h, kv_h, seq, heads, kv_heads, dim) -
                   *max_s);
}

__device__ float prob(const __nv_bfloat16* q, const __nv_bfloat16* k, int b,
                      int t, int s, int h, int kv_h, int seq, int heads,
                      int kv_heads, int dim, float max_s, float denom) {
  return expf(score(q, k, b, t, s, h, kv_h, seq, heads, kv_heads, dim) -
              max_s) /
         denom;
}

__device__ float dscore(const __nv_bfloat16* q, const __nv_bfloat16* k,
                        const __nv_bfloat16* v, const __nv_bfloat16* d_out,
                        int b, int t, int s, int h, int kv_h, int seq,
                        int heads, int kv_heads, int dim) {
  float max_s = 0.0f, denom = 0.0f;
  stats(q, k, b, t, h, kv_h, seq, heads, kv_heads, dim, &max_s, &denom);
  float dot_s = 0.0f, expected = 0.0f;
  size_t db = ((static_cast<size_t>(b) * seq + t) * heads + h) * dim;
  for (int u = 0; u <= t; ++u) {
    size_t vb = ((static_cast<size_t>(b) * seq + u) * kv_heads + kv_h) * dim;
    float dot = 0.0f;
    for (int d = 0; d < dim; ++d)
      dot += __bfloat162float(d_out[db + d]) * __bfloat162float(v[vb + d]);
    float p = prob(q, k, b, t, u, h, kv_h, seq, heads, kv_heads, dim, max_s,
                   denom);
    expected += p * dot;
    if (u == s) dot_s = dot;
  }
  float p_s = prob(q, k, b, t, s, h, kv_h, seq, heads, kv_heads, dim, max_s,
                   denom);
  return p_s * (dot_s - expected);
}

__global__ void backward_q(const __nv_bfloat16* q, const __nv_bfloat16* k,
                           const __nv_bfloat16* v,
                           const __nv_bfloat16* d_out, __nv_bfloat16* d_q,
                           int total, int seq, int heads, int kv_heads,
                           int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  int d = i % dim, h = (i / dim) % heads;
  int t = (i / (dim * heads)) % seq;
  int b = i / (dim * heads * seq);
  int kv_h = h / (heads / kv_heads);
  float scale = rsqrtf(static_cast<float>(dim));
  float sum = 0.0f;
  for (int s = 0; s <= t; ++s) {
    size_t kb = ((static_cast<size_t>(b) * seq + s) * kv_heads + kv_h) * dim;
    sum += dscore(q, k, v, d_out, b, t, s, h, kv_h, seq, heads, kv_heads,
                  dim) *
           __bfloat162float(k[kb + d]) * scale;
  }
  d_q[i] = __float2bfloat16(sum);
}

__global__ void backward_kv(const __nv_bfloat16* q, const __nv_bfloat16* k,
                            const __nv_bfloat16* v,
                            const __nv_bfloat16* d_out, __nv_bfloat16* d_k,
                            __nv_bfloat16* d_v, int total, int seq, int heads,
                            int kv_heads, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  int d = i % dim, kv_h = (i / dim) % kv_heads;
  int s = (i / (dim * kv_heads)) % seq;
  int b = i / (dim * kv_heads * seq);
  float scale = rsqrtf(static_cast<float>(dim));
  float dk = 0.0f, dv = 0.0f;
  for (int gh = 0; gh < heads / kv_heads; ++gh) {
    int h = kv_h * (heads / kv_heads) + gh;
    for (int t = s; t < seq; ++t) {
      float max_s = 0.0f, denom = 0.0f;
      stats(q, k, b, t, h, kv_h, seq, heads, kv_heads, dim, &max_s, &denom);
      float p = prob(q, k, b, t, s, h, kv_h, seq, heads, kv_heads, dim,
                     max_s, denom);
      size_t qb = ((static_cast<size_t>(b) * seq + t) * heads + h) * dim;
      float ds = dscore(q, k, v, d_out, b, t, s, h, kv_h, seq, heads,
                        kv_heads, dim);
      dk += ds * __bfloat162float(q[qb + d]) * scale;
      dv += p * __bfloat162float(d_out[qb + d]);
    }
  }
  d_k[i] = __float2bfloat16(dk);
  d_v[i] = __float2bfloat16(dv);
}

}  // namespace

void decoder_launch_causal_gqa_attention_backward_bf16(
    const void* q_bf16, const void* k_bf16, const void* v_bf16,
    const void* d_out_bf16, void* d_q_bf16, void* d_k_bf16, void* d_v_bf16,
    int batch, int seq, int heads, int kv_heads, int dim,
    cudaStream_t stream) {
  if (batch <= 0 || seq <= 0 || heads <= 0 || kv_heads <= 0 || dim <= 0)
    return;
  if (heads % kv_heads != 0)
    throw std::runtime_error("decoder GQA attention backward shape mismatch");
  int q_total = batch * seq * heads * dim;
  int kv_total = batch * seq * kv_heads * dim;
  backward_q<<<(q_total + 127) / 128, 128, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(q_bf16),
      static_cast<const __nv_bfloat16*>(k_bf16),
      static_cast<const __nv_bfloat16*>(v_bf16),
      static_cast<const __nv_bfloat16*>(d_out_bf16),
      static_cast<__nv_bfloat16*>(d_q_bf16), q_total, seq, heads, kv_heads,
      dim);
  require_cuda(cudaGetLastError(), "decoder GQA backward Q");
  backward_kv<<<(kv_total + 127) / 128, 128, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(q_bf16),
      static_cast<const __nv_bfloat16*>(k_bf16),
      static_cast<const __nv_bfloat16*>(v_bf16),
      static_cast<const __nv_bfloat16*>(d_out_bf16),
      static_cast<__nv_bfloat16*>(d_k_bf16),
      static_cast<__nv_bfloat16*>(d_v_bf16), kv_total, seq, heads, kv_heads,
      dim);
  require_cuda(cudaGetLastError(), "decoder GQA backward KV");
}

}  // namespace lkjai
