#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <vector>

#include "decoder_cuda_block.hpp"
#include "decoder_cuda_block_internal.hpp"
#include "runtime_device.hpp"

namespace {

std::vector<float> cpu_attention(const std::vector<float>& q,
                                 const std::vector<float>& k,
                                 const std::vector<float>& v, int batch,
                                 int seq, int heads, int kv_heads,
                                 int head_dim) {
  std::vector<float> out(q.size());
  for (int b = 0; b < batch; ++b) {
    for (int t = 0; t < seq; ++t) {
      for (int h = 0; h < heads; ++h) {
        int kv_h = h % kv_heads;
        size_t qb = ((size_t(b) * seq + t) * heads + h) * head_dim;
        float max_score = -INFINITY;
        std::vector<float> score(t + 1);
        for (int s = 0; s <= t; ++s) {
          size_t kb = ((size_t(b) * seq + s) * kv_heads + kv_h) * head_dim;
          for (int d = 0; d < head_dim; ++d) score[s] += q[qb + d] * k[kb + d];
          score[s] *= 1.0f / std::sqrt(float(head_dim));
          max_score = std::max(max_score, score[s]);
        }
        float denom = 0.0f;
        for (float& value : score) {
          value = std::exp(value - max_score);
          denom += value;
        }
        for (int d = 0; d < head_dim; ++d) {
          float sum = 0.0f;
          for (int s = 0; s <= t; ++s) {
            size_t vb = ((size_t(b) * seq + s) * kv_heads + kv_h) * head_dim;
            sum += score[s] * v[vb + d];
          }
          out[qb + d] = sum / denom;
        }
      }
    }
  }
  return out;
}

bool close(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    float diff = std::fabs(a[i] - b[i]);
    if (diff > 0.035f + 0.01f * std::fabs(b[i])) {
      std::cerr << "mismatch at " << i << ": " << a[i] << " vs " << b[i]
                << "\n";
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  try {
    lkjai::CudaExecutionContext ctx;
    lkjai::decoder_cuda_projection_plan_cache_reset();
    constexpr int rows = 3;
    constexpr int in = 8;
    constexpr int out = 6;
    std::vector<float> x(rows * in), w(out * in);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(float(i) * 0.13f);
    for (size_t i = 0; i < w.size(); ++i) w[i] = std::cos(float(i) * 0.07f);
    lkjai::DeviceTensor dx({lkjai::DeviceDType::bf16, {rows, in}}, ctx.stream());
    lkjai::DeviceTensor dw({lkjai::DeviceDType::bf16, {out, in}}, ctx.stream());
    lkjai::DeviceTensor dy({lkjai::DeviceDType::bf16, {rows, out}}, ctx.stream());
    dx.copy_from_host_f32(x, ctx.stream());
    dw.copy_from_host_f32(w, ctx.stream());
    lkjai::decoder_cuda_project_bf16(ctx.cublaslt(), ctx.stream(), dx.data(),
                                     dw.data(), dy.data(), rows, in, out,
                                     nullptr, 0);
    lkjai::decoder_cuda_project_bf16(ctx.cublaslt(), ctx.stream(), dx.data(),
                                     dw.data(), dy.data(), rows, in, out,
                                     nullptr, 0);
    if (lkjai::decoder_cuda_projection_plan_cache_size() != 1) {
      std::cerr << "projection plan cache was not reused\n";
      return 1;
    }
    constexpr int batch = 1;
    constexpr int seq = 4;
    constexpr int heads = 4;
    constexpr int kv_heads = 2;
    constexpr int head_dim = 8;
    std::vector<float> q(batch * seq * heads * head_dim);
    std::vector<float> k(batch * seq * kv_heads * head_dim);
    std::vector<float> v(k.size());
    for (size_t i = 0; i < q.size(); ++i) q[i] = std::sin(float(i) * 0.11f);
    for (size_t i = 0; i < k.size(); ++i) {
      k[i] = std::cos(float(i) * 0.09f);
      v[i] = std::sin(float(i) * 0.05f) * 0.7f;
    }
    lkjai::DeviceTensor dq({lkjai::DeviceDType::bf16, {batch, seq, heads, head_dim}}, ctx.stream());
    lkjai::DeviceTensor dk({lkjai::DeviceDType::bf16, {batch, seq, kv_heads, head_dim}}, ctx.stream());
    lkjai::DeviceTensor dv({lkjai::DeviceDType::bf16, {batch, seq, kv_heads, head_dim}}, ctx.stream());
    lkjai::DeviceTensor dout({lkjai::DeviceDType::bf16, {batch, seq, heads, head_dim}}, ctx.stream());
    dq.copy_from_host_f32(q, ctx.stream());
    dk.copy_from_host_f32(k, ctx.stream());
    dv.copy_from_host_f32(v, ctx.stream());
    lkjai::decoder_launch_causal_gqa_attention_bf16(
        dq.data(), dk.data(), dv.data(), dout.data(), batch, seq, heads,
        kv_heads, head_dim, ctx.stream());
    if (!close(dout.copy_to_host_f32(ctx.stream()),
               cpu_attention(q, k, v, batch, seq, heads, kv_heads, head_dim))) {
      return 1;
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  std::cout << "{\"status\":\"pass\",\"attention_backend\":"
            << "\"cuda_causal_gqa_bf16_reference\"}\n";
  return 0;
}
