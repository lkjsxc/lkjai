#include "dense_cuda_internal.hpp"

#include <cuda_bf16.h>

namespace lkjai {
namespace {

constexpr float kB1 = 0.9f;
constexpr float kB2 = 0.999f;
constexpr float kEps = 1.0e-8f;
constexpr float kWd = 0.01f;

__global__ void gather_kernel(const uint16_t* tokens, const __nv_bfloat16* emb,
                              __nv_bfloat16* hidden, int batch, int seq,
                              int vocab, int hidden_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * (seq - 1) * hidden_size;
  if (idx >= total) return;
  int h = idx % hidden_size;
  int row_pos = idx / hidden_size;
  int row = row_pos / (seq - 1);
  int pos = row_pos % (seq - 1);
  int token = static_cast<int>(tokens[row * seq + pos]) % vocab;
  hidden[idx] = emb[token * hidden_size + h];
}

__global__ void loss_kernel(const float* logits, const uint16_t* tokens,
                            const uint8_t* mask, float* grad_logits,
                            float* loss_out, int batch, int seq, int vocab,
                            int supervised) {
  int row_pos = blockIdx.x * blockDim.x + threadIdx.x;
  int rows = batch * (seq - 1);
  if (row_pos >= rows) return;
  int row = row_pos / (seq - 1);
  int pos = row_pos % (seq - 1);
  int token_base = row * seq + pos;
  auto* row_grad = grad_logits + static_cast<size_t>(row_pos) * vocab;
  if (mask[token_base + 1] == 0 || supervised <= 0) {
    for (int v = 0; v < vocab; ++v) row_grad[v] = 0.0f;
    return;
  }
  auto* row_logits = logits + static_cast<size_t>(row_pos) * vocab;
  int label = static_cast<int>(tokens[token_base + 1]) % vocab;
  float max_logit = -INFINITY;
  for (int v = 0; v < vocab; ++v) max_logit = fmaxf(max_logit, row_logits[v]);
  float denom = 0.0f;
  for (int v = 0; v < vocab; ++v) denom += expf(row_logits[v] - max_logit);
  float label_prob = fmaxf(expf(row_logits[label] - max_logit) / denom,
                           1.0e-20f);
  atomicAdd(loss_out, -logf(label_prob) / static_cast<float>(supervised));
  float scale = 1.0f / static_cast<float>(supervised);
  for (int v = 0; v < vocab; ++v) {
    float prob = expf(row_logits[v] - max_logit) / denom;
    row_grad[v] = (prob - (v == label ? 1.0f : 0.0f)) * scale;
  }
}

__global__ void head_grad_kernel(const float* grad_logits,
                                 const __nv_bfloat16* hidden, float* grad_head,
                                 int rows, int vocab, int hidden_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = vocab * hidden_size;
  if (idx >= total) return;
  int h = idx % hidden_size;
  int v = idx / hidden_size;
  float sum = 0.0f;
  for (int n = 0; n < rows; ++n) {
    sum += grad_logits[static_cast<size_t>(n) * vocab + v] *
           __bfloat162float(hidden[static_cast<size_t>(n) * hidden_size + h]);
  }
  grad_head[idx] = sum;
}

__global__ void emb_grad_kernel(const float* grad_logits,
                                const __nv_bfloat16* head,
                                const uint16_t* tokens, float* grad_emb,
                                int batch, int seq, int vocab, int hidden_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int rows = batch * (seq - 1);
  if (idx >= rows * hidden_size) return;
  int h = idx % hidden_size;
  int row_pos = idx / hidden_size;
  int row = row_pos / (seq - 1);
  int pos = row_pos % (seq - 1);
  int token = static_cast<int>(tokens[row * seq + pos]) % vocab;
  float sum = 0.0f;
  for (int v = 0; v < vocab; ++v) {
    sum += grad_logits[static_cast<size_t>(row_pos) * vocab + v] *
           __bfloat162float(head[static_cast<size_t>(v) * hidden_size + h]);
  }
  atomicAdd(grad_emb + static_cast<size_t>(token) * hidden_size + h, sum);
}

__global__ void adamw_kernel(float* weight, float* m, float* v, const float* grad,
                             __nv_bfloat16* shadow, int n, float lr, int step) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float bc1 = 1.0f - powf(kB1, static_cast<float>(step));
  float bc2 = 1.0f - powf(kB2, static_cast<float>(step));
  float mi = kB1 * m[i] + (1.0f - kB1) * grad[i];
  float vi = kB2 * v[i] + (1.0f - kB2) * grad[i] * grad[i];
  m[i] = mi;
  v[i] = vi;
  float update = (mi / bc1) / (sqrtf(vi / bc2) + kEps);
  float next = weight[i] - lr * (update + kWd * weight[i]);
  weight[i] = next;
  shadow[i] = __float2bfloat16(next);
}

}  // namespace

void dense_launch_gather(const uint16_t* tokens, const void* emb, void* hidden,
                         int batch, int seq, int vocab, int hidden_size,
                         cudaStream_t stream) {
  int n = batch * (seq - 1) * hidden_size;
  gather_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      tokens, static_cast<const __nv_bfloat16*>(emb),
      static_cast<__nv_bfloat16*>(hidden), batch, seq, vocab, hidden_size);
  require_cuda(cudaGetLastError(), "gather_embeddings_kernel");
}

void dense_launch_loss_grad(const float* logits, const uint16_t* tokens,
                            const uint8_t* mask, float* grad_logits,
                            float* loss, int batch, int seq, int vocab,
                            int supervised, cudaStream_t stream) {
  int rows = batch * (seq - 1);
  loss_kernel<<<(rows + 127) / 128, 128, 0, stream>>>(
      logits, tokens, mask, grad_logits, loss, batch, seq, vocab, supervised);
  require_cuda(cudaGetLastError(), "loss_grad_kernel");
}

void dense_launch_head_grad(const float* grad_logits, const void* hidden,
                            float* grad_head, int rows, int vocab,
                            int hidden_size, cudaStream_t stream) {
  int n = vocab * hidden_size;
  head_grad_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      grad_logits, static_cast<const __nv_bfloat16*>(hidden), grad_head, rows,
      vocab, hidden_size);
  require_cuda(cudaGetLastError(), "head_grad_kernel");
}

void dense_launch_emb_grad(const float* grad_logits, const void* head,
                           const uint16_t* tokens, float* grad_emb, int batch,
                           int seq, int vocab, int hidden_size,
                           cudaStream_t stream) {
  int n = batch * (seq - 1) * hidden_size;
  emb_grad_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      grad_logits, static_cast<const __nv_bfloat16*>(head), tokens, grad_emb,
      batch, seq, vocab, hidden_size);
  require_cuda(cudaGetLastError(), "emb_grad_kernel");
}

void dense_launch_adamw(float* weight, float* m, float* v, const float* grad,
                        void* shadow, int n, float lr, int step,
                        cudaStream_t stream) {
  adamw_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      weight, m, v, grad, static_cast<__nv_bfloat16*>(shadow), n, lr, step);
  require_cuda(cudaGetLastError(), "adamw_kernel");
}

}  // namespace lkjai
