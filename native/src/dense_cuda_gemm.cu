#include "dense_cuda_internal.hpp"

#include <cublasLt.h>

namespace lkjai {
namespace {

void set_row_major(cublasLtMatrixLayout_t layout) {
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  require_cublaslt(cublasLtMatrixLayoutSetAttribute(
                       layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                       sizeof(order)),
                   "cublasLt row major");
}

const cublasLtMatmulAlgo_t* select_algo(cublasLtHandle_t handle,
                                        cublasLtMatmulDesc_t op,
                                        cublasLtMatrixLayout_t a,
                                        cublasLtMatrixLayout_t b,
                                        cublasLtMatrixLayout_t c,
                                        cublasLtMatmulPreference_t pref,
                                        cublasLtMatmulAlgo_t* algo) {
  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned = 0;
  auto hs = cublasLtMatmulAlgoGetHeuristic(handle, op, a, b, c, c, pref, 1,
                                           &heuristic, &returned);
  if (hs == CUBLAS_STATUS_SUCCESS && returned > 0) {
    *algo = heuristic.algo;
    return algo;
  }
  return nullptr;
}

}  // namespace

void DenseCudaState::gemm(const DeviceTensor& hidden, DeviceTensor& out,
                          int rows) {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_T;
  float alpha = 1.0f;
  float beta = 0.0f;
  int h = cfg_.hidden_size;
  int v = cfg_.vocab_size;
  require_cublaslt(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F,
                                            CUDA_R_32F),
                   "cublasLtMatmulDescCreate");
  require_cublaslt(cublasLtMatmulDescSetAttribute(
                       op, CUBLASLT_MATMUL_DESC_TRANSA, &transa,
                       sizeof(transa)),
                   "cublasLt transa");
  require_cublaslt(cublasLtMatmulDescSetAttribute(
                       op, CUBLASLT_MATMUL_DESC_TRANSB, &transb,
                       sizeof(transb)),
                   "cublasLt transb");
  require_cublaslt(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, rows, h, h),
                   "cublasLt A layout");
  require_cublaslt(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, v, h, h),
                   "cublasLt B layout");
  require_cublaslt(cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, rows, v, v),
                   "cublasLt C layout");
  for (auto layout : {a, b, c}) {
    set_row_major(layout);
  }
  require_cublaslt(cublasLtMatmulPreferenceCreate(&pref),
                   "cublasLt preference");
  size_t workspace_bytes = 4 * 1024 * 1024;
  void* ws = workspace_.allocate(workspace_bytes);
  require_cublaslt(cublasLtMatmulPreferenceSetAttribute(
                       pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                       &workspace_bytes, sizeof(workspace_bytes)),
                   "cublasLt workspace pref");
  cublasLtMatmulAlgo_t heuristic_algo{};
  const cublasLtMatmulAlgo_t* algo =
      select_algo(ctx_->cublaslt(), op, a, b, c, pref, &heuristic_algo);
  require_cublaslt(cublasLtMatmul(ctx_->cublaslt(), op, &alpha, hidden.data(),
                                  a, head_shadow_.data(), b, &beta, out.data(),
                                  c, out.data(), c, algo, ws, workspace_bytes,
                                  ctx_->stream()),
                   "dense bf16 matmul");
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(c);
  cublasLtMatrixLayoutDestroy(b);
  cublasLtMatrixLayoutDestroy(a);
  cublasLtMatmulDescDestroy(op);
}

void DenseCudaState::gemm_head_grad(const DeviceTensor& grad_logits,
                                    const DeviceTensor& hidden, int rows) {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;
  float alpha = 1.0f;
  float beta = 1.0f;
  int h = cfg_.hidden_size;
  int v = cfg_.vocab_size;
  require_cublaslt(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F,
                                            CUDA_R_32F),
                   "cublasLtMatmulDescCreate head grad");
  require_cublaslt(cublasLtMatmulDescSetAttribute(
                       op, CUBLASLT_MATMUL_DESC_TRANSA, &transa,
                       sizeof(transa)),
                   "cublasLt head grad transa");
  require_cublaslt(cublasLtMatmulDescSetAttribute(
                       op, CUBLASLT_MATMUL_DESC_TRANSB, &transb,
                       sizeof(transb)),
                   "cublasLt head grad transb");
  require_cublaslt(cublasLtMatrixLayoutCreate(&a, CUDA_R_32F, rows, v, v),
                   "cublasLt head grad A layout");
  require_cublaslt(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, rows, h, h),
                   "cublasLt head grad B layout");
  require_cublaslt(cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, v, h, h),
                   "cublasLt head grad C layout");
  for (auto layout : {a, b, c}) set_row_major(layout);
  require_cublaslt(cublasLtMatmulPreferenceCreate(&pref),
                   "cublasLt head grad preference");
  size_t workspace_bytes = 4 * 1024 * 1024;
  void* ws = workspace_.allocate(workspace_bytes);
  require_cublaslt(cublasLtMatmulPreferenceSetAttribute(
                       pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                       &workspace_bytes, sizeof(workspace_bytes)),
                   "cublasLt head grad workspace pref");
  cublasLtMatmulAlgo_t heuristic_algo{};
  const cublasLtMatmulAlgo_t* algo =
      select_algo(ctx_->cublaslt(), op, a, b, c, pref, &heuristic_algo);
  require_cublaslt(cublasLtMatmul(ctx_->cublaslt(), op, &alpha,
                                  grad_logits.data(), a, hidden.data(), b,
                                  &beta, grad_head_.data(), c,
                                  grad_head_.data(), c, algo, ws,
                                  workspace_bytes, ctx_->stream()),
                   "dense head grad matmul");
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(c);
  cublasLtMatrixLayoutDestroy(b);
  cublasLtMatrixLayoutDestroy(a);
  cublasLtMatmulDescDestroy(op);
}

void DenseCudaState::gemm_d_hidden(const DeviceTensor& grad_logits,
                                   DeviceTensor& d_hidden, int rows) {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;
  float alpha = 1.0f;
  float beta = 0.0f;
  int h = cfg_.hidden_size;
  int v = cfg_.vocab_size;
  require_cublaslt(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F,
                                            CUDA_R_32F),
                   "cublasLtMatmulDescCreate d hidden");
  require_cublaslt(cublasLtMatmulDescSetAttribute(
                       op, CUBLASLT_MATMUL_DESC_TRANSA, &transa,
                       sizeof(transa)),
                   "cublasLt d hidden transa");
  require_cublaslt(cublasLtMatmulDescSetAttribute(
                       op, CUBLASLT_MATMUL_DESC_TRANSB, &transb,
                       sizeof(transb)),
                   "cublasLt d hidden transb");
  require_cublaslt(cublasLtMatrixLayoutCreate(&a, CUDA_R_32F, rows, v, v),
                   "cublasLt d hidden A layout");
  require_cublaslt(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, v, h, h),
                   "cublasLt d hidden B layout");
  require_cublaslt(cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, rows, h, h),
                   "cublasLt d hidden C layout");
  for (auto layout : {a, b, c}) set_row_major(layout);
  require_cublaslt(cublasLtMatmulPreferenceCreate(&pref),
                   "cublasLt d hidden preference");
  size_t workspace_bytes = 4 * 1024 * 1024;
  void* ws = workspace_.allocate(workspace_bytes);
  require_cublaslt(cublasLtMatmulPreferenceSetAttribute(
                       pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                       &workspace_bytes, sizeof(workspace_bytes)),
                   "cublasLt d hidden workspace pref");
  cublasLtMatmulAlgo_t heuristic_algo{};
  const cublasLtMatmulAlgo_t* algo =
      select_algo(ctx_->cublaslt(), op, a, b, c, pref, &heuristic_algo);
  require_cublaslt(cublasLtMatmul(ctx_->cublaslt(), op, &alpha,
                                  grad_logits.data(), a, head_shadow_.data(),
                                  b, &beta, d_hidden.data(), c,
                                  d_hidden.data(), c, algo, ws,
                                  workspace_bytes, ctx_->stream()),
                   "dense d_hidden matmul");
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(c);
  cublasLtMatrixLayoutDestroy(b);
  cublasLtMatrixLayoutDestroy(a);
  cublasLtMatmulDescDestroy(op);
}

}  // namespace lkjai
