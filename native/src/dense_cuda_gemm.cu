#include "dense_cuda_internal.hpp"

#include <cublasLt.h>

namespace lkjai {

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
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  for (auto layout : {a, b, c}) {
    require_cublaslt(cublasLtMatrixLayoutSetAttribute(
                         layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                         sizeof(order)),
                     "cublasLt row major");
  }
  require_cublaslt(cublasLtMatmulPreferenceCreate(&pref),
                   "cublasLt preference");
  size_t workspace_bytes = 4 * 1024 * 1024;
  void* ws = workspace_.allocate(workspace_bytes);
  require_cublaslt(cublasLtMatmulPreferenceSetAttribute(
                       pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                       &workspace_bytes, sizeof(workspace_bytes)),
                   "cublasLt workspace pref");
  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned = 0;
  auto hs = cublasLtMatmulAlgoGetHeuristic(ctx_->cublaslt(), op, a, b, c, c,
                                           pref, 1, &heuristic, &returned);
  const cublasLtMatmulAlgo_t* algo =
      (hs == CUBLAS_STATUS_SUCCESS && returned > 0) ? &heuristic.algo : nullptr;
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

}  // namespace lkjai
