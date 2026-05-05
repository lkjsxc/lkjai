#include "decoder_cuda_block_internal.hpp"

#include <cstddef>

#include "runtime_device.hpp"

namespace lkjai {
namespace {

void set_row_major(cublasLtMatrixLayout_t layout) {
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  require_cublaslt(cublasLtMatrixLayoutSetAttribute(
                       layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                       sizeof(order)),
                   "decoder cublasLt row major");
}

struct LtProjectionPlan {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t c = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;

  LtProjectionPlan(int rows, int in_features, int out_features,
                   size_t workspace_bytes) {
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_T;
    require_cublaslt(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F,
                                              CUDA_R_32F),
                     "decoder cublasLtMatmulDescCreate");
    require_cublaslt(cublasLtMatmulDescSetAttribute(
                         op, CUBLASLT_MATMUL_DESC_TRANSA, &transa,
                         sizeof(transa)),
                     "decoder cublasLt transa");
    require_cublaslt(cublasLtMatmulDescSetAttribute(
                         op, CUBLASLT_MATMUL_DESC_TRANSB, &transb,
                         sizeof(transb)),
                     "decoder cublasLt transb");
    require_cublaslt(cublasLtMatrixLayoutCreate(
                         &a, CUDA_R_16BF, rows, in_features, in_features),
                     "decoder cublasLt A layout");
    require_cublaslt(cublasLtMatrixLayoutCreate(
                         &b, CUDA_R_16BF, out_features, in_features,
                         in_features),
                     "decoder cublasLt B layout");
    require_cublaslt(cublasLtMatrixLayoutCreate(
                         &c, CUDA_R_16BF, rows, out_features, out_features),
                     "decoder cublasLt C layout");
    set_row_major(a);
    set_row_major(b);
    set_row_major(c);
    require_cublaslt(cublasLtMatmulPreferenceCreate(&pref),
                     "decoder cublasLt preference");
    require_cublaslt(cublasLtMatmulPreferenceSetAttribute(
                         pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                         &workspace_bytes, sizeof(workspace_bytes)),
                     "decoder cublasLt workspace preference");
  }

  ~LtProjectionPlan() {
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (op) cublasLtMatmulDescDestroy(op);
  }
};

}  // namespace

void decoder_cuda_project_bf16(cublasLtHandle_t handle, cudaStream_t stream,
                               const void* x_bf16, const void* w_bf16,
                               void* y_bf16, int rows, int in_features,
                               int out_features, void* workspace,
                               size_t workspace_bytes) {
  LtProjectionPlan plan(rows, in_features, out_features, workspace_bytes);
  float alpha = 1.0f;
  float beta = 0.0f;
  require_cublaslt(cublasLtMatmul(
                       handle, plan.op, &alpha, x_bf16, plan.a, w_bf16,
                       plan.b, &beta, y_bf16, plan.c, y_bf16, plan.c, nullptr,
                       workspace, workspace_bytes, stream),
                   "decoder bf16 projection matmul");
}

}  // namespace lkjai
