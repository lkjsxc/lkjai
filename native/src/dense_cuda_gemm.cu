#include "dense_cuda_internal.hpp"

#include <cublasLt.h>

namespace lkjai {
namespace {

enum class DenseGemmKind { logits, head_grad, hidden_grad };

void set_row_major(cublasLtMatrixLayout_t layout) {
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  require_cublaslt(cublasLtMatrixLayoutSetAttribute(
                       layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                       sizeof(order)),
                   "cublasLt row major");
}

struct Ops {
  cublasOperation_t transa;
  cublasOperation_t transb;
  cudaDataType_t a_type;
  cudaDataType_t b_type;
  cudaDataType_t c_type;
  int a_rows;
  int a_cols;
  int b_rows;
  int b_cols;
  int c_rows;
  int c_cols;
};

Ops ops_for(DenseGemmKind kind, int rows, int vocab, int hidden) {
  if (kind == DenseGemmKind::logits) {
    return {CUBLAS_OP_N, CUBLAS_OP_T, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F,
            rows, hidden, vocab, hidden, rows, vocab};
  }
  if (kind == DenseGemmKind::head_grad) {
    return {CUBLAS_OP_T, CUBLAS_OP_N, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F,
            rows, vocab, rows, hidden, vocab, hidden};
  }
  return {CUBLAS_OP_N, CUBLAS_OP_N, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F,
          rows, vocab, vocab, hidden, rows, hidden};
}

DenseMatmulPlan* ensure_plan(DenseMatmulPlan** slot, DenseGemmKind kind,
                             cublasLtHandle_t handle, int rows, int vocab,
                             int hidden);

}  // namespace

struct DenseMatmulPlan {
  DenseGemmKind kind = DenseGemmKind::logits;
  int rows = 0;
  int vocab = 0;
  int hidden = 0;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t c = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  cublasLtMatmulAlgo_t algo{};
  bool has_algo = false;
  size_t workspace_bytes = 4 * 1024 * 1024;
};

namespace {

void destroy_plan(DenseMatmulPlan* plan) {
  if (!plan) return;
  if (plan->pref) cublasLtMatmulPreferenceDestroy(plan->pref);
  if (plan->c) cublasLtMatrixLayoutDestroy(plan->c);
  if (plan->b) cublasLtMatrixLayoutDestroy(plan->b);
  if (plan->a) cublasLtMatrixLayoutDestroy(plan->a);
  if (plan->op) cublasLtMatmulDescDestroy(plan->op);
  delete plan;
}

DenseMatmulPlan* make_plan(DenseGemmKind kind, cublasLtHandle_t handle,
                           int rows, int vocab, int hidden) {
  auto* plan = new DenseMatmulPlan();
  plan->kind = kind;
  plan->rows = rows;
  plan->vocab = vocab;
  plan->hidden = hidden;
  auto ops = ops_for(kind, rows, vocab, hidden);
  require_cublaslt(cublasLtMatmulDescCreate(&plan->op, CUBLAS_COMPUTE_32F,
                                            CUDA_R_32F),
                   "cublasLtMatmulDescCreate");
  require_cublaslt(cublasLtMatmulDescSetAttribute(
                       plan->op, CUBLASLT_MATMUL_DESC_TRANSA, &ops.transa,
                       sizeof(ops.transa)),
                   "cublasLt transa");
  require_cublaslt(cublasLtMatmulDescSetAttribute(
                       plan->op, CUBLASLT_MATMUL_DESC_TRANSB, &ops.transb,
                       sizeof(ops.transb)),
                   "cublasLt transb");
  require_cublaslt(cublasLtMatrixLayoutCreate(
                       &plan->a, ops.a_type, ops.a_rows, ops.a_cols,
                       ops.a_cols), "cublasLt A layout");
  require_cublaslt(cublasLtMatrixLayoutCreate(
                       &plan->b, ops.b_type, ops.b_rows, ops.b_cols,
                       ops.b_cols), "cublasLt B layout");
  require_cublaslt(cublasLtMatrixLayoutCreate(
                       &plan->c, ops.c_type, ops.c_rows, ops.c_cols,
                       ops.c_cols), "cublasLt C layout");
  for (auto layout : {plan->a, plan->b, plan->c}) set_row_major(layout);
  require_cublaslt(cublasLtMatmulPreferenceCreate(&plan->pref),
                   "cublasLt preference");
  require_cublaslt(cublasLtMatmulPreferenceSetAttribute(
                       plan->pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                       &plan->workspace_bytes, sizeof(plan->workspace_bytes)),
                   "cublasLt workspace pref");
  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned = 0;
  auto hs = cublasLtMatmulAlgoGetHeuristic(handle, plan->op, plan->a, plan->b,
                                           plan->c, plan->c, plan->pref, 1,
                                           &heuristic, &returned);
  if (hs == CUBLAS_STATUS_SUCCESS && returned > 0) {
    plan->algo = heuristic.algo;
    plan->has_algo = true;
  }
  return plan;
}

DenseMatmulPlan* ensure_plan(DenseMatmulPlan** slot, DenseGemmKind kind,
                             cublasLtHandle_t handle, int rows, int vocab,
                             int hidden) {
  auto* plan = *slot;
  if (plan && plan->kind == kind && plan->rows == rows &&
      plan->vocab == vocab && plan->hidden == hidden) {
    return plan;
  }
  destroy_plan(plan);
  *slot = make_plan(kind, handle, rows, vocab, hidden);
  return *slot;
}

}  // namespace

void destroy_dense_matmul_plan(DenseMatmulPlan* plan) { destroy_plan(plan); }

void DenseCudaState::gemm(const DeviceTensor& hidden, DeviceTensor& out,
                          int rows) {
  auto* plan = ensure_plan(&logits_plan_, DenseGemmKind::logits,
                           ctx_->cublaslt(), rows, cfg_.vocab_size,
                           cfg_.hidden_size);
  float alpha = 1.0f, beta = 0.0f;
  void* ws = workspace_.allocate(plan->workspace_bytes);
  require_cublaslt(cublasLtMatmul(ctx_->cublaslt(), plan->op, &alpha,
                                  hidden.data(), plan->a, head_shadow_.data(),
                                  plan->b, &beta, out.data(), plan->c,
                                  out.data(), plan->c,
                                  plan->has_algo ? &plan->algo : nullptr, ws,
                                  plan->workspace_bytes, ctx_->stream()),
                   "dense bf16 matmul");
}

void DenseCudaState::gemm_head_grad(const DeviceTensor& grad_logits,
                                    const DeviceTensor& hidden, int rows) {
  auto* plan = ensure_plan(&head_grad_plan_, DenseGemmKind::head_grad,
                           ctx_->cublaslt(), rows, cfg_.vocab_size,
                           cfg_.hidden_size);
  float alpha = 1.0f, beta = 1.0f;
  void* ws = workspace_.allocate(plan->workspace_bytes);
  require_cublaslt(cublasLtMatmul(ctx_->cublaslt(), plan->op, &alpha,
                                  grad_logits.data(), plan->a, hidden.data(),
                                  plan->b, &beta, grad_head_.data(), plan->c,
                                  grad_head_.data(), plan->c,
                                  plan->has_algo ? &plan->algo : nullptr, ws,
                                  plan->workspace_bytes, ctx_->stream()),
                   "dense head grad matmul");
}

void DenseCudaState::gemm_d_hidden(const DeviceTensor& grad_logits,
                                   DeviceTensor& d_hidden, int rows) {
  auto* plan = ensure_plan(&hidden_grad_plan_, DenseGemmKind::hidden_grad,
                           ctx_->cublaslt(), rows, cfg_.vocab_size,
                           cfg_.hidden_size);
  float alpha = 1.0f, beta = 0.0f;
  void* ws = workspace_.allocate(plan->workspace_bytes);
  require_cublaslt(cublasLtMatmul(ctx_->cublaslt(), plan->op, &alpha,
                                  grad_logits.data(), plan->a,
                                  step_head_f32_.data(), plan->b, &beta,
                                  d_hidden.data(), plan->c, d_hidden.data(),
                                  plan->c,
                                  plan->has_algo ? &plan->algo : nullptr, ws,
                                  plan->workspace_bytes, ctx_->stream()),
                   "dense d_hidden matmul");
}

}  // namespace lkjai
