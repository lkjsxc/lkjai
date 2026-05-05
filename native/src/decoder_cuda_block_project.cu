#include "decoder_cuda_block_internal.hpp"

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>

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

struct LtProjectionKey {
  int rows = 0;
  int in_features = 0;
  int out_features = 0;
  size_t workspace_bytes = 0;
  bool operator<(const LtProjectionKey& other) const {
    return std::tie(rows, in_features, out_features, workspace_bytes) <
           std::tie(other.rows, other.in_features, other.out_features,
                    other.workspace_bytes);
  }
};

std::mutex& plan_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::map<LtProjectionKey, std::unique_ptr<LtProjectionPlan>>& plan_cache() {
  static auto* cache =
      new std::map<LtProjectionKey, std::unique_ptr<LtProjectionPlan>>();
  return *cache;
}

LtProjectionPlan* cached_plan(int rows, int in_features, int out_features,
                              size_t workspace_bytes) {
  std::lock_guard<std::mutex> lock(plan_mutex());
  LtProjectionKey key{rows, in_features, out_features, workspace_bytes};
  auto& cache = plan_cache();
  auto found = cache.find(key);
  if (found != cache.end()) return found->second.get();
  auto inserted = cache.emplace(
      key, std::make_unique<LtProjectionPlan>(rows, in_features, out_features,
                                              workspace_bytes));
  return inserted.first->second.get();
}

}  // namespace

void decoder_cuda_project_bf16(cublasLtHandle_t handle, cudaStream_t stream,
                               const void* x_bf16, const void* w_bf16,
                               void* y_bf16, int rows, int in_features,
                               int out_features, void* workspace,
                               size_t workspace_bytes) {
  LtProjectionPlan* plan =
      cached_plan(rows, in_features, out_features, workspace_bytes);
  float alpha = 1.0f;
  float beta = 0.0f;
  require_cublaslt(cublasLtMatmul(
                       handle, plan->op, &alpha, x_bf16, plan->a, w_bf16,
                       plan->b, &beta, y_bf16, plan->c, y_bf16, plan->c, nullptr,
                       workspace, workspace_bytes, stream),
                   "decoder bf16 projection matmul");
}

size_t decoder_cuda_projection_plan_cache_size() {
  std::lock_guard<std::mutex> lock(plan_mutex());
  return plan_cache().size();
}

void decoder_cuda_projection_plan_cache_reset() {
  std::lock_guard<std::mutex> lock(plan_mutex());
  plan_cache().clear();
}

}  // namespace lkjai
