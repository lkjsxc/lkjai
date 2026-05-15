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
                   "decoder backward cublasLt row major");
}

struct Layout {
  cublasLtMatrixLayout_t value = nullptr;
  Layout(cudaDataType_t type, int rows, int cols) {
    require_cublaslt(cublasLtMatrixLayoutCreate(&value, type, rows, cols, cols),
                     "decoder backward layout");
    set_row_major(value);
  }
  ~Layout() {
    if (value) cublasLtMatrixLayoutDestroy(value);
  }
};

struct Desc {
  cublasLtMatmulDesc_t value = nullptr;
  Desc(cublasOperation_t a, cublasOperation_t b) {
    require_cublaslt(cublasLtMatmulDescCreate(&value, CUBLAS_COMPUTE_32F,
                                              CUDA_R_32F),
                     "decoder backward matmul desc");
    require_cublaslt(cublasLtMatmulDescSetAttribute(
                         value, CUBLASLT_MATMUL_DESC_TRANSA, &a, sizeof(a)),
                     "decoder backward transa");
    require_cublaslt(cublasLtMatmulDescSetAttribute(
                         value, CUBLASLT_MATMUL_DESC_TRANSB, &b, sizeof(b)),
                     "decoder backward transb");
  }
  ~Desc() {
    if (value) cublasLtMatmulDescDestroy(value);
  }
};

void matmul(cublasLtHandle_t handle, cudaStream_t stream,
            cublasLtMatmulDesc_t desc, const void* a,
            cublasLtMatrixLayout_t a_layout, const void* b,
            cublasLtMatrixLayout_t b_layout, void* c,
            cublasLtMatrixLayout_t c_layout, void* workspace,
            size_t workspace_bytes, float beta, const char* label) {
  float alpha = 1.0f;
  require_cublaslt(cublasLtMatmul(handle, desc, &alpha, a, a_layout, b,
                                  b_layout, &beta, c, c_layout, c, c_layout,
                                  nullptr, workspace, workspace_bytes, stream),
                   label);
}

}  // namespace

void decoder_cuda_project_backward_bf16(
    cublasLtHandle_t handle, cudaStream_t stream, const void* x_bf16,
    const void* w_bf16, const void* dy_bf16, void* dx_f32, void* dw_f32,
    int rows, int in_features, int out_features, void* workspace,
    size_t workspace_bytes, float dw_beta) {
  Desc dx_desc(CUBLAS_OP_N, CUBLAS_OP_N);
  Layout dy(CUDA_R_16BF, rows, out_features);
  Layout w(CUDA_R_16BF, out_features, in_features);
  Layout dx(CUDA_R_32F, rows, in_features);
  matmul(handle, stream, dx_desc.value, dy_bf16, dy.value, w_bf16, w.value,
         dx_f32, dx.value, workspace, workspace_bytes, 0.0f,
         "decoder projection backward dX");

  Desc dw_desc(CUBLAS_OP_T, CUBLAS_OP_N);
  Layout x(CUDA_R_16BF, rows, in_features);
  Layout dw(CUDA_R_32F, out_features, in_features);
  matmul(handle, stream, dw_desc.value, dy_bf16, dy.value, x_bf16, x.value,
         dw_f32, dw.value, workspace, workspace_bytes, dw_beta,
         "decoder projection backward dW");
}

}  // namespace lkjai
