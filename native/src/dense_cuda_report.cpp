#include "dense_cuda_internal.hpp"

namespace lkjai {

void dense_fill_runtime_report(DenseCudaState& state, DenseTrainReport* report) {
  report->cublaslt_workspace_bytes = state.cublaslt_workspace_bytes();
  auto logits = state.logits_matmul_stats();
  auto head = state.head_grad_matmul_stats();
  auto hidden = state.hidden_grad_matmul_stats();
  report->dense_autotune_enabled = state.tuning().autotune_enabled();
  report->dense_autotune_mode = state.tuning().autotune_mode;
  report->dense_workspace_sweep_bytes = state.tuning().workspace_sweep;
  report->dense_cublaslt_logits_algo_id = logits.algo_id;
  report->dense_cublaslt_head_grad_algo_id = head.algo_id;
  report->dense_cublaslt_hidden_grad_algo_id = hidden.algo_id;
  report->dense_cublaslt_logits_workspace_bytes = logits.workspace_bytes;
  report->dense_cublaslt_head_grad_workspace_bytes = head.workspace_bytes;
  report->dense_cublaslt_hidden_grad_workspace_bytes = hidden.workspace_bytes;
  report->dense_allocator_backend = state.workspace_backend();
  report->dense_async_alloc_supported = state.workspace_async_supported();
  report->dense_mempool_release_threshold_bytes =
      state.workspace_release_threshold_bytes();
  report->dense_workspace_high_water_bytes = state.workspace_high_water_bytes();
  report->dense_workspace_reallocations = state.workspace_reallocations();
  report->dense_timing_mode = state.tuning().timing_mode;
  report->dense_head_f32_cache_refreshes = state.head_f32_cache_refreshes();
}

}  // namespace lkjai
