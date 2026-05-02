#include "dense_cuda.hpp"

#include <algorithm>
#include <cmath>

#include "cuda_probe.hpp"
#include "dense_cuda_internal.hpp"

namespace lkjai {
namespace {

void fill_capability(const CudaStatus& status, DenseCudaCheck* check) {
  check->device = status.device;
  check->compute_major = status.compute_major;
  check->compute_minor = status.compute_minor;
  check->cuda_runtime_version = status.cuda_runtime_version;
  check->cudnn_version = status.cudnn_version;
  check->bf16_supported = status.bf16_supported;
  check->cublaslt_available = status.cublaslt_available;
  check->cudnn_available = status.cudnn_available;
  check->sdpa_eligible = status.sdpa_eligible;
  check->async_alloc_supported = status.async_alloc_supported;
  check->error = status.error;
}

DenseConfig parity_config() {
  DenseConfig cfg;
  cfg.vocab_size = 16;
  cfg.context = 5;
  cfg.hidden_size = 8;
  cfg.heads = 2;
  cfg.kv_heads = 2;
  cfg.head_dim = 4;
  cfg.ffn_size = 16;
  cfg.seed = 42;
  return cfg;
}

PackedBatch parity_batch() {
  PackedBatch batch;
  batch.batch_size = 1;
  batch.sequence_len = 5;
  batch.tokens = {1, 2, 3, 4, 5};
  batch.loss_mask = {1, 1, 1, 1, 1};
  return batch;
}

}  // namespace

DenseCudaCheck run_dense_cuda_check() {
  DenseCudaCheck check;
  auto status = cuda_status();
  fill_capability(status, &check);
  if (!cuda_required_ok(status)) {
    if (check.error.empty()) check.error = status.warning;
    return check;
  }
  try {
    auto cfg = parity_config();
    DenseTrainState master;
    init_dense_state(cfg, &master);
    DenseTrainState forward_ref = master;
    for (float& v : forward_ref.emb) v = dense_round_bf16(v);
    for (float& v : forward_ref.head) v = dense_round_bf16(v);
    DenseTrainState cpu = master;
    auto batch = parity_batch();
    auto cpu_out = cpu_dense_forward_backward_with_logits(batch, &forward_ref);
    cpu.grad_emb = forward_ref.grad_emb;
    cpu.grad_head = forward_ref.grad_head;
    dense_adamw(&cpu.emb, &cpu.m_emb, &cpu.v_emb, cpu.grad_emb, 1.0e-3f, 1);
    dense_adamw(&cpu.head, &cpu.m_head, &cpu.v_head, cpu.grad_head, 1.0e-3f, 1);

    CudaExecutionContext ctx;
    DenseCudaState state(cfg, master, &ctx);
    std::vector<float> logits;
    check.loss =
        state.forward_backward(batch, &logits, nullptr, nullptr, nullptr);
    state.adamw(1.0e-3f, 1);
    auto got = state.copy_to_host();
    check.cpu_loss = cpu_out.loss;
    check.max_logit_diff = dense_max_abs_diff(logits, cpu_out.logits);
    check.max_grad_diff =
        std::max(dense_max_abs_diff(got.grad_emb, cpu.grad_emb),
                 dense_max_abs_diff(got.grad_head, cpu.grad_head));
    check.max_update_diff =
        std::max(dense_max_abs_diff(got.emb, cpu.emb),
                 dense_max_abs_diff(got.head, cpu.head));
    check.ok = std::fabs(check.loss - check.cpu_loss) < 1.0e-3 &&
               check.max_logit_diff < 1.0e-3 &&
               check.max_grad_diff < 1.0e-3 &&
               check.max_update_diff < 1.0e-3;
    if (!check.ok) check.error = "dense CUDA parity mismatch";
  } catch (const std::exception& e) {
    check.error = e.what();
  }
  return check;
}

}  // namespace lkjai
