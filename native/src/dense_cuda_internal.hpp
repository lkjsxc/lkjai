#pragma once

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#include "dense_train_internal.hpp"
#include "runtime_device.hpp"

namespace lkjai {

struct CpuDenseForward {
  double loss = 0.0;
  std::vector<float> logits;
};

struct DenseMatmulPlan;

struct DenseCudaPinnedBatch {
  uint16_t* tokens = nullptr;
  uint8_t* mask = nullptr;
};

class DenseCudaState {
 public:
  DenseCudaState(const DenseConfig& cfg, const DenseTrainState& host,
                 CudaExecutionContext* ctx);
  ~DenseCudaState();
  double forward_backward(const PackedBatch& batch, std::vector<float>* logits,
                          double* h2d_seconds, double* forward_seconds,
                          double* backward_seconds, float grad_scale = 1.0f,
                          bool reset_grads = true);
  DenseCudaPinnedBatch prepare_batch_slot(int slot, size_t token_count,
                                          size_t mask_count);
  void stage_batch_slot(int slot, int batch_size, int seq_len,
                        double* h2d_seconds);
  void forward_backward_slot(int slot, bool capture_logits,
                             double* forward_seconds, double* backward_seconds,
                             float grad_scale = 1.0f,
                             bool reset_grads = true);
  void wait_batch_slot(int slot);
  double slot_loss(int slot);
  void slot_logits(int slot, std::vector<float>* logits);
  void adamw(float lr, int step);
  DenseTrainState copy_to_host();
  size_t cublaslt_workspace_bytes() const { return workspace_.bytes_reserved(); }

 private:
  void zero_gradients();
  void ensure_step_buffers(int rows);
  void gemm(const DeviceTensor& hidden, DeviceTensor& out, int rows);
  void gemm_head_grad(const DeviceTensor& grad_logits,
                      const DeviceTensor& hidden, int rows);
  void gemm_d_hidden(const DeviceTensor& grad_logits, DeviceTensor& d_hidden,
                     int rows);
  void ensure_slot_buffers(int slot, size_t token_count, size_t mask_count);

  DenseConfig cfg_;
  CudaExecutionContext* ctx_ = nullptr;
  DeviceWorkspace workspace_;
  DeviceTensor emb_;
  DeviceTensor head_;
  DeviceTensor emb_shadow_;
  DeviceTensor head_shadow_;
  DeviceTensor grad_emb_;
  DeviceTensor grad_head_;
  DeviceTensor m_emb_;
  DeviceTensor v_emb_;
  DeviceTensor m_head_;
  DeviceTensor v_head_;
  DeviceTensor step_hidden_;
  DeviceTensor step_hidden_f32_;
  DeviceTensor step_out_;
  DeviceTensor step_grad_logits_;
  DeviceTensor step_d_hidden_;
  DeviceTensor step_head_f32_;
  DeviceTensor step_loss_;
  int step_rows_ = 0;
  struct Slot {
    uint16_t* host_tokens = nullptr;
    uint8_t* host_mask = nullptr;
    uint16_t* device_tokens = nullptr;
    uint8_t* device_mask = nullptr;
    float* host_loss = nullptr;
    float* host_logits = nullptr;
    size_t token_capacity = 0;
    size_t mask_capacity = 0;
    size_t logits_capacity = 0;
    int batch_size = 0;
    int seq_len = 0;
    int supervised = 0;
    bool used = false;
    bool capture_logits = false;
    cudaEvent_t h2d_done = nullptr;
    cudaEvent_t compute_done = nullptr;
  };
  static constexpr int kBatchSlots = 3;
  Slot slots_[kBatchSlots];
  DenseMatmulPlan* logits_plan_ = nullptr;
  DenseMatmulPlan* head_grad_plan_ = nullptr;
  DenseMatmulPlan* hidden_grad_plan_ = nullptr;
};

void destroy_dense_matmul_plan(DenseMatmulPlan* plan);

double dense_seconds_since(std::chrono::steady_clock::time_point start);
float dense_step_lr(const DenseTrainOptions& opt, int step);
double dense_max_abs_diff(const std::vector<float>& a,
                          const std::vector<float>& b);
std::string dense_checksum_floats(const std::vector<float>& values);
int dense_resume_step(const std::filesystem::path& dir);
int dense_supervised_count(const PackedBatch& batch);
int dense_supervised_count_raw(const uint8_t* mask, int batch, int seq);
CpuDenseForward cpu_dense_forward_backward_with_logits(const PackedBatch& batch,
                                                       DenseTrainState* state);
DenseConfig dense_config_from_artifact(const std::filesystem::path& dir);
std::vector<float> read_dense_tensor(const std::filesystem::path& dir,
                                     const std::string& name,
                                     std::string* error);
bool dense_logits_for_tokens(const DenseConfig& cfg,
                             const std::vector<float>& emb,
                             const std::vector<float>& head,
                             const std::string& token_csv,
                             std::vector<float>* logits, std::string* error);
std::string dense_logits_check_json(const DenseConfig& cfg,
                                    const std::vector<float>& logits,
                                    const std::string& reference_status,
                                    double max_abs_diff,
                                    double mean_abs_diff, double tolerance);

void dense_launch_gather(const uint16_t* tokens, const void* emb, void* hidden,
                         int batch, int seq, int vocab, int hidden_size,
                         cudaStream_t stream);
void dense_launch_bf16_to_f32(const void* bf16, float* f32, int n,
                              cudaStream_t stream);
void dense_launch_loss_grad(const float* logits, const uint16_t* tokens,
                            const uint8_t* mask, float* grad_logits,
                            float* loss, int batch, int seq, int vocab,
                            int supervised, float grad_scale,
                            cudaStream_t stream);
void dense_launch_scatter_emb_grad(const uint16_t* tokens,
                                   const float* d_hidden, float* grad_emb,
                                   int batch, int seq, int vocab,
                                   int hidden_size, cudaStream_t stream);
void dense_launch_adamw(float* weight, float* m, float* v, const float* grad,
                        void* shadow, int n, float lr, int step,
                        cudaStream_t stream);

}  // namespace lkjai
