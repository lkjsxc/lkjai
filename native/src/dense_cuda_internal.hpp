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

class DenseCudaState {
 public:
  DenseCudaState(const DenseConfig& cfg, const DenseTrainState& host,
                 CudaExecutionContext* ctx);
  ~DenseCudaState();
  double forward_backward(const PackedBatch& batch, std::vector<float>* logits,
                          double* h2d_seconds, double* forward_seconds,
                          double* backward_seconds, float grad_scale = 1.0f,
                          bool reset_grads = true);
  void adamw(float lr, int step);
  DenseTrainState copy_to_host();

 private:
  void zero_gradients();
  void ensure_batch_buffers(size_t token_count, size_t mask_count);
  void gemm(const DeviceTensor& hidden, DeviceTensor& out, int rows);
  void gemm_head_grad(const DeviceTensor& grad_logits,
                      const DeviceTensor& hidden, int rows);
  void gemm_d_hidden(const DeviceTensor& grad_logits, DeviceTensor& d_hidden,
                     int rows);

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
  uint16_t* host_tokens_ = nullptr;
  uint8_t* host_mask_ = nullptr;
  uint16_t* device_tokens_ = nullptr;
  uint8_t* device_mask_ = nullptr;
  size_t host_token_capacity_ = 0;
  size_t host_mask_capacity_ = 0;
  size_t device_token_capacity_ = 0;
  size_t device_mask_capacity_ = 0;
};

double dense_seconds_since(std::chrono::steady_clock::time_point start);
float dense_step_lr(const DenseTrainOptions& opt, int step);
double dense_max_abs_diff(const std::vector<float>& a,
                          const std::vector<float>& b);
std::string dense_checksum_floats(const std::vector<float>& values);
int dense_resume_step(const std::filesystem::path& dir);
int dense_supervised_count(const PackedBatch& batch);
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
void dense_launch_loss_grad(const float* logits, const uint16_t* tokens,
                            const uint8_t* mask, float* grad_logits,
                            float* loss, int batch, int seq, int vocab,
                            int supervised, float grad_scale,
                            cudaStream_t stream);
void dense_launch_emb_scatter(const float* d_hidden, const uint16_t* tokens,
                              float* grad_emb, int batch, int seq, int vocab,
                              int hidden_size, cudaStream_t stream);
void dense_launch_adamw(float* weight, float* m, float* v, const float* grad,
                        void* shadow, int n, float lr, int step,
                        cudaStream_t stream);

}  // namespace lkjai
