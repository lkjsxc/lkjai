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
  double forward_backward(const PackedBatch& batch, std::vector<float>* logits,
                          double* forward_seconds, double* backward_seconds,
                          float grad_scale = 1.0f,
                          bool reset_grads = true);
  void adamw(float lr, int step);
  DenseTrainState copy_to_host();

 private:
  void zero_gradients();
  void gemm(const DeviceTensor& hidden, DeviceTensor& out, int rows);

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
};

double dense_seconds_since(std::chrono::steady_clock::time_point start);
float dense_step_lr(const DenseTrainOptions& opt, int step);
float dense_round_bf16(float value);
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

void dense_launch_gather(const uint16_t* tokens, const void* emb, void* hidden,
                         int batch, int seq, int vocab, int hidden_size,
                         cudaStream_t stream);
void dense_launch_loss_grad(const float* logits, const uint16_t* tokens,
                            const uint8_t* mask, float* grad_logits,
                            float* loss, int batch, int seq, int vocab,
                            int supervised, float grad_scale,
                            cudaStream_t stream);
void dense_launch_head_grad(const float* grad_logits, const void* hidden,
                            float* grad_head, int rows, int vocab,
                            int hidden_size, cudaStream_t stream);
void dense_launch_emb_grad(const float* grad_logits, const void* head,
                           const uint16_t* tokens, float* grad_emb, int batch,
                           int seq, int vocab, int hidden_size,
                           cudaStream_t stream);
void dense_launch_adamw(float* weight, float* m, float* v, const float* grad,
                        void* shadow, int n, float lr, int step,
                        cudaStream_t stream);

}  // namespace lkjai
