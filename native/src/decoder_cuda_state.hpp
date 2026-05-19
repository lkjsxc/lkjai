#pragma once

#include <string>
#include <vector>

#include "decoder_cuda_slice_internal.hpp"
#include "decoder_cuda_layer_forward.hpp"
#include "decoder_cuda_tape.hpp"
#include "dense_cuda_internal.hpp"
#include "transformer_train.hpp"
#include "transformer_state.hpp"

namespace lkjai {

struct DecoderCudaStepResult {
  double loss = 0.0;
  std::vector<float> logits;
};

class DecoderCudaState {
 public:
  struct RegistryTensor {
    Parameter* param = nullptr;
    std::string name;
    std::string role;
    std::string tied_alias;
    DeviceTensor weight;
    DeviceTensor grad;
    DeviceTensor moment_m;
    DeviceTensor moment_v;
    DeviceTensor shadow;
  };

  DecoderCudaState(const TransformerConfig& cfg,
                   const TransformerState& initial,
                   bool require_cudnn_attention = false);

  double forward_backward(const PackedBatch& batch, std::vector<float>* logits,
                          double* h2d_seconds, double* forward_seconds,
                          double* backward_seconds, float grad_scale,
                          bool reset_grads);
  std::vector<float> debug_last_grad_logits();
  void optimizer_step(float lr, int step);
  TransformerState copy_to_host();
  void fill_report(TransformerTrainReport* report);
  void record_weight_change(const TransformerState& before,
                            TransformerTrainReport* report);
  DenseCudaState& dense_cuda() { return dense_cuda_; }
  std::vector<float> debug_last_final_norm_input();
  std::vector<float> debug_last_final_norm();
  std::vector<float> debug_last_grad_final_norm();
  std::vector<float> debug_last_grad_final_norm_input();
  std::vector<float> debug_last_layer_tape(int layer, const std::string& name);
  size_t debug_last_layer_tape_elements(int layer, const std::string& name);

 private:
  void build_registry();
  void copy_registry_to_host();
  void sync_registry_from_host();
  void sync_registry_grads_from_host();
  void refresh_layer_forwards();
  void refresh_layer_forwards_from_registry();
  bool decoder_parity_sample_this_step();
  void record_decoder_parity(double loss, const std::vector<float>* logits,
                             const PackedBatch& batch, bool compare_logits);
  void ensure_tape_capacity(int rows, int vocab, int hidden, int layers);
  void run_device_backward(float loss, int batch_size, int sequence_len,
                           int capture_row, float grad_scale,
                           bool reset_grads);
  DeviceTensor* run_device_layer_backward(int layer_index, int batch_size,
                                          int sequence_len,
                                          DeviceTensor* upstream);
  RegistryTensor* find_registry_tensor(const std::string& name);
  void scale_and_accumulate_grads(const TransformerState& previous,
                                  float grad_scale, bool reset_grads);

  TransformerState state_;
  CudaExecutionContext ctx_;
  DenseTrainState dense_host_;
  DenseCudaState dense_cuda_;
  DeviceWorkspace workspace_;
  std::vector<DecoderCudaLayerForward> layer_forwards_;
  DecoderCudaTape tape_;
  std::vector<RegistryTensor> registry_;
  uint64_t registry_shadow_bytes_ = 0;
  uint64_t optimizer_step_d2h_bytes_ = 0;
  uint64_t full_registry_d2h_bytes_ = 0;
  bool require_cudnn_attention_ = false;
  DecoderCudaRuntimeEvidence runtime_evidence_;
  std::string parity_mode_ = "off";
  int parity_interval_ = 128;
  int parity_calls_ = 0;
  int parity_sample_count_ = 0;
  int parity_failure_count_ = 0;
  std::string parity_sample_status_ = "not_sampled";
  double parity_sample_loss_diff_ = 0.0;
  double parity_sample_logits_max_diff_ = 0.0;
  double parity_sample_logits_mean_diff_ = 0.0;
};

}  // namespace lkjai
