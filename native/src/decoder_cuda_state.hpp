#pragma once

#include <string>
#include <vector>

#include "decoder_cuda_slice_internal.hpp"
#include "decoder_cuda_layer_forward.hpp"
#include "decoder_cuda_tape.hpp"
#include "dense_cuda_internal.hpp"
#include "transformer_state.hpp"

namespace lkjai {

struct DecoderCudaStepResult {
  double loss = 0.0;
  std::vector<float> logits;
};

class DecoderCudaLayerBackward {
 public:
  bool available() const { return false; }
  const char* backend_name() const { return "not_implemented"; }
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
                   const TransformerState& initial);

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
  void ensure_tape_capacity(int rows, int vocab, int hidden, int layers);
  void run_device_backward(float loss, int rows, int capture_row,
                           float grad_scale, bool reset_grads);
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
};

}  // namespace lkjai
