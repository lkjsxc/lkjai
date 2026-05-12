#pragma once

#include <string>
#include <vector>

#include "decoder_cuda_slice_internal.hpp"
#include "dense_cuda_internal.hpp"
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
    DeviceTensor weight;
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
  void optimizer_step(float lr, int step);
  TransformerState copy_to_host();
  void fill_report(TransformerTrainReport* report);
  void record_weight_change(const TransformerState& before,
                            TransformerTrainReport* report);
  DenseCudaState& dense_cuda() { return dense_cuda_; }

 private:
  void build_registry();
  void copy_registry_to_host();
  void sync_registry_from_host();
  void scale_and_accumulate_grads(const TransformerState& previous,
                                  float grad_scale, bool reset_grads);

  TransformerState state_;
  CudaExecutionContext ctx_;
  DenseTrainState dense_host_;
  DenseCudaState dense_cuda_;
  std::vector<RegistryTensor> registry_;
  uint64_t registry_shadow_bytes_ = 0;
};

}  // namespace lkjai
