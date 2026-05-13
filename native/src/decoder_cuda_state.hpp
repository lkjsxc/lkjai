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

struct DecoderCudaLayerTape {
  DeviceTensor norm_input;
  DeviceTensor q_rope;
  DeviceTensor k_rope;
  DeviceTensor v;
  DeviceTensor attention_state;
  DeviceTensor attention_residual;
  DeviceTensor mlp_norm_input;
  DeviceTensor gate;
  DeviceTensor up;
  DeviceTensor swiglu;
  DeviceTensor block_residual;
};

struct DecoderCudaTape {
  DeviceTensor tokens;
  DeviceTensor labels;
  DeviceTensor loss_mask;
  DeviceTensor embeddings;
  std::vector<DecoderCudaLayerTape> layers;
  DeviceTensor final_norm_input;
  DeviceTensor final_norm;
  DeviceTensor logits;
  DeviceTensor loss;
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
