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
  TransformerState state_;
  CudaExecutionContext ctx_;
  DenseTrainState dense_host_;
  DenseCudaState dense_cuda_;
};

}  // namespace lkjai
