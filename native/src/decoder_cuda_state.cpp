#include "decoder_cuda_state.hpp"

namespace lkjai {

DecoderCudaState::DecoderCudaState(const TransformerConfig& cfg,
                                   const TransformerState& initial)
    : state_(initial),
      ctx_(),
      dense_host_(decoder_dense_state(decoder_dense_cfg(cfg), initial)),
      dense_cuda_(dense_host_.cfg, dense_host_, &ctx_) {}

TransformerState DecoderCudaState::copy_to_host() {
  auto dense = dense_cuda_.copy_to_host();
  decoder_copy_dense_back(dense, &state_);
  return state_;
}

void DecoderCudaState::fill_report(TransformerTrainReport* report) {
  decoder_fill_cuda_slice_report(dense_cuda_, report);
}

void DecoderCudaState::record_weight_change(const TransformerState& before,
                                            TransformerTrainReport* report) {
  auto dense = dense_cuda_.copy_to_host();
  decoder_record_partial_weight_change(before.tok_embeddings.w,
                                       before.lm_head.w, dense, report);
}

}  // namespace lkjai
