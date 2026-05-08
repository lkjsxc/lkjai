#pragma once

#include "dense_cuda_internal.hpp"
#include "transformer_state.hpp"

namespace lkjai {

struct DecoderCudaForwardSubstrateReport;

DenseConfig decoder_dense_cfg(const TransformerConfig& cfg);
DenseTrainState decoder_dense_state(const DenseConfig& cfg,
                                    const TransformerState& source);
void decoder_copy_dense_back(const DenseTrainState& dense,
                             TransformerState* state);
bool decoder_validate_layer_shapes(const TransformerConfig& cfg,
                                   std::string* error);
std::string decoder_shape_report(const TransformerConfig& cfg);
bool decoder_write_all(const TransformerTrainOptions& opt,
                       const TransformerState& state,
                       TransformerTrainReport* report, int seq_len);
bool decoder_cuda_slice_run_block_forward(
    const TransformerState& state, const PackedBatch& batch,
    DecoderCudaForwardSubstrateReport* report, std::string* error);
void decoder_fill_cuda_slice_report(DenseCudaState& cuda,
                                    TransformerTrainReport* report);

}  // namespace lkjai
