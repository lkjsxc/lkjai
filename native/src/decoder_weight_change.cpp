#include "decoder_cuda_slice_internal.hpp"

#include <algorithm>
#include <cmath>

namespace lkjai {
namespace {

DecoderWeightChangePart part_delta(const std::vector<float>& before,
                                   const std::vector<float>& after) {
  DecoderWeightChangePart out;
  size_t n = std::min(before.size(), after.size());
  for (size_t i = 0; i < n; ++i) {
    double delta = std::fabs(static_cast<double>(before[i]) -
                             static_cast<double>(after[i]));
    if (delta <= 0.0) continue;
    ++out.changed_elements;
    out.max_abs_delta = std::max(out.max_abs_delta, delta);
  }
  out.changed_tensors = out.changed_elements > 0 ? 1 : 0;
  return out;
}

}  // namespace

void decoder_record_partial_weight_change(const std::vector<float>& before_emb,
                                          const std::vector<float>& before_head,
                                          const DenseTrainState& after,
                                          TransformerTrainReport* report) {
  auto& w = report->decoder_weight_change;
  w.embedding = part_delta(before_emb, after.emb);
  w.lm_head = part_delta(before_head, after.head);
  w.non_embedding = {};
  w.decoder_block = {};
  w.changed_tensors = w.embedding.changed_tensors + w.lm_head.changed_tensors;
  report->embedding_weight_changed = w.embedding.changed_tensors > 0;
  report->lm_head_weight_changed = w.lm_head.changed_tensors > 0;
  report->non_embedding_weight_changed = false;
  report->decoder_block_weight_changed = false;
  report->trainable_weight_changed =
      report->embedding_weight_changed || report->lm_head_weight_changed;
}

}  // namespace lkjai
