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

void merge_part(DecoderWeightChangePart* total,
                const DecoderWeightChangePart& part) {
  total->max_abs_delta = std::max(total->max_abs_delta, part.max_abs_delta);
  total->changed_elements += part.changed_elements;
  total->changed_tensors += part.changed_tensors;
}

template <typename Fn>
void each_decoder_block_param(const TransformerState& s, Fn fn) {
  for (const auto& l : s.layers) {
    fn(l.attn_norm);
    fn(l.q_proj);
    fn(l.k_proj);
    fn(l.v_proj);
    fn(l.o_proj);
    fn(l.mlp_norm);
    fn(l.gate_proj);
    fn(l.up_proj);
    fn(l.down_proj);
  }
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

void decoder_record_full_weight_change(const TransformerState& before,
                                       const TransformerState& after,
                                       TransformerTrainReport* report) {
  auto& w = report->decoder_weight_change;
  w = {};
  w.embedding = part_delta(before.tok_embeddings.w, after.tok_embeddings.w);
  w.lm_head = part_delta(before.lm_head.w, after.lm_head.w);
  size_t i = 0;
  each_decoder_block_param(before, [&](const Parameter& before_param) {
    const auto& after_layer = after.layers[i / 9];
    const Parameter* after_param = nullptr;
    switch (i % 9) {
      case 0:
        after_param = &after_layer.attn_norm;
        break;
      case 1:
        after_param = &after_layer.q_proj;
        break;
      case 2:
        after_param = &after_layer.k_proj;
        break;
      case 3:
        after_param = &after_layer.v_proj;
        break;
      case 4:
        after_param = &after_layer.o_proj;
        break;
      case 5:
        after_param = &after_layer.mlp_norm;
        break;
      case 6:
        after_param = &after_layer.gate_proj;
        break;
      case 7:
        after_param = &after_layer.up_proj;
        break;
      default:
        after_param = &after_layer.down_proj;
        break;
    }
    auto part = part_delta(before_param.w, after_param->w);
    merge_part(&w.decoder_block, part);
    merge_part(&w.non_embedding, part);
    ++i;
  });
  merge_part(&w.non_embedding,
             part_delta(before.final_norm.w, after.final_norm.w));
  w.changed_tensors = w.embedding.changed_tensors + w.lm_head.changed_tensors +
                      w.non_embedding.changed_tensors;
  report->embedding_weight_changed = w.embedding.changed_tensors > 0;
  report->lm_head_weight_changed = w.lm_head.changed_tensors > 0;
  report->non_embedding_weight_changed = w.non_embedding.changed_tensors > 0;
  report->decoder_block_weight_changed = w.decoder_block.changed_tensors > 0;
  report->trainable_weight_changed =
      report->embedding_weight_changed || report->lm_head_weight_changed ||
      report->non_embedding_weight_changed;
}

}  // namespace lkjai
