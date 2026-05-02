#include "transformer_state.hpp"

namespace lkjai {

long long transformer_parameter_count(const TransformerState& state) {
  long long total = static_cast<long long>(state.tok_embeddings.w.size() +
                                          state.pos_embeddings.w.size() +
                                          state.final_norm.w.size() +
                                          state.lm_head.w.size());
  for (const auto& layer : state.layers) {
    total += static_cast<long long>(
        layer.attn_norm.w.size() + layer.q_proj.w.size() +
        layer.k_proj.w.size() + layer.v_proj.w.size() + layer.o_proj.w.size() +
        layer.mlp_norm.w.size() + layer.gate_proj.w.size() +
        layer.up_proj.w.size() + layer.down_proj.w.size());
  }
  return total;
}

}  // namespace lkjai
