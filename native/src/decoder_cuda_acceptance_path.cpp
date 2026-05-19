#include "decoder_cuda_slice_internal.hpp"

namespace lkjai {

bool decoder_acceptance_path_required(const TransformerTrainOptions& opt,
                                      const TransformerConfig& cfg,
                                      int seq_len) {
  return opt.run_purpose == "accepted_training" &&
         opt.config_path.filename() == "decoder_40m_bf16_3070.json" &&
         opt.train_config_path.filename() == "decoder_2h_40m_3070.json" &&
         opt.target_seconds >= 7200 && seq_len == 1024 &&
         cfg.context == 1024 && cfg.layers == 10 && cfg.hidden_size == 576 &&
         cfg.heads == 8 && cfg.kv_heads == 2 && cfg.head_dim == 72 &&
         cfg.ffn_size == 1536;
}

}  // namespace lkjai
