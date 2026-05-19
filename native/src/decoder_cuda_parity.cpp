#include "decoder_cuda_state.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "env.hpp"

namespace lkjai {
namespace {

constexpr double kLossTolerance = 0.15;
constexpr double kLogitsMaxTolerance = 0.08;
constexpr double kLogitsMeanTolerance = 0.025;

}  // namespace

bool DecoderCudaState::decoder_parity_sample_this_step() {
  if (parity_calls_ == 0) {
    parity_mode_ = env_string("TRAIN_DECODER_PARITY_MODE", "off");
    parity_interval_ = std::max(1, env_int("TRAIN_DECODER_PARITY_INTERVAL",
                                           128));
    if (env_int("TRAIN_DECODER_PARITY_FIRST_STEPS", 0) > 0 &&
        parity_mode_ == "off") {
      parity_mode_ = "sampled";
    }
  }
  ++parity_calls_;
  int first_steps = env_int("TRAIN_DECODER_PARITY_FIRST_STEPS", 0);
  if (first_steps > 0 && parity_calls_ <= first_steps) return true;
  if (parity_mode_ == "strict") return true;
  if (parity_mode_ == "sampled") return parity_calls_ % parity_interval_ == 0;
  if (parity_mode_ == "final_only") return false;
  if (parity_mode_ != "off") parity_mode_ = "off";
  return false;
}

void DecoderCudaState::record_decoder_parity(double loss,
                                             const std::vector<float>* logits,
                                             const PackedBatch& batch,
                                             bool compare_logits) {
  auto ref = transformer_forward(batch, state_);
  parity_sample_loss_diff_ = std::abs(loss - ref.loss);
  parity_sample_logits_max_diff_ = 0.0;
  parity_sample_logits_mean_diff_ = 0.0;
  bool logits_ok = true;
  if (compare_logits && logits) {
    logits_ok = logits->size() == ref.loss_logits.size();
    for (size_t i = 0; logits_ok && i < logits->size(); ++i) {
      if (!std::isfinite((*logits)[i])) logits_ok = false;
      double diff = std::abs(static_cast<double>((*logits)[i]) -
                             ref.loss_logits[i]);
      parity_sample_logits_max_diff_ =
          std::max(parity_sample_logits_max_diff_, diff);
      parity_sample_logits_mean_diff_ += diff;
    }
    if (logits_ok && !logits->empty()) {
      parity_sample_logits_mean_diff_ /=
          static_cast<double>(logits->size());
    }
    logits_ok = logits_ok &&
        parity_sample_logits_max_diff_ <= kLogitsMaxTolerance &&
        parity_sample_logits_mean_diff_ <= kLogitsMeanTolerance;
  }
  bool loss_ok = std::isfinite(loss) &&
                 parity_sample_loss_diff_ <= kLossTolerance;
  parity_sample_status_ = loss_ok && logits_ok ? "pass" : "fail";
  ++parity_sample_count_;
  if (parity_sample_status_ != "pass") ++parity_failure_count_;
  if (parity_sample_status_ != "pass" && parity_mode_ == "strict") {
    throw std::runtime_error("decoder CUDA strict parity check failed");
  }
}

}  // namespace lkjai
