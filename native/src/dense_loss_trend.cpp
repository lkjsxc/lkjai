#include "dense_loss_trend.hpp"

#include <algorithm>
#include <cmath>

namespace lkjai {
namespace {

double sample_mean(const std::vector<DenseLossSample>& samples, size_t begin,
                   size_t end) {
  if (begin >= end || begin >= samples.size()) return 0.0;
  end = std::min(end, samples.size());
  double sum = 0.0;
  for (size_t i = begin; i < end; ++i) sum += samples[i].loss;
  return sum / static_cast<double>(end - begin);
}

bool finite_samples(const std::vector<DenseLossSample>& samples) {
  for (const auto& sample : samples) {
    if (!std::isfinite(sample.loss)) return false;
  }
  return true;
}

}  // namespace

void finalize_dense_loss_trend(DenseTrainReport* report) {
  report->loss_delta = report->initial_loss - report->loss;
  report->loss_decrease_fraction =
      report->initial_loss > 0.0 ? report->loss_delta / report->initial_loss
                                 : 0.0;
  size_t quarter = std::max<size_t>(1, report->loss_samples.size() / 4);
  report->first_quarter_loss_mean =
      sample_mean(report->loss_samples, 0, quarter);
  report->last_quarter_loss_mean = sample_mean(
      report->loss_samples, report->loss_samples.size() > quarter
                                ? report->loss_samples.size() - quarter
                                : 0,
      report->loss_samples.size());
  bool finite = std::isfinite(report->initial_loss) &&
                std::isfinite(report->loss) &&
                finite_samples(report->loss_samples);
  if (!finite) {
    report->learning_status = "non_finite";
  } else if (report->loss_decrease_fraction >= 0.10 &&
             report->last_quarter_loss_mean <
                 report->first_quarter_loss_mean &&
             report->weight_changed) {
    report->learning_status = "learning";
  } else if (report->loss < report->initial_loss) {
    report->learning_status = "decreasing";
  } else {
    report->learning_status = "not_learning";
  }
}

}  // namespace lkjai
