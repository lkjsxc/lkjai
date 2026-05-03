#include "dense_weight_change.hpp"

#include <algorithm>
#include <cmath>
#include <ostream>

#include "json_min.hpp"

namespace lkjai {
namespace {

DenseWeightTensorDelta tensor_delta(const std::string& name,
                                    const std::vector<float>& before,
                                    const std::vector<float>& after) {
  DenseWeightTensorDelta delta;
  delta.name = name;
  if (before.size() != after.size()) {
    delta.status = "size_mismatch";
    return delta;
  }
  delta.checked_elements = static_cast<uint64_t>(before.size());
  double sum_abs = 0.0;
  for (size_t i = 0; i < before.size(); ++i) {
    double abs_delta = std::fabs(static_cast<double>(after[i] - before[i]));
    if (abs_delta > 0.0) ++delta.changed_elements;
    delta.max_abs_delta = std::max(delta.max_abs_delta, abs_delta);
    sum_abs += abs_delta;
  }
  if (delta.checked_elements > 0) {
    delta.mean_abs_delta = sum_abs / static_cast<double>(delta.checked_elements);
  }
  delta.status = delta.changed_elements > 0 ? "pass" : "fail";
  return delta;
}

void append_tensor_json(std::ostream& out,
                        const DenseWeightTensorDelta& tensor) {
  out << "{\"name\":\"" << json_escape(tensor.name) << "\""
      << ",\"checked_elements\":" << tensor.checked_elements
      << ",\"changed_elements\":" << tensor.changed_elements
      << ",\"max_abs_delta\":" << tensor.max_abs_delta
      << ",\"mean_abs_delta\":" << tensor.mean_abs_delta
      << ",\"status\":\"" << json_escape(tensor.status) << "\"}";
}

}  // namespace

DenseWeightChangeReport dense_weight_change_report(
    const DenseTrainState& before, const DenseTrainState& after) {
  DenseWeightChangeReport report;
  report.tolerance = 0.0;
  report.tensors.push_back(tensor_delta("tok_embeddings", before.emb, after.emb));
  report.tensors.push_back(tensor_delta("lm_head", before.head, after.head));
  double sum_abs = 0.0;
  for (const auto& tensor : report.tensors) {
    ++report.checked_tensors;
    report.checked_elements += tensor.checked_elements;
    report.changed_elements += tensor.changed_elements;
    report.max_abs_delta = std::max(report.max_abs_delta, tensor.max_abs_delta);
    report.changed_tensors += tensor.changed_elements > 0 ? 1 : 0;
    sum_abs += tensor.mean_abs_delta * static_cast<double>(tensor.checked_elements);
  }
  if (report.checked_elements > 0) {
    report.mean_abs_delta = sum_abs / static_cast<double>(report.checked_elements);
  }
  bool all_changed = report.checked_tensors == 2 && report.changed_tensors == 2;
  report.status = all_changed ? "pass" : "fail";
  report.reason = all_changed
                      ? "all trainable FP32 master tensors changed"
                      : "one or more trainable FP32 master tensors did not change";
  return report;
}

void append_dense_weight_change_json(std::ostream& out,
                                     const DenseWeightChangeReport& report) {
  out << "{\"checked_tensors\":" << report.checked_tensors
      << ",\"changed_tensors\":" << report.changed_tensors
      << ",\"checked_elements\":" << report.checked_elements
      << ",\"changed_elements\":" << report.changed_elements
      << ",\"max_abs_delta\":" << report.max_abs_delta
      << ",\"mean_abs_delta\":" << report.mean_abs_delta
      << ",\"tolerance\":" << report.tolerance << ",\"tensors\":[";
  for (size_t i = 0; i < report.tensors.size(); ++i) {
    if (i) out << ",";
    append_tensor_json(out, report.tensors[i]);
  }
  out << "],\"status\":\"" << json_escape(report.status) << "\""
      << ",\"reason\":\"" << json_escape(report.reason) << "\"}";
}

}  // namespace lkjai
