#pragma once

#include <filesystem>
#include <sstream>
#include <string>

#include "dense_train.hpp"
#include "train_report_digest.hpp"

namespace lkjai {

long long dense_report_parameter_count(const DenseTrainReport& report);
void append_dense_loss_samples(std::ostringstream* out,
                               const std::vector<DenseLossSample>& samples);
void append_dense_tuning_fields(std::ostringstream* out,
                                const DenseTrainReport& report);
void append_dense_run_control_fields(std::ostringstream* out,
                                     const DenseTrainReport& report);

}  // namespace lkjai
