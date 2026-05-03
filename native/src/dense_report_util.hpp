#pragma once

#include <filesystem>
#include <sstream>
#include <string>

#include "dense_train.hpp"

namespace lkjai {

std::string train_report_file_digest(const std::filesystem::path& path);
std::string train_report_packed_cache_digest(const std::filesystem::path& dir);
std::string train_report_manifest_checksum(const std::filesystem::path& dir);
long long dense_report_parameter_count(const DenseTrainReport& report);
void append_dense_loss_samples(std::ostringstream* out,
                               const std::vector<DenseLossSample>& samples);
void append_dense_tuning_fields(std::ostringstream* out,
                                const DenseTrainReport& report);

}  // namespace lkjai
