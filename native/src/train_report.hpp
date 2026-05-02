#pragma once

#include <string>

#include "cuda_probe.hpp"
#include "dense_train.hpp"
#include "transformer_train.hpp"

namespace lkjai {

std::string dense_train_report_json(const DenseTrainReport& report,
                                    const CudaStatus& cuda,
                                    const std::string& trainer_mode,
                                    const std::string& status,
                                    const std::string& failure_reason);
bool write_dense_train_report(const DenseTrainReport& report,
                              const CudaStatus& cuda,
                              const std::string& trainer_mode,
                              const std::string& status,
                              const std::string& failure_reason,
                              std::string* error);
std::string transformer_train_report_json(const TransformerTrainReport& report,
                                          const CudaStatus& cuda,
                                          const std::string& trainer_mode,
                                          const std::string& status,
                                          const std::string& failure_reason);
bool write_transformer_train_report(const TransformerTrainReport& report,
                                    const CudaStatus& cuda,
                                    const std::string& trainer_mode,
                                    const std::string& status,
                                    const std::string& failure_reason,
                                    std::string* error);

}  // namespace lkjai
