#include "train_report.hpp"

#include <filesystem>
#include <fstream>

namespace lkjai {

bool write_transformer_train_report(const TransformerTrainReport& report,
                                    const CudaStatus& cuda,
                                    const std::string& trainer_mode,
                                    const std::string& status,
                                    const std::string& failure_reason,
                                    std::string* error) {
  auto path = report.checkpoint_dir.parent_path().parent_path() / "runs" /
              "train-report.json";
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  if (!out) {
    *error = "failed to write train report: " + path.string();
    return false;
  }
  out << transformer_train_report_json(report, cuda, trainer_mode, status,
                                       failure_reason) << "\n";
  return true;
}

}  // namespace lkjai
