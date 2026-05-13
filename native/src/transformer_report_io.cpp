#include "train_report.hpp"

#include <filesystem>
#include <fstream>

#include "transformer_report_acceptance.hpp"

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
  auto body = transformer_train_report_json(report, cuda, trainer_mode, status,
                                            failure_reason);
  out << body << "\n";
  if (transformer_report_shape_accepted_decoder(report)) {
    for (const auto& dir : {report.checkpoint_dir, report.export_dir,
                            report.served_dir}) {
      std::ofstream copy(dir / "decoder_train_report.json");
      if (copy) copy << body << "\n";
    }
  }
  return true;
}

}  // namespace lkjai
