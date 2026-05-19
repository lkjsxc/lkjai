#include "repo_check.hpp"

#include <fstream>
#include <iterator>
#include <string>

#include "json_min.hpp"

namespace lkjai {
namespace {

std::string read_file(const std::filesystem::path& path) {
  std::ifstream file(path);
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void require_text(const std::filesystem::path& path, const std::string& text,
                  RepoCheckResult* result) {
  if (read_file(path).find(text) == std::string::npos) {
    result->fail(path.string() + " missing documented default " + text);
  }
}

}  // namespace

void check_decoder_doc_defaults(const std::filesystem::path& repo,
                                RepoCheckResult* result) {
  auto train = read_file(repo / "configs/training/decoder_2h_40m_3070.json");
  auto doc = repo / "docs/operations/training/runbooks/long-run.md";
  require_text(doc, "sequence length `" +
                        std::to_string(json_int_value(train, "sequence_len", 0)) +
                        "`", result);
  require_text(doc, "batch size `" +
                        std::to_string(json_int_value(train, "batch_size", 0)) +
                        "`", result);
  require_text(doc, "gradient accumulation `" +
                        std::to_string(json_int_value(train, "gradient_accumulation", 0)) +
                        "`", result);
  require_text(doc, "learning rate `0.0003`", result);
  require_text(doc, "warmup `" +
                        std::to_string(json_int_value(train, "warmup_steps", 0)) +
                        "`", result);
  require_text(doc, "optimizer-step cap `" +
                        std::to_string(json_int_value(train, "max_optimizer_steps", 0)) +
                        "`", result);
  require_text(doc, "wall-clock target `" +
                        std::to_string(json_int_value(train, "target_seconds", 0)) +
                        "`", result);
  require_text(doc, "latest-checkpoint cadence `" +
                        std::to_string(json_int_value(
                            train, "save_latest_every_optimizer_steps", 0)) +
                        "`", result);
}

}  // namespace lkjai
