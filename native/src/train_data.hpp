#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace lkjai {

std::vector<std::filesystem::path> collect_jsonl(const std::filesystem::path& root);
std::string extract_json_string(const std::string& line, const std::string& key);
std::string training_text_from_jsonl(const std::string& line);

class CorpusCursor {
 public:
  explicit CorpusCursor(std::vector<std::filesystem::path> files);
  bool next(std::string* out);
  long long rows() const;
  int file_count() const;

 private:
  std::vector<std::filesystem::path> files_;
  std::ifstream stream_;
  size_t index_ = 0;
  long long rows_ = 0;
};

}  // namespace lkjai
