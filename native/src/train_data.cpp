#include "train_data.hpp"

#include <algorithm>

#include "json_min.hpp"

namespace lkjai {

std::vector<std::filesystem::path> collect_jsonl(const std::filesystem::path& root) {
  std::vector<std::filesystem::path> files;
  if (!std::filesystem::exists(root)) return files;
  for (const auto& item : std::filesystem::recursive_directory_iterator(root)) {
    if (item.is_regular_file() && item.path().extension() == ".jsonl") {
      files.push_back(item.path());
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

std::string extract_json_string(const std::string& line, const std::string& key) {
  return json_first_string(line, key);
}

std::string training_text_from_jsonl(const std::string& line) {
  auto text = json_string_values(line, "text");
  auto content = json_string_values(line, "content");
  text.insert(text.end(), content.begin(), content.end());
  if (text.empty()) return line;
  std::string out;
  for (const auto& item : text) {
    if (!out.empty()) out += "\n";
    out += item;
  }
  return out;
}

CorpusCursor::CorpusCursor(std::vector<std::filesystem::path> files)
    : files_(std::move(files)) {}

bool CorpusCursor::next(std::string* out) {
  while (!files_.empty()) {
    if (!stream_.is_open()) stream_.open(files_[index_]);
    std::string line;
    if (std::getline(stream_, line)) {
      ++rows_;
      *out = line;
      return true;
    }
    stream_.close();
    index_ = (index_ + 1) % files_.size();
  }
  return false;
}

long long CorpusCursor::rows() const { return rows_; }

int CorpusCursor::file_count() const {
  return static_cast<int>(files_.size());
}

}  // namespace lkjai
