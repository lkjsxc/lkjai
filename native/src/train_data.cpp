#include "train_data.hpp"

#include <algorithm>

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
  const std::string needle = "\"" + key + "\"";
  auto pos = line.find(needle);
  if (pos == std::string::npos) return "";
  pos = line.find(':', pos + needle.size());
  pos = line.find('"', pos == std::string::npos ? 0 : pos);
  if (pos == std::string::npos) return "";
  std::string out;
  bool escaped = false;
  for (size_t i = pos + 1; i < line.size(); ++i) {
    char ch = line[i];
    if (escaped) {
      if (ch == 'n') out.push_back('\n');
      else if (ch == 't') out.push_back('\t');
      else if (ch != 'r') out.push_back(ch);
      escaped = false;
    } else if (ch == '\\') {
      escaped = true;
    } else if (ch == '"') {
      break;
    } else {
      out.push_back(ch);
    }
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
