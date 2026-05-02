#include "train_data.hpp"

#include <algorithm>

namespace lkjai {
namespace {

std::vector<std::string> extract_json_strings(const std::string& line,
                                              const std::string& key);

}  // namespace

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
  auto values = extract_json_strings(line, key);
  return values.empty() ? "" : values.front();
}

std::string training_text_from_jsonl(const std::string& line) {
  auto text = extract_json_strings(line, "text");
  auto content = extract_json_strings(line, "content");
  text.insert(text.end(), content.begin(), content.end());
  if (text.empty()) return line;
  std::string out;
  for (const auto& item : text) {
    if (!out.empty()) out += "\n";
    out += item;
  }
  return out;
}

namespace {

std::vector<std::string> extract_json_strings(const std::string& line,
                                              const std::string& key) {
  std::vector<std::string> values;
  const std::string needle = "\"" + key + "\"";
  size_t search = 0;
  while (true) {
    auto pos = line.find(needle, search);
    if (pos == std::string::npos) break;
    pos = line.find(':', pos + needle.size());
    pos = line.find('"', pos == std::string::npos ? 0 : pos);
    if (pos == std::string::npos) break;
    std::string out;
    bool escaped = false;
    for (size_t i = pos + 1; i < line.size(); ++i) {
      char ch = line[i];
      search = i + 1;
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
    values.push_back(out);
  }
  return values;
}

}  // namespace

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
