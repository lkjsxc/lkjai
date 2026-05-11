#include "native_tokenizer_build.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "json_min.hpp"
#include "native_tokenizer_internal.hpp"
#include "packed_cache_digest.hpp"

namespace lkjai {
namespace {

constexpr const char* kCanonicalTags[] = {
    "<pad>",     "<unk>",       "<bos>",       "<eos>",
    "<assistant_action>",       "<dialogue>",  "</dialogue>",
    "<message>", "</message>",  "<role>",      "</role>",
    "<tool_name>", "</tool_name>", "<content>", "</content>",
    "<action>",  "</action>",
};

std::vector<std::string> byte_vocab() {
  std::vector<std::string> out;
  out.reserve(256);
  for (int byte = 0; byte < 256; ++byte) {
    std::string raw(1, static_cast<char>(byte));
    auto pieces = tokenizer_byte_pieces(raw);
    out.push_back(pieces.empty() ? "" : pieces.front());
  }
  return out;
}

}  // namespace

bool build_native_tokenizer_json(const std::filesystem::path& out,
                                 int max_vocab_size,
                                 NativeTokenizerBuildResult* result,
                                 std::string* error) {
  auto vocab = byte_vocab();
  int next_id = static_cast<int>(vocab.size());
  int total_vocab = next_id + static_cast<int>(std::size(kCanonicalTags));
  if (max_vocab_size <= 0 || total_vocab > max_vocab_size) {
    *error = "native tokenizer vocab exceeds requested max_vocab_size";
    return false;
  }
  if (out.empty()) {
    *error = "tokenizer build requires --out";
    return false;
  }
  std::filesystem::create_directories(out.parent_path());
  std::ofstream file(out);
  if (!file) {
    *error = "failed to open tokenizer output: " + out.string();
    return false;
  }

  file << "{\n";
  file << "  \"version\":\"1.0\",\n";
  file << "  \"truncation\":null,\n";
  file << "  \"padding\":null,\n";
  file << "  \"model\":{\"type\":\"BPE\",\"unk_token\":\"<unk>\",\"vocab\":{";
  for (size_t i = 0; i < vocab.size(); ++i) {
    if (i > 0) file << ",";
    file << "\n    \"" << json_escape(vocab[i]) << "\":" << i;
  }
  file << "\n  },\"merges\":[]},\n";
  file << "  \"pre_tokenizer\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false},\n";
  file << "  \"decoder\":{\"type\":\"ByteLevel\"},\n";
  file << "  \"added_tokens\":[";
  for (size_t i = 0; i < std::size(kCanonicalTags); ++i) {
    if (i > 0) file << ",";
    file << "\n    {\"id\":" << (next_id + static_cast<int>(i))
         << ",\"content\":\"" << json_escape(kCanonicalTags[i])
         << "\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,"
            "\"normalized\":false,\"special\":true}";
  }
  file << "\n  ],\n";
  file << "  \"lkjai\":{\"kind\":\"native_bytelevel_bpe\","
       << "\"canonical_tags\":\"atomic\","
       << "\"vocab_size\":" << total_vocab << ","
       << "\"max_vocab_size\":" << max_vocab_size << "}\n";
  file << "}\n";
  file.close();

  if (result) {
    result->vocab_size = total_vocab;
    result->digest = packed_file_digest(out);
  }
  return true;
}

}  // namespace lkjai
