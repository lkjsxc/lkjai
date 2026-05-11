#include "packed_cache_digest.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>

namespace lkjai {
namespace {

void mix_byte(uint64_t* hash, unsigned char value) {
  *hash = (*hash ^ value) * 1099511628211ull;
}

std::string hex(uint64_t hash) {
  std::ostringstream out;
  out << std::hex << hash;
  return out.str();
}

}  // namespace

std::string packed_file_digest(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  uint64_t hash = 1469598103934665603ull;
  char ch = 0;
  while (in.get(ch)) mix_byte(&hash, static_cast<unsigned char>(ch));
  return hex(hash);
}

std::string packed_source_digest(const std::filesystem::path& path) {
  if (std::filesystem::is_regular_file(path)) return packed_file_digest(path);
  uint64_t hash = 1469598103934665603ull;
  if (!std::filesystem::is_directory(path)) return hex(hash);
  std::vector<std::filesystem::path> shards;
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    if (entry.is_regular_file() && entry.path().extension() == ".jsonl") {
      shards.push_back(entry.path());
    }
  }
  std::sort(shards.begin(), shards.end());
  for (const auto& shard : shards) {
    auto name = shard.filename().string();
    for (char ch : name) mix_byte(&hash, static_cast<unsigned char>(ch));
    auto digest = packed_file_digest(shard);
    for (char ch : digest) mix_byte(&hash, static_cast<unsigned char>(ch));
    auto size = std::filesystem::file_size(shard);
    for (int i = 0; i < 8; ++i) {
      mix_byte(&hash, static_cast<unsigned char>((size >> (i * 8)) & 0xffu));
    }
  }
  return hex(hash);
}

std::string packed_payload_digest(const std::filesystem::path& dir) {
  uint64_t hash = 1469598103934665603ull;
  for (const auto& name : {"tokens.bin", "loss_mask.bin", "starts.bin"}) {
    for (char ch : std::string(name)) mix_byte(&hash, static_cast<unsigned char>(ch));
    auto digest = packed_file_digest(dir / name);
    for (char ch : digest) mix_byte(&hash, static_cast<unsigned char>(ch));
    if (!std::filesystem::is_regular_file(dir / name)) continue;
    auto size = std::filesystem::file_size(dir / name);
    for (int i = 0; i < 8; ++i) {
      mix_byte(&hash, static_cast<unsigned char>((size >> (i * 8)) & 0xffu));
    }
  }
  return hex(hash);
}

}  // namespace lkjai
