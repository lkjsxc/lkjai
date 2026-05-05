#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "packed_cache.hpp"

namespace {

void write_bytes(const std::filesystem::path& path, const char* data,
                 size_t bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(data, static_cast<std::streamsize>(bytes));
}

void write_u16(const std::filesystem::path& path, std::vector<uint16_t> values) {
  write_bytes(path, reinterpret_cast<const char*>(values.data()),
              values.size() * sizeof(uint16_t));
}

void write_u64(const std::filesystem::path& path, std::vector<uint64_t> values) {
  write_bytes(path, reinterpret_cast<const char*>(values.data()),
              values.size() * sizeof(uint64_t));
}

std::filesystem::path make_cache(const std::string& name,
                                 std::vector<uint64_t> starts) {
  auto dir = std::filesystem::temp_directory_path() / name;
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  std::ofstream(dir / "metadata.json")
      << "{\"format\":\"lkjai-packed-cache-v2\",\"sequence_len\":4,"
      << "\"vocab_size\":16,\"smoke_fixture\":true,"
      << "\"token_dtype\":\"uint16\",\"token_count\":8,"
      << "\"row_count\":" << starts.size() << "}\n";
  write_u16(dir / "tokens.bin", {0, 1, 2, 3, 4, 5, 6, 7});
  write_bytes(dir / "loss_mask.bin", "\1\1\1\1\1\1\1\1", 8);
  write_u64(dir / "starts.bin", starts);
  return dir;
}

bool expect(bool condition, const std::string& label) {
  if (condition) return true;
  std::cerr << "failed: " << label << "\n";
  return false;
}

}  // namespace

int main() {
  std::string error;
  auto ok_dir = make_cache("lkjai-reader-ok", {0, 4});
  lkjai::PackedCacheReader reader;
  if (!expect(reader.open(ok_dir, 4, 16, &error), error)) return 1;
  lkjai::PackedBatch batch;
  if (!expect(reader.load_batch(1, 3, &batch, &error), error)) return 1;
  if (!expect(batch.tokens.size() == 12, "batch token count")) return 1;
  if (!expect(batch.tokens[0] == 4 && batch.tokens[4] == 0 &&
                  batch.tokens[8] == 4,
              "wraparound read")) return 1;
  lkjai::PackedCacheReader mismatch;
  if (!expect(!mismatch.open(ok_dir, 3, 16, &error), "seq mismatch")) return 1;
  lkjai::PackedCacheReader vocab;
  if (!expect(!vocab.open(ok_dir, 4, 8, &error), "vocab mismatch")) return 1;
  auto bad_dir = make_cache("lkjai-reader-bad-start", {5});
  lkjai::PackedCacheReader bad;
  if (!expect(!bad.open(bad_dir, 4, 16, &error), "bad start rejected")) {
    return 1;
  }
  std::filesystem::resize_file(ok_dir / "tokens.bin", 6);
  lkjai::PackedCacheReader truncated;
  if (!expect(!truncated.open(ok_dir, 4, 16, &error), "truncated rejected")) {
    return 1;
  }
  return 0;
}
