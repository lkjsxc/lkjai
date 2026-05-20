#include <filesystem>
#include <fstream>
#include <iostream>

#include "train_report_digest.hpp"
#include "transformer_report_acceptance.hpp"

namespace {
bool expect(bool ok, const char* message) {
  if (ok) return true;
  std::cerr << message << "\n";
  return false;
}
void write(const std::filesystem::path& path, const std::string& body) {
  std::ofstream(path) << body;
}
std::string transcript(const std::string& digest, int steady) {
  return "{\"route\":\"/v1/chat/completions\",\"request\":{\"model\":\"m\"},"
         "\"response_status\":200,\"choices_present\":true,"
         "\"decode_backend\":\"cuda_kv_cache\","
         "\"kv_cache_backend\":\"cuda_contiguous_bf16\","
         "\"kv_cache_prefill_allocated_bytes\":4096,"
         "\"kv_cache_steady_state_token_allocations\":" +
         std::to_string(steady) + ",\"train_report_digest\":\"" + digest +
         "\",\"artifact_manifest_digest\":\"artifact\","
         "\"created_at\":\"2026-05-20T00:00:00Z\"}";
}
}  // namespace

int main() {
  auto root = std::filesystem::temp_directory_path() / "lkjai-route-transcript";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  auto report = root / "train-report.json";
  auto route = root / "decoder-40m-3070-route-transcript.json";
  write(report, "{\"status\":\"success\"}\n");
  std::string error;
  auto digest = lkjai::train_report_file_digest(report);
  write(route, transcript(digest, 0));
  bool ok = expect(lkjai::transformer_route_transcript_accepted(route, report,
                                                               &error),
                   error.c_str());
  write(route, transcript(digest, 1));
  ok = ok && expect(!lkjai::transformer_route_transcript_accepted(route, report,
                                                                 &error),
                    "steady allocation rejected");
  write(route, transcript("wrong", 0));
  ok = ok && expect(!lkjai::transformer_route_transcript_accepted(route, report,
                                                                 &error),
                    "digest mismatch rejected");
  return ok ? 0 : 1;
}
