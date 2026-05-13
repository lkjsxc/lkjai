#include <iostream>
#include <string>

#include "native_server_routes.hpp"

namespace {

bool has(const std::string& text, const std::string& needle) {
  return text.find(needle) != std::string::npos;
}

bool expect(bool ok, const std::string& message) {
  if (ok) return true;
  std::cerr << message << "\n";
  return false;
}

bool dense_api_rejection_contract() {
  lkjai::ArtifactStatus artifact;
  artifact.loaded = true;
  artifact.model_name = "dense-real";
  lkjai::CudaStatus cuda;
  lkjai::RuntimeConfig cfg{"127.0.0.1", 8081, "", "", artifact.model_name};
  auto status = lkjai::native_server_route({"GET", "/api/dense/status", ""},
                                           artifact, cuda, cfg);
  auto next = lkjai::native_server_route(
      {"POST", "/api/dense/next-token", "{\"tokens\":[1,2,3],\"top_k\":3}"},
      artifact, cuda, cfg);
  return expect(status.status == 404, "dense status rejected") &&
         expect(next.status == 404, "dense next rejected") &&
         expect(has(status.body, "not found"), "reject body");
}

}  // namespace

int main() { return dense_api_rejection_contract() ? 0 : 1; }
