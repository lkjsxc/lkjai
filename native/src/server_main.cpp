#include "artifact.hpp"
#include "cuda_probe.hpp"
#include "env.hpp"
#include "http_server.hpp"
#include "native_server_routes.hpp"

int main() {
  auto host = lkjai::env_string("INFERENCE_HOST", "0.0.0.0");
  int port = lkjai::env_int("INFERENCE_PORT", 8081);
  auto root = lkjai::env_string("MODEL_ROOT", "/models");
  auto model = lkjai::env_string("MODEL_NAME", "lkjai-scratch-40m");
  auto artifact = lkjai::load_artifact(root, model);
  auto cuda = lkjai::cuda_status();
  lkjai::RuntimeConfig runtime{host, port, "", "", model};
  return lkjai::serve_http(host, port, [&](const lkjai::HttpRequest& request) {
    return lkjai::native_server_route(request, artifact, cuda, runtime);
  });
}
