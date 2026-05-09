#include "artifact.hpp"
#include "cuda_probe.hpp"
#include "env.hpp"
#include "http_server.hpp"
#include "native_server_routes.hpp"
#include "runtime_api.hpp"

int main() {
  auto host = lkjai::env_string("INFERENCE_HOST", "0.0.0.0");
  int port = lkjai::env_int("INFERENCE_PORT", 8081);
  auto root = lkjai::env_string("MODEL_ROOT", "/models");
  auto model = lkjai::env_string("MODEL_NAME", "lkjai-scratch-40m");
  auto artifact = lkjai::load_artifact(root, model);
  auto cuda = lkjai::cuda_status();
  lkjai::RuntimeConfig runtime{
      host, port, lkjai::env_string("DATA_DIR", "/app/data"),
      "local-native-engine", model,
      lkjai::env_string("AGENT_TOOL_PROFILE", "readonly"),
      lkjai::env_string("TOOL_WORKSPACE_DIR", "/app/data/workspace"),
      lkjai::env_string("KJXLKJ_API_URL", "http://127.0.0.1:8080"),
      lkjai::env_string("KJXLKJ_USER", "default"),
      lkjai::env_string("KJXLKJ_BEARER_TOKEN", "")};
  return lkjai::serve_http(host, port, [&](const lkjai::HttpRequest& request) {
    return lkjai::native_server_route(request, artifact, cuda, runtime);
  });
}
