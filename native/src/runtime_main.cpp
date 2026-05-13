#include "env.hpp"
#include "http_server.hpp"
#include "runtime_api.hpp"

using lkjai::HttpRequest;

namespace {
lkjai::RuntimeConfig config_from_env() {
  return {
      lkjai::env_string("SANDBOX_HOST",
                         lkjai::env_string("APP_HOST", "127.0.0.1")),
      lkjai::env_int("SANDBOX_PORT", lkjai::env_int("APP_PORT", 8082)),
      lkjai::env_string("DATA_DIR", "/app/data"),
      lkjai::env_string("MODEL_API_URL",
                         "http://127.0.0.1:8081/v1/chat/completions"),
      lkjai::env_string("MODEL_NAME", "lkjai-scratch-40m"),
      lkjai::env_string("AGENT_TOOL_PROFILE", "readonly"),
      lkjai::env_string("TOOL_WORKSPACE_DIR", "/app/data/workspace"),
      lkjai::env_string("KJXLKJ_API_URL", "http://127.0.0.1:8080"),
      lkjai::env_string("KJXLKJ_USER", "default"),
      lkjai::env_string("KJXLKJ_BEARER_TOKEN", ""),
  };
}

}  // namespace

int main() {
  auto cfg = config_from_env();
  return lkjai::serve_http(cfg.host, cfg.port, [&](const HttpRequest& request) {
    return lkjai::runtime_route(cfg, request);
  });
}
