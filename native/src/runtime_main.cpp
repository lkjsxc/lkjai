#include "env.hpp"
#include "http_server.hpp"
#include "runtime_api.hpp"

using lkjai::HttpRequest;

namespace {
lkjai::RuntimeConfig config_from_env() {
  return {
      lkjai::env_string("APP_HOST", "0.0.0.0"),
      lkjai::env_int("APP_PORT", 8080),
      lkjai::env_string("DATA_DIR", "/app/data"),
      lkjai::env_string("MODEL_API_URL",
                         "http://127.0.0.1:8081/v1/chat/completions"),
      lkjai::env_string("MODEL_NAME", "lkjai-scratch-40m"),
  };
}

}  // namespace

int main() {
  auto cfg = config_from_env();
  return lkjai::serve_http(cfg.host, cfg.port, [&](const HttpRequest& request) {
    return lkjai::runtime_route(cfg, request);
  });
}
