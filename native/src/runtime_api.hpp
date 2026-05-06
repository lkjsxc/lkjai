#pragma once

#include <string>
#include <vector>

#include "http_server.hpp"
#include "native_http_client.hpp"

namespace lkjai {

struct RuntimeConfig {
  std::string host;
  int port = 8080;
  std::string data_dir;
  std::string model_url;
  std::string model;
  std::string tool_profile = "readonly";
  std::string workspace_dir;
  std::string kjxlkj_api_url = "http://127.0.0.1:8080";
  std::string kjxlkj_user = "default";
  std::string kjxlkj_bearer_token;
};

HttpResponse runtime_route(const RuntimeConfig& cfg, const HttpRequest& request);
HttpResponse runtime_chat_with_model_response(const RuntimeConfig& cfg,
                                              const HttpRequest& request,
                                              const NativeHttpResponse& model);
std::string runtime_model_status_json(const RuntimeConfig& cfg,
                                      const NativeHttpResponse& probe);
std::string runtime_config_status_json(const RuntimeConfig& cfg);
std::string runtime_health_json(const RuntimeConfig& cfg);
std::vector<std::string> runtime_visible_event_kinds(std::string_view body);

}  // namespace lkjai
