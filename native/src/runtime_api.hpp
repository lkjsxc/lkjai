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
};

HttpResponse runtime_route(const RuntimeConfig& cfg, const HttpRequest& request);
HttpResponse runtime_chat_with_model_response(const RuntimeConfig& cfg,
                                              const HttpRequest& request,
                                              const NativeHttpResponse& model);
std::string runtime_model_status_json(const RuntimeConfig& cfg,
                                      const NativeHttpResponse& probe);
std::vector<std::string> runtime_visible_event_kinds(std::string_view body);

}  // namespace lkjai
