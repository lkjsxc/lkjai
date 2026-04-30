#pragma once

#include <functional>
#include <string>

namespace lkjai {

struct HttpRequest {
  std::string method;
  std::string path;
  std::string body;
};

struct HttpResponse {
  int status = 200;
  std::string body;
};

using Handler = std::function<HttpResponse(const HttpRequest&)>;

int serve_http(const std::string& host, int port, const Handler& handler);

}  // namespace lkjai
