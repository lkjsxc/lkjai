#pragma once

#include <string>

namespace lkjai {

struct NativeHttpResponse {
  int status = 0;
  std::string body;
  std::string error;
};

NativeHttpResponse native_http_get(const std::string& url);
NativeHttpResponse native_http_post_json(const std::string& url,
                                         const std::string& body);
std::string model_url_to_models_url(const std::string& chat_url);

}  // namespace lkjai
