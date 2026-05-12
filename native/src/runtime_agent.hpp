#pragma once

#include <functional>
#include <string>

#include "runtime_api.hpp"

namespace lkjai {

using RuntimeModelCall =
    std::function<NativeHttpResponse(const std::string& request_body)>;

HttpResponse runtime_chat_with_model_callback(const RuntimeConfig& cfg,
                                              const HttpRequest& request,
                                              RuntimeModelCall model_call);

}  // namespace lkjai
