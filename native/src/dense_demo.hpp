#pragma once

#include "artifact.hpp"
#include "http_server.hpp"

namespace lkjai {

HttpResponse dense_demo_status_response(const ArtifactStatus& artifact);
HttpResponse dense_demo_next_token_response(const ArtifactStatus& artifact,
                                            const HttpRequest& request);

}  // namespace lkjai
