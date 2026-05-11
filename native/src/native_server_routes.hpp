#pragma once

#include "artifact.hpp"
#include "cuda_probe.hpp"
#include "dense_demo.hpp"
#include "http_server.hpp"
#include "runtime_api.hpp"

namespace lkjai {

HttpResponse native_server_route(const HttpRequest& request,
                                 const ArtifactStatus& artifact,
                                 const CudaStatus& cuda,
                                 const RuntimeConfig& runtime);
HttpResponse native_server_route(const HttpRequest& request,
                                 const ArtifactStatus& artifact,
                                 const CudaStatus& cuda,
                                 const RuntimeConfig& runtime,
                                 const DenseDemoRuntime& dense);

}  // namespace lkjai
