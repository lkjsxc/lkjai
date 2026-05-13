#include "native_http_client.hpp"

#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <sstream>

namespace lkjai {
namespace {

struct UrlParts {
  std::string host;
  std::string port = "80";
  std::string path = "/";
};

bool parse_url(const std::string& url, UrlParts* out) {
  const std::string prefix = "http://";
  if (url.rfind(prefix, 0) != 0) return false;
  auto start = prefix.size();
  auto slash = url.find('/', start);
  auto authority = url.substr(start, slash - start);
  out->path = slash == std::string::npos ? "/" : url.substr(slash);
  auto colon = authority.rfind(':');
  if (colon == std::string::npos) {
    out->host = authority;
  } else {
    out->host = authority.substr(0, colon);
    out->port = authority.substr(colon + 1);
  }
  return !out->host.empty();
}

int connect_tcp(const UrlParts& url, std::string* error) {
  addrinfo hints{};
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* result = nullptr;
  int rc = getaddrinfo(url.host.c_str(), url.port.c_str(), &hints, &result);
  if (rc != 0) {
    *error = gai_strerror(rc);
    return -1;
  }
  int fd = -1;
  for (auto* ai = result; ai != nullptr; ai = ai->ai_next) {
    fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (fd < 0) continue;
    if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
    close(fd);
    fd = -1;
  }
  freeaddrinfo(result);
  if (fd < 0) *error = "connect failed";
  return fd;
}

NativeHttpResponse request(const std::string& method, const std::string& url,
                           const std::string& body) {
  NativeHttpResponse response;
  UrlParts parts;
  if (!parse_url(url, &parts)) {
    response.error = "only http:// URLs are supported";
    return response;
  }
  int fd = connect_tcp(parts, &response.error);
  if (fd < 0) return response;
  std::ostringstream wire;
  wire << method << " " << parts.path << " HTTP/1.1\r\n"
       << "host: " << parts.host << "\r\n"
       << "connection: close\r\n";
  if (!body.empty()) {
    wire << "content-type: application/json\r\n"
         << "content-length: " << body.size() << "\r\n";
  }
  wire << "\r\n" << body;
  auto text = wire.str();
  (void)send(fd, text.data(), text.size(), MSG_NOSIGNAL);
  char buffer[8192];
  std::string raw;
  for (;;) {
    ssize_t n = recv(fd, buffer, sizeof(buffer), 0);
    if (n <= 0) break;
    raw.append(buffer, static_cast<size_t>(n));
  }
  close(fd);
  auto line_end = raw.find("\r\n");
  if (line_end == std::string::npos) {
    response.error = "invalid HTTP response";
    return response;
  }
  try {
    response.status = std::stoi(raw.substr(9, 3));
  } catch (...) {
    response.error = "invalid HTTP status";
  }
  auto body_start = raw.find("\r\n\r\n");
  response.body = body_start == std::string::npos ? "" : raw.substr(body_start + 4);
  return response;
}

}  // namespace

NativeHttpResponse native_http_get(const std::string& url) {
  return request("GET", url, "");
}

NativeHttpResponse native_http_post_json(const std::string& url,
                                         const std::string& body) {
  return request("POST", url, body);
}

std::string model_url_to_models_url(const std::string& chat_url) {
  const std::string suffix = "/v1/chat/completions";
  if (chat_url.size() >= suffix.size() &&
      chat_url.compare(chat_url.size() - suffix.size(), suffix.size(), suffix) == 0) {
    return chat_url.substr(0, chat_url.size() - suffix.size()) + "/v1/models";
  }
  return chat_url;
}

std::string model_url_to_health_url(const std::string& chat_url) {
  const std::string suffix = "/v1/chat/completions";
  if (chat_url.size() >= suffix.size() &&
      chat_url.compare(chat_url.size() - suffix.size(), suffix.size(),
                       suffix) == 0) {
    return chat_url.substr(0, chat_url.size() - suffix.size()) + "/healthz";
  }
  return chat_url;
}

}  // namespace lkjai
