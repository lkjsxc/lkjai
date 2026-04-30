#include "http_server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <iostream>

namespace lkjai {
namespace {

HttpRequest parse_request(const std::string& raw) {
  HttpRequest request;
  auto first = raw.find("\r\n");
  auto line = raw.substr(0, first);
  auto a = line.find(' ');
  auto b = line.find(' ', a + 1);
  if (a != std::string::npos && b != std::string::npos) {
    request.method = line.substr(0, a);
    request.path = line.substr(a + 1, b - a - 1);
  }
  auto split = raw.find("\r\n\r\n");
  if (split != std::string::npos) {
    request.body = raw.substr(split + 4);
  }
  return request;
}

std::string reason(int status) {
  if (status == 200) return "OK";
  if (status == 404) return "Not Found";
  if (status == 503) return "Service Unavailable";
  return "Internal Server Error";
}

void write_response(int client, const HttpResponse& response) {
  std::string head = "HTTP/1.1 " + std::to_string(response.status) + " " +
                     reason(response.status) + "\r\n";
  head += "content-type: application/json\r\n";
  head += "content-length: " + std::to_string(response.body.size()) + "\r\n";
  head += "connection: close\r\n\r\n";
  auto wire = head + response.body;
  (void)::send(client, wire.data(), wire.size(), MSG_NOSIGNAL);
}

}  // namespace

int serve_http(const std::string& host, int port, const Handler& handler) {
  int server = ::socket(AF_INET, SOCK_STREAM, 0);
  if (server < 0) return 1;
  int reuse = 1;
  setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<uint16_t>(port));
  if (::inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1) {
    address.sin_addr.s_addr = INADDR_ANY;
  }
  if (::bind(server, reinterpret_cast<sockaddr*>(&address), sizeof(address))) {
    std::cerr << "bind failed: " << std::strerror(errno) << "\n";
    return 1;
  }
  if (::listen(server, 64)) return 1;
  std::cerr << "lkjai-native-server listening on " << host << ":" << port
            << "\n";
  for (;;) {
    int client = ::accept(server, nullptr, nullptr);
    if (client < 0) continue;
    char buffer[65536];
    ssize_t size = ::recv(client, buffer, sizeof(buffer), 0);
    if (size > 0) {
      auto response = handler(parse_request(std::string(buffer, size)));
      write_response(client, response);
    }
    ::close(client);
  }
}

}  // namespace lkjai
