#include <iostream>
#include <string>

#include "http_server.hpp"
#include "native_status_page.hpp"

int main() {
  lkjai::HttpResponse json{200, "{\"ok\":true}"};
  if (json.content_type != "application/json") {
    std::cerr << "default content type changed\n";
    return 1;
  }
  lkjai::HttpResponse html{200, std::string(lkjai::native_status_page_html()),
                           "text/html; charset=utf-8"};
  if (html.content_type.find("text/html") == std::string::npos ||
      html.body.find("<!doctype html>") == std::string::npos ||
      html.body.find("fetch('/healthz')") == std::string::npos ||
      html.body.find("fetch('/api/model')") == std::string::npos ||
      html.body.find("fetch('/api/dense/status')") == std::string::npos ||
      html.body.find("fetch('/api/dense/next-token'") == std::string::npos ||
      html.body.find("lkjai dense demo") == std::string::npos) {
    std::cerr << "native status page contract failed\n";
    return 1;
  }
  std::cout << "{\"status\":\"pass\",\"content_type\":\"text/html\"}\n";
  return 0;
}
