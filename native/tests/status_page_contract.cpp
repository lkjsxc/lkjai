#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

std::string read_file(const std::filesystem::path& path) {
  std::ifstream file(path);
  std::ostringstream out;
  out << file.rdbuf();
  return out.str();
}

bool has(const std::string& text, const std::string& needle) {
  return text.find(needle) != std::string::npos;
}

bool expect(bool ok, const std::string& message) {
  if (ok) return true;
  std::cerr << message << "\n";
  return false;
}

std::filesystem::path repo_root() {
  const char* env = std::getenv("LKJAI_REPO_ROOT");
  if (env && env[0]) return env;
  return std::filesystem::current_path();
}

bool static_web_contract() {
  auto root = repo_root();
  auto index = read_file(root / "web" / "index.html");
  auto app = read_file(root / "web" / "app.js");
  auto nginx = read_file(root / "web" / "nginx.conf");
  return expect(has(index, "app.js"), "index loads app") &&
         expect(has(app, "http://127.0.0.1:8082/api"),
                "sandbox api base") &&
         expect(has(app, "http://127.0.0.1:8082/healthz"),
                "sandbox health base") &&
         expect(has(app, "http://127.0.0.1:8081/v1"),
                "inference v1 base") &&
         expect(!has(app, "fetch('/api"), "no local api fetch") &&
         expect(!has(app, "fetch(\"/api"), "no local api fetch double") &&
         expect(!has(app, "fetch('/v1"), "no local v1 fetch") &&
         expect(!has(app, "fetch(\"/v1"), "no local v1 fetch double") &&
         expect(has(nginx, "listen 8080"), "web listens on 8080");
}

}  // namespace

int main() { return static_web_contract() ? 0 : 1; }
