#include <iostream>

#include "dense_cuda.hpp"
#include "json_min.hpp"

int main() {
  auto check = lkjai::run_dense_cuda_check();
  std::cout << "{\"status\":\"" << (check.ok ? "pass" : "fail")
            << "\",\"device\":\"" << lkjai::json_escape(check.device)
            << "\",\"error\":\"" << lkjai::json_escape(check.error)
            << "\"}\n";
  return check.ok ? 0 : 1;
}
