#include "runtime_device.hpp"

#include <atomic>

namespace lkjai {
namespace {

std::atomic<uint64_t> g_allocation_count{0};
std::atomic<uint64_t> g_free_count{0};
std::atomic<uint64_t> g_allocated_bytes{0};
std::atomic<uint64_t> g_freed_bytes{0};
std::atomic<uint64_t> g_live_bytes{0};
std::atomic<uint64_t> g_high_water_live_bytes{0};

}  // namespace

DeviceAllocationStats device_allocation_stats() {
  DeviceAllocationStats stats;
  stats.allocation_count = g_allocation_count.load();
  stats.free_count = g_free_count.load();
  stats.allocated_bytes = g_allocated_bytes.load();
  stats.freed_bytes = g_freed_bytes.load();
  stats.live_bytes = g_live_bytes.load();
  stats.high_water_live_bytes = g_high_water_live_bytes.load();
  return stats;
}

uint64_t device_allocation_count_delta(const DeviceAllocationStats& before,
                                       const DeviceAllocationStats& after) {
  return after.allocation_count >= before.allocation_count
             ? after.allocation_count - before.allocation_count
             : 0;
}

void device_allocation_account_alloc(size_t bytes) {
  if (bytes == 0) return;
  g_allocation_count.fetch_add(1);
  g_allocated_bytes.fetch_add(bytes);
  uint64_t live = g_live_bytes.fetch_add(bytes) + bytes;
  uint64_t high = g_high_water_live_bytes.load();
  while (live > high &&
         !g_high_water_live_bytes.compare_exchange_weak(high, live)) {
  }
}

void device_allocation_account_free(size_t bytes) {
  if (bytes == 0) return;
  g_free_count.fetch_add(1);
  g_freed_bytes.fetch_add(bytes);
  g_live_bytes.fetch_sub(bytes);
}

}  // namespace lkjai
