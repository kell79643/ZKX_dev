#pragma once
#include <cstdint>
#include <functional>
#include <string>
namespace zkx::common {
struct ResourceSample {
    bool available{};
    std::int64_t bytes{};
    std::string source;
    std::string error;
};
ResourceSample sample_cpu_heap();
ResourceSample sample_cpu_rss();
ResourceSample sample_gpu_used(const std::function<std::int64_t()>& platform_sampler,
                               const std::string& source);
}  // namespace zkx::common
