#include "memory.h"
#include <fstream>
#include <stdexcept>
#include <unistd.h>
#if defined(__GLIBC__)
#include <malloc.h>
#endif
namespace zkx::common {
ResourceSample sample_cpu_heap()
{
#if defined(__GLIBC__)
    const auto info = mallinfo2();
    return {true, static_cast<std::int64_t>(info.uordblks), "mallinfo2.uordblks", {}};
#else
    return {false, 0, "unsupported", "mallinfo2 unavailable"};
#endif
}
ResourceSample sample_cpu_rss()
{
    std::ifstream stream("/proc/self/statm");
    std::int64_t total = 0, resident = 0;
    if (!(stream >> total >> resident)) return {false, 0, "/proc/self/statm", "read failed"};
    return {true, resident * static_cast<std::int64_t>(sysconf(_SC_PAGESIZE)), "/proc/self/statm", {}};
}
ResourceSample sample_gpu_used(const std::function<std::int64_t()>& sampler, const std::string& source)
{
    try {
        const auto value = sampler();
        if (value < 0) throw std::runtime_error("negative byte count");
        return {true, value, source, {}};
    } catch (const std::exception& error) {
        return {false, 0, source, error.what()};
    }
}
}  // namespace zkx::common
