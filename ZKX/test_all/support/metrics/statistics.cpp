#include "statistics.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace zkx::common {
static double percentile(const std::vector<double>& sorted, double q)
{
    const double position = q * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<std::size_t>(position);
    const auto upper = std::min(lower + 1, sorted.size() - 1);
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}
Statistics summarize_ms(const std::vector<double>& samples)
{
    if (samples.empty()) throw std::invalid_argument("timing samples must not be empty");
    for (double value : samples)
        if (!std::isfinite(value) || value < 0.0) throw std::invalid_argument("invalid timing sample");
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    Statistics out;
    out.mean_ms = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
    out.p50_ms = percentile(sorted, 0.50);
    out.p95_ms = percentile(sorted, 0.95);
    out.p99_ms = percentile(sorted, 0.99);
    out.min_ms = sorted.front();
    out.max_ms = sorted.back();
    double sum = 0.0;
    for (double value : sorted) sum += (value - out.mean_ms) * (value - out.mean_ms);
    out.stddev_ms = std::sqrt(sum / sorted.size());
    out.cv = out.mean_ms == 0.0 ? 0.0 : out.stddev_ms / out.mean_ms;
    return out;
}
double speedup(const Statistics& cpu, const Statistics& gpu,
               const std::string& cpu_scope, const std::string& gpu_scope)
{
    if (cpu_scope != gpu_scope) throw std::invalid_argument("timing scope mismatch");
    if (gpu.mean_ms <= 0.0) throw std::invalid_argument("GPU mean must be positive");
    return cpu.mean_ms / gpu.mean_ms;
}
AuxiliaryGpuTiming measure_cuda_event_auxiliary(const std::function<double()>& measurement)
{
    try {
        const double value = measurement();
        if (!std::isfinite(value) || value < 0.0) throw std::runtime_error("invalid cudaEvent duration");
        return {true, value, "cudaEvent", {}};
    } catch (const std::exception& error) {
        return {false, 0.0, "cudaEvent", error.what()};
    }
}
}  // namespace zkx::common
