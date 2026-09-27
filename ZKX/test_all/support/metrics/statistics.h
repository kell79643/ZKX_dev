#pragma once
#include <chrono>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace zkx::common {
struct Statistics {
    double mean_ms{}, p50_ms{}, p95_ms{}, p99_ms{}, min_ms{}, max_ms{}, stddev_ms{}, cv{};
};
struct AuxiliaryGpuTiming { bool available{}; double cuda_event_ms{}; std::string source{"cudaEvent"}; std::string error; };
Statistics summarize_ms(const std::vector<double>& samples);
double speedup(const Statistics& cpu, const Statistics& gpu,
               const std::string& cpu_scope, const std::string& gpu_scope);
AuxiliaryGpuTiming measure_cuda_event_auxiliary(const std::function<double()>& cuda_event_measurement);

template <class Work, class Synchronize>
double measure_ms(Work&& work, Synchronize&& synchronize)
{
    const auto begin = std::chrono::steady_clock::now();
    std::forward<Work>(work)();
    std::forward<Synchronize>(synchronize)();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

template <class Work, class Synchronize>
std::vector<double> collect_ms(std::size_t warmup, std::size_t measured,
                               Work&& work, Synchronize&& synchronize)
{
    if (measured == 0) throw std::invalid_argument("measured iterations must be positive");
    for (std::size_t i = 0; i < warmup; ++i) { work(); synchronize(); }
    std::vector<double> samples;
    samples.reserve(measured);
    for (std::size_t i = 0; i < measured; ++i) samples.push_back(measure_ms(work, synchronize));
    return samples;
}
}  // namespace zkx::common
