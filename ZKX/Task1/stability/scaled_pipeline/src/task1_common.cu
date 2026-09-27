#include "task1_common.h"

#include <cusignal/runtime/host_output_finalize.h>

namespace task1 {

double milliseconds(Clock::time_point begin, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

std::vector<cusignal::TypedComplex<float>> radar_result_to_host(
    const cusignal::RadarComplexDeviceResult& result)
{
    std::vector<cusignal::TypedComplex<float>> output;
    if (result.is_fp64) {
        const auto values = cusignal::cuda_utils::narrow_complex_fp64_to_fp32_on_host(
            result.fp64.to_host());
        output.resize(values.size());
        for (std::size_t index = 0; index < values.size(); ++index)
            output[index] = {values[index].re, values[index].im};
    } else {
        const auto values = result.fp32.to_host();
        output.resize(values.size());
        for (std::size_t index = 0; index < values.size(); ++index)
            output[index] = {values[index].re, values[index].im};
    }
    return output;
}

std::vector<cusignal::TypedComplex<float>> radar_cpu_to_host(
    const cusignal::RadarComplexCpuResult& result)
{
    std::vector<cusignal::TypedComplex<float>> output;
    if (result.is_fp64) {
        const auto values = cusignal::cuda_utils::narrow_complex_fp64_to_fp32_on_host(
            result.fp64);
        output.resize(values.size());
        for (std::size_t index = 0; index < values.size(); ++index)
            output[index] = {values[index].re, values[index].im};
    } else {
        output.resize(result.fp32.size());
        for (std::size_t index = 0; index < result.fp32.size(); ++index)
            output[index] = {result.fp32[index].re, result.fp32[index].im};
    }
    return output;
}

}  // namespace task1
