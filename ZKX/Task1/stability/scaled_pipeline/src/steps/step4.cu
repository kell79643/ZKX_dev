#include "steps/step4.h"

#include <cusignal/runtime/device_array.h>
#include <cusignal/runtime/kernel_launch.h>

#include <numeric>

namespace task1 {
namespace {

__global__ void power_kernel(
    const cusignal::TypedComplex<float>* input, float* output, int count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    const auto value = input[index];
    output[index] = value.real * value.real + value.imag * value.imag;
}

}  // namespace

StepEvidence run_step4(PipelineState& state)
{
    StepEvidence evidence;
    evidence.name = "step4";
    const auto formal_begin = Clock::now();
    auto begin = Clock::now();
    const int count = kNumPulses * kSamplesPerPulse;
    require(state.range_doppler.size() == static_cast<std::size_t>(count),
            "step4 did not receive the step3 range-Doppler map");
    cusignal::CaCfarOptions options;
    options.guard_cells = {1, 2};
    options.reference_cells = {2, 6};
    options.pfa = kPfa;
    constexpr int reference_count = 104;
    evidence.prep_ms = milliseconds(begin, Clock::now());
    begin = Clock::now();
    auto d_range_doppler =
        cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host(state.range_doppler);
    evidence.h2d_ms = milliseconds(begin, Clock::now());
    cusignal::DeviceArray<float> d_power(count);
    evidence.operator_ms["range_doppler_power"] = time_gpu([&] {
        cusignal::cuda_utils::launch_1d_kernel(
            power_kernel, static_cast<std::size_t>(count),
            d_range_doppler.data(), d_power.data(), count);
    });
    cusignal::DeviceArray<double> d_alpha;
    evidence.operator_ms["cfar_alpha"] = time_gpu([&] {
        d_alpha = cusignal::cfar_alpha_device<float>(kPfa, reference_count);
    });
    cusignal::CaCfarDeviceResult gpu_result;
    evidence.operator_ms["ca_cfar"] = time_gpu([&] {
        gpu_result = cusignal::ca_cfar_device<float>(
            d_power, {kNumPulses, kSamplesPerPulse}, options);
    });
    evidence.compute_ms = std::accumulate(
        evidence.operator_ms.begin(), evidence.operator_ms.end(), 0.0,
        [](double sum, const auto& item) { return sum + item.second; });
    begin = Clock::now();
    state.power = d_power.to_host();
    state.cfar_alpha = d_alpha.to_host().front();
    state.cfar_threshold = gpu_result.threshold.to_host();
    state.detections = gpu_result.detections.to_host();
    evidence.d2h_ms = milliseconds(begin, Clock::now());
    evidence.formal_execution_ms = milliseconds(formal_begin, Clock::now());
    return evidence;
}

}  // namespace task1
