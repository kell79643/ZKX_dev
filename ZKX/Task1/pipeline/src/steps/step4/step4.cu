#include <task1/steps/step4.h>

#include <cusignal/runtime/device_array.h>
#include <cusignal/runtime/kernel_launch.h>

#include <numeric>

namespace task1 {
namespace {

__global__ void power_kernel(
    const ComplexFloat* input, float* output, int count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    const auto value = input[index];
    output[index] = value.re * value.re + value.im * value.im;
}

}  // namespace

StepEvidence run_step4(PipelineState& state)
{
    const auto& config = state.config;
    StepEvidence evidence;
    evidence.name = "step4";
    const auto formal_begin = Clock::now();
    auto begin = Clock::now();
    const int count = config.num_pulses * config.samples_per_pulse;
    const bool has_resident_input =
        state.range_doppler_device.size() == static_cast<std::size_t>(count);
    require_task1_upstream(has_resident_input ||
            state.range_doppler.size() == static_cast<std::size_t>(count),
            "Step4", "Step3.range_doppler");
    cusignal::CaCfarOptions options;
    options.guard_cells = {config.cfar_guard_doppler, config.cfar_guard_range};
    options.reference_cells = {config.cfar_reference_doppler, config.cfar_reference_range};
    options.pfa = config.pfa;
    const int outer_doppler = config.cfar_guard_doppler + config.cfar_reference_doppler;
    const int outer_range = config.cfar_guard_range + config.cfar_reference_range;
    const int reference_count = (2 * outer_doppler + 1) * (2 * outer_range + 1) -
        (2 * config.cfar_guard_doppler + 1) * (2 * config.cfar_guard_range + 1);
    evidence.prep_ms = milliseconds(begin, Clock::now());
    begin = Clock::now();
    cusignal::DeviceArray<cusignal::TypedComplex<float>> d_range_doppler_fallback;
    const ComplexFloat* range_doppler_device = nullptr;
    if (has_resident_input) {
        range_doppler_device = state.range_doppler_device.data();
        evidence.h2d_ms = 0.0;
        evidence.metrics["resident_input_reused"] = 1.0;
    } else {
        static_assert(sizeof(cusignal::TypedComplex<float>) ==
            sizeof(ComplexFloat));
        d_range_doppler_fallback =
            cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host(
                state.range_doppler);
        range_doppler_device =
            reinterpret_cast<const ComplexFloat*>(
                d_range_doppler_fallback.data());
        evidence.h2d_ms = milliseconds(begin, Clock::now());
        evidence.metrics["resident_input_reused"] = 0.0;
    }
    cusignal::DeviceArray<float> d_power(count);
    evidence.operator_ms["range_doppler_power"] = time_gpu([&] {
        cusignal::cuda_utils::launch_1d_kernel(
            power_kernel, static_cast<std::size_t>(count),
            range_doppler_device, d_power.data(), count);
    });
    cusignal::DeviceArray<double> d_alpha;
    evidence.operator_ms["cfar_alpha"] = time_gpu([&] {
        d_alpha = cusignal::cfar_alpha_device<float>(config.pfa, reference_count);
    });
    thread_local cusignal::DeviceArray<float> d_cfar_threshold;
    thread_local cusignal::DeviceArray<cusignal::CfarDetection> d_cfar_detections;
    if (d_cfar_threshold.size() != static_cast<std::size_t>(count)) {
        d_cfar_threshold.reset(static_cast<std::size_t>(count));
    }
    if (d_cfar_detections.size() != static_cast<std::size_t>(count)) {
        d_cfar_detections.reset(static_cast<std::size_t>(count));
    }
    evidence.operator_ms["ca_cfar"] = time_gpu([&] {
        cusignal::ca_cfar_device<float>(
            d_power, {config.num_pulses, config.samples_per_pulse}, options,
            d_cfar_threshold, d_cfar_detections);
    });
    evidence.compute_ms = std::accumulate(
        evidence.operator_ms.begin(), evidence.operator_ms.end(), 0.0,
        [](double sum, const auto& item) { return sum + item.second; });
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    evidence.metrics["gpu_used_peak_bytes"] = static_cast<double>(total_bytes - free_bytes);
    begin = Clock::now();
    state.power = d_power.to_host();
    state.cfar_alpha = d_alpha.to_host().front();
    state.cfar_threshold = d_cfar_threshold.to_host();
    state.detections = d_cfar_detections.to_host();
    evidence.d2h_ms = milliseconds(begin, Clock::now());
    evidence.formal_execution_ms = milliseconds(formal_begin, Clock::now());
    return evidence;
}

}  // namespace task1
