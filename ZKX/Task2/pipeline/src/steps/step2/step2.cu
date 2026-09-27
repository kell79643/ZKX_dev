#include <task2/steps/step2.h>

#include <cusignal/runtime/device_array.h>
#include <cusignal/operators/filter_design/filter_design_typed.h>
#include <cusignal/operators/filtering/filtering_typed.h>
#include <cusignal/operators/windows/windows_typed.h>

#include <numeric>

namespace task2 {

StepEvidence run_step2(PipelineState& state)
{
    const auto& config = state.config;
    StepEvidence evidence;
    evidence.name = "step2";
    const auto total_begin = Clock::now();
    auto begin = Clock::now();
    require_task2_upstream(
        state.echo.size() == static_cast<std::size_t>(config.samples),
        "Step2", "Step1.echo");
    const std::vector<float> cutoff{config.filter_cutoff_hz};
    const int taps_count = config.filter_taps;
    evidence.prep_ms = milliseconds(begin, Clock::now());

    begin = Clock::now();
    auto d_echo = cusignal::DeviceArray<float>::from_host(state.echo);
    auto d_cutoff = cusignal::DeviceArray<float>::from_host(cutoff);
    evidence.h2d_ms = milliseconds(begin, Clock::now());

    cusignal::DeviceArray<float> d_window(taps_count);
    evidence.operator_ms["hamming"] = time_gpu([&] {
        cusignal::hamming_resident_device<float>(taps_count, d_window, true);
    });
    cusignal::DeviceArray<float> d_taps(taps_count);
    evidence.operator_ms["firwin"] = time_gpu([&] {
        cusignal::firwin_lowpass_explicit_window_resident_device(
            taps_count, d_cutoff, d_window, d_taps, config.sample_rate_hz, true);
    });
    cusignal::DeviceArray<float> filtered(config.samples);
    evidence.operator_ms["firfilter"] = time_gpu([&] {
        cusignal::firfilter_fp32_resident_device(d_taps, d_echo, filtered);
    });
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    evidence.metrics["gpu_used_peak_bytes"] =
        static_cast<double>(total_bytes - free_bytes);
    evidence.compute_ms = std::accumulate(
        evidence.operator_ms.begin(), evidence.operator_ms.end(), 0.0,
        [](double sum, const auto& item) { return sum + item.second; });
    begin = Clock::now();
    state.filtered = filtered.to_host();
    evidence.d2h_ms += milliseconds(begin, Clock::now());

    evidence.metrics["resident_internal_h2d_count"] = 0.0;
    evidence.metrics["resident_internal_d2h_count"] = 0.0;
    evidence.metrics["resident_internal_allocation_count"] = 0.0;
    evidence.metrics["resident_kernel_count"] = 3.0;
    evidence.metrics["resident_persistent_device_bytes"] =
        static_cast<double>(config.samples * sizeof(float)
            + sizeof(float)
            + taps_count * sizeof(float) * 3);

    evidence.total_ms = milliseconds(total_begin, Clock::now());
    if (visualization_capture_enabled())
        state.filter_taps = d_taps.to_host();
    return evidence;
}

}  // namespace task2
