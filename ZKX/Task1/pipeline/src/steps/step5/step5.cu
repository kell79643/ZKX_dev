#include <task1/steps/step5.h>

#include <cusignal/runtime/device_array.h>

#include <numeric>

namespace task1 {
namespace {

std::vector<double> ambgfun_to_double(const cusignal::AmbgfunDeviceResult& result)
{
    if (result.dtype == cusignal::AmbgfunOutputDtype::fp32) {
        const auto values = result.fp32.to_host();
        return std::vector<double>(values.begin(), values.end());
    }
    return result.fp64.to_host();
}

}  // namespace

StepEvidence run_step5(PipelineState& state)
{
    const auto& config = state.config;
    StepEvidence evidence;
    evidence.name = "step5";
    const auto formal_begin = Clock::now();
    auto begin = Clock::now();
    require_task1_upstream(
            state.waveform.size() == static_cast<std::size_t>(config.pulse_samples),
            "Step5", "Step1.waveform");
    cusignal::AmbgfunOptions options;
    options.fs = config.sample_rate_hz;
    options.prf = config.prf_hz;
    options.cut = cusignal::AmbgfunCut::two_dimensional;
    evidence.prep_ms = milliseconds(begin, Clock::now());
    begin = Clock::now();
    auto d_waveform =
        cusignal::DeviceArray<cusignal::TypedComplex<float>>::from_host(state.waveform);
    evidence.h2d_ms = milliseconds(begin, Clock::now());
    cusignal::AmbgfunDeviceResult gpu_2d;
    evidence.operator_ms["ambgfun_2d"] = time_gpu([&] {
        cusignal::ambgfun_device<float>(d_waveform, gpu_2d, options);
    });
    options.cut = cusignal::AmbgfunCut::doppler;
    options.cut_value = 0.0;
    cusignal::AmbgfunDeviceResult gpu_delay;
    evidence.operator_ms["ambgfun_delay_cut"] = time_gpu([&] {
        cusignal::ambgfun_device<float>(d_waveform, gpu_delay, options);
    });
    options.cut = cusignal::AmbgfunCut::delay;
    cusignal::AmbgfunDeviceResult gpu_doppler;
    evidence.operator_ms["ambgfun_doppler_cut"] = time_gpu([&] {
        cusignal::ambgfun_device<float>(d_waveform, gpu_doppler, options);
    });
    evidence.compute_ms = std::accumulate(
        evidence.operator_ms.begin(), evidence.operator_ms.end(), 0.0,
        [](double sum, const auto& item) { return sum + item.second; });
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    evidence.metrics["gpu_used_peak_bytes"] = static_cast<double>(total_bytes - free_bytes);
    begin = Clock::now();
    const auto gpu_2d_double = ambgfun_to_double(gpu_2d);
    state.ambiguity_2d.assign(gpu_2d_double.begin(), gpu_2d_double.end());
    state.ambiguity_delay = ambgfun_to_double(gpu_delay);
    state.ambiguity_doppler = ambgfun_to_double(gpu_doppler);
    evidence.d2h_ms = milliseconds(begin, Clock::now());
    evidence.formal_execution_ms = milliseconds(formal_begin, Clock::now());
    return evidence;
}

}  // namespace task1
