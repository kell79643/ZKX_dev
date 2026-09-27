#include "steps/step3.h"

#include <cusignal/operators/bsplines/bsplines_typed.h>
#include <cusignal/runtime/device_array.h>
#include <cusignal/operators/filtering/filtering_typed.h>

#include <numeric>

namespace task2 {

StepEvidence run_step3(PipelineState& state)
{
    StepEvidence evidence;
    evidence.name = "step3";
    const auto total_begin = Clock::now();
    auto begin = Clock::now();
    require(state.filtered.size() == kSamples, "step3 did not receive step2 output");
    constexpr int crop_begin = kStep3CropEach;
    constexpr int crop_end = kSamples - kStep3CropEach;
    std::vector<float> cropped(state.filtered.begin() + crop_begin,
                               state.filtered.begin() + crop_end);
    std::vector<float> offsets{-2.0F, -1.5F, -1.0F, -0.5F, 0.0F,
                               0.5F, 1.0F, 1.5F, 2.0F};
    evidence.prep_ms = milliseconds(begin, Clock::now());

    begin = Clock::now();
    auto d_cropped = cusignal::DeviceArray<float>::from_host(cropped);
    auto d_offsets = cusignal::DeviceArray<float>::from_host(offsets);
    evidence.h2d_ms = milliseconds(begin, Clock::now());
    cusignal::DeviceArray<float> d_spline(offsets.size());
    evidence.operator_ms["cubic"] = time_gpu([&] {
        cusignal::cubic_device(d_offsets, d_spline);
    });
    begin = Clock::now();
    auto weights = d_spline.to_host();
    evidence.d2h_ms += milliseconds(begin, Clock::now());
    const float weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0F);
    require(weight_sum > 0.0F, "step3 degenerate cubic window");
    for (float& value : weights) value /= weight_sum;
    begin = Clock::now();
    auto d_weights = cusignal::DeviceArray<float>::from_host(weights);
    evidence.h2d_ms += milliseconds(begin, Clock::now());
    cusignal::FirfilterOptions options;
    options.shape = {static_cast<int>(cropped.size())};
    options.axis = -1;
    cusignal::FirfilterDeviceResult smoothed;
    evidence.operator_ms["firfilter_b_spline"] = time_gpu([&] {
        cusignal::firfilter_device(d_weights, d_cropped, smoothed, options);
    });
    evidence.compute_ms = evidence.operator_ms["cubic"] +
        evidence.operator_ms["firfilter_b_spline"];
    begin = Clock::now();
    state.smoothed = smoothed.y.to_host();
    evidence.d2h_ms += milliseconds(begin, Clock::now());

    begin = Clock::now();
    state.reference_for_correlation.assign(
        state.target_waveform.begin() + crop_begin,
        state.target_waveform.begin() + crop_end);
    evidence.post_ms = milliseconds(begin, Clock::now());
    evidence.total_ms = milliseconds(total_begin, Clock::now());
    if (visualization_capture_enabled())
        state.spline_weights = weights;
    return evidence;
}

}  // namespace task2
