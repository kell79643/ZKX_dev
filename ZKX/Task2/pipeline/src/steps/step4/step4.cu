#include <task2/steps/step4.h>

#include <cusignal/operators/convolution/convolution_typed.h>
#include <cusignal/runtime/device_array.h>
#include <cusignal/runtime/kernel_launch.h>
#include <cusignal/operators/demod/demod_typed.h>
#include <cusignal/operators/spectral_analysis/spectral_analysis_typed.h>
#include <cusignal/operators/wavelets/wavelets_typed.h>

#include <numeric>

namespace task2 {
namespace {

__global__ void analytic_signal_kernel(
    const float* input, cusignal::DemodComplex<float>* output, int count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    const float previous = index == 0 ? input[0] : input[index - 1];
    const float next = index + 1 == count ? input[count - 1] : input[index + 1];
    output[index] = {input[index], 0.5F * (next - previous)};
}

std::vector<float> spectral_envelope(
    const std::vector<float>& flattened, int frequencies, int frames)
{
    std::vector<float> output(static_cast<std::size_t>(frames), 0.0F);
    for (int frequency = 0; frequency < frequencies; ++frequency)
        for (int frame = 0; frame < frames; ++frame)
            output[frame] = std::max(output[frame],
                flattened[static_cast<std::size_t>(frequency) * frames + frame]);
    return output;
}

std::vector<float> wavelet_envelope(
    const std::vector<double>& flattened, int widths, int samples)
{
    std::vector<float> output(static_cast<std::size_t>(samples), 0.0F);
    for (int width = 0; width < widths; ++width)
        for (int sample = 0; sample < samples; ++sample)
            output[sample] = std::max(output[sample],
                cusignal::cuda_utils::narrow_fp64_value_to_fp32_on_host(std::abs(
                    flattened[static_cast<std::size_t>(width) * samples + sample])));
    return output;
}

std::vector<float> make_bundle(
    const std::vector<float>& fm,
    const std::vector<float>& correlation,
    const std::vector<float>& spectral,
    const std::vector<float>& wavelet,
    std::size_t count, const TaskConfig& config)
{
    const auto fm_n = normalize_abs(resample_linear(fm, count));
    const auto corr_n = normalize_abs(resample_linear(correlation, count));
    const auto spec_n = normalize_abs(resample_linear(spectral, count));
    const auto wave_n = normalize_abs(resample_linear(wavelet, count));
    std::vector<float> bundle(count);
    for (std::size_t index = 0; index < count; ++index)
        bundle[index] = config.fusion_weight_fm * fm_n[index] +
                        config.fusion_weight_correlation * corr_n[index] +
                        config.fusion_weight_spectral * spec_n[index] +
                        config.fusion_weight_wavelet * wave_n[index];
    return bundle;
}

}  // namespace

StepEvidence run_step4(PipelineState& state)
{
    StepEvidence evidence;
    evidence.name = "step4";
    const auto total_begin = Clock::now();
    auto begin = Clock::now();
    const int count = static_cast<int>(state.smoothed.size());
    require_task2_upstream(count > 64, "Step4", "Step3.smoothed");
    require_task2_upstream(
        state.reference_for_correlation.size() == state.smoothed.size(),
        "Step4", "Step3.reference_for_correlation");
    constexpr int frame = 64;
    constexpr int hop = 32;
    const int frames = 1 + (count - frame) / hop;
    const std::vector<float> widths{2.0F, 4.0F, 8.0F, 12.0F};
    const cusignal::CwtRealWaveletCallable ricker =
        [](int points, int width) {
            return cusignal::ricker_typed_cpu<float>(points, static_cast<float>(width));
        };
    auto cwt_workspace = cusignal::prepare_cwt_workspace(count, widths, ricker);
    evidence.prep_ms = milliseconds(begin, Clock::now());

    begin = Clock::now();
    auto d_signal = cusignal::DeviceArray<float>::from_host(state.smoothed);
    auto d_reference = cusignal::DeviceArray<float>::from_host(state.reference_for_correlation);
    evidence.h2d_ms = milliseconds(begin, Clock::now());
    cusignal::DeviceArray<cusignal::DemodComplex<float>> d_analytic(count);
    evidence.operator_ms["analytic_glue"] = time_gpu([&] {
        cusignal::cuda_utils::launch_1d_kernel(
            analytic_signal_kernel, static_cast<std::size_t>(count),
            d_signal.data(), d_analytic.data(), count);
    });
    cusignal::FmDemodWorkspace demod_workspace(cusignal::FmDemodOptions{{count}, -1});
    cusignal::DeviceArray<float> d_fm(demod_workspace.output_size());
    evidence.operator_ms["fm_demod"] = time_gpu([&] {
        cusignal::fm_demod_device(d_analytic, d_fm, demod_workspace);
    });
    cusignal::DeviceArray<float> d_correlation(count);
    evidence.operator_ms["correlate"] = time_gpu([&] {
        cusignal::correlate_device(d_signal, d_reference, d_correlation, "same");
    });
    cusignal::DeviceArray<float> d_spectral(static_cast<std::size_t>(frame) * frames);
    evidence.operator_ms["spectrogram"] = time_gpu([&] {
        cusignal::spectrogram_device(d_signal, frame, hop, d_spectral);
    });
    cusignal::DeviceArray<double> d_cwt(cwt_workspace.output_size());
    evidence.operator_ms["cwt_ricker"] = time_gpu([&] {
        cusignal::cwt_device(d_signal, cwt_workspace, d_cwt);
    });
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    evidence.metrics["gpu_used_peak_bytes"] =
        static_cast<double>(total_bytes - free_bytes);
    evidence.compute_ms = std::accumulate(
        evidence.operator_ms.begin(), evidence.operator_ms.end(), 0.0,
        [](double sum, const auto& item) { return sum + item.second; });

    begin = Clock::now();
    state.fm_feature = d_fm.to_host();
    state.correlation_feature = d_correlation.to_host();
    const auto spectral_flat = d_spectral.to_host();
    const auto cwt_flat = d_cwt.to_host();
    evidence.d2h_ms = milliseconds(begin, Clock::now());
    begin = Clock::now();
    state.spectral_feature = spectral_envelope(spectral_flat, frame, frames);
    state.wavelet_feature = wavelet_envelope(
        cwt_flat, static_cast<int>(widths.size()), count);
    state.feature_bundle = make_bundle(
        state.fm_feature, state.correlation_feature,
        state.spectral_feature, state.wavelet_feature, count, state.config);

    evidence.post_ms = milliseconds(begin, Clock::now());
    evidence.total_ms = milliseconds(total_begin, Clock::now());
    if (visualization_capture_enabled()) {
        state.spectrogram_grid = spectral_flat;
        state.cwt_grid = cwt_flat;
        state.spectrogram_frequency_bins = frame;
        state.spectrogram_frames = frames;
        state.cwt_width_count = static_cast<int>(widths.size());
    }
    return evidence;
}

}  // namespace task2
