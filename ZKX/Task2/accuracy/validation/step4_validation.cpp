#include "accuracy/validation/step4_validation.h"

#include <cusignal/operators/convolution/convolution_typed.h>
#include <cusignal/operators/demod/demod_typed.h>
#include <cusignal/operators/spectral_analysis/spectral_analysis_typed.h>
#include <cusignal/operators/wavelets/wavelets_typed.h>

namespace task2::accuracy {
namespace {

std::vector<cusignal::DemodComplex<float>> analytic_cpu(const std::vector<float>& input)
{
    std::vector<cusignal::DemodComplex<float>> output(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float previous = index == 0 ? input.front() : input[index - 1];
        const float next = index + 1 == input.size() ? input.back() : input[index + 1];
        output[index] = {input[index], 0.5F * (next - previous)};
    }
    return output;
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
            output[sample] = std::max(output[sample], static_cast<float>(std::abs(
                flattened[static_cast<std::size_t>(width) * samples + sample])));
    return output;
}

std::vector<float> make_bundle(
    const std::vector<float>& fm, const std::vector<float>& correlation,
    const std::vector<float>& spectral, const std::vector<float>& wavelet,
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

Step4ValidationData collect_step4_validation_impl(
    const PipelineState& state, std::vector<float>* precomputed_correlation)
{
    Step4ValidationData data;
    const int count = static_cast<int>(state.smoothed.size());
    constexpr int frame = 64;
    constexpr int hop = 32;
    const int frames = 1 + (count - frame) / hop;
    const std::vector<float> widths{2.0F, 4.0F, 8.0F, 12.0F};
    const cusignal::CwtRealWaveletCallable ricker = [](int points, int width) {
        return cusignal::ricker_typed_cpu<float>(points, static_cast<float>(width));
    };
    cusignal::FmDemodWorkspace demod_workspace(cusignal::FmDemodOptions{{count}, -1});
    data.fm_cpu = cusignal::fm_demod_typed_cpu(
        analytic_cpu(state.smoothed), demod_workspace);
    if (precomputed_correlation) {
        if (precomputed_correlation->size() != state.smoothed.size())
            throw std::invalid_argument(
                "precomputed Task2 Step4 correlation size mismatch");
        data.correlation_cpu = std::move(*precomputed_correlation);
    } else {
        data.correlation_cpu = cusignal::correlate_typed_cpu(
            state.smoothed, state.reference_for_correlation, "same",
            cusignal::CorrelateMethod::direct);
    }
    data.spectral_cpu = spectral_envelope(
        cusignal::spectrogram_typed_cpu(state.smoothed, frame, hop), frame, frames);
    data.wavelet_cpu = wavelet_envelope(
        cusignal::cwt_typed_cpu(state.smoothed, widths, ricker),
        static_cast<int>(widths.size()), count);
    data.bundle_cpu = make_bundle(
        data.fm_cpu, data.correlation_cpu, data.spectral_cpu, data.wavelet_cpu,
        count, state.config);
    return data;
}

Step4ValidationData collect_step4_validation(const PipelineState& state)
{
    return collect_step4_validation_impl(state, nullptr);
}

Step4ValidationData collect_step4_validation_with_precomputed_correlation(
    const PipelineState& state, std::vector<float> correlation_cpu)
{
    return collect_step4_validation_impl(state, &correlation_cpu);
}

}  // namespace task2::accuracy
