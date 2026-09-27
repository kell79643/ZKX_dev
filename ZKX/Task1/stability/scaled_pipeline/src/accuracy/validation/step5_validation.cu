#include "accuracy/validation/step5_validation.h"

namespace task1::accuracy {
namespace {

std::vector<double> to_double(const cusignal::AmbgfunCpuResult& result)
{
    if (result.dtype == cusignal::AmbgfunOutputDtype::fp32)
        return std::vector<double>(result.fp32.begin(), result.fp32.end());
    return result.fp64;
}

}  // namespace

Step5ValidationData collect_step5_validation(const PipelineState& state)
{
    Step5ValidationData data;
    cusignal::AmbgfunOptions options;
    options.fs = kSampleRate;
    options.prf = kPrf;
    options.cut = cusignal::AmbgfunCut::two_dimensional;
    data.ambiguity_2d_cpu = to_double(
        cusignal::ambgfun_typed_cpu<float>(state.waveform, options));
    options.cut = cusignal::AmbgfunCut::doppler;
    options.cut_value = 0.0;
    data.ambiguity_delay_cpu = to_double(
        cusignal::ambgfun_typed_cpu<float>(state.waveform, options));
    options.cut = cusignal::AmbgfunCut::delay;
    data.ambiguity_doppler_cpu = to_double(
        cusignal::ambgfun_typed_cpu<float>(state.waveform, options));
    auto changed_waveform = state.waveform;
    changed_waveform[kPulseSamples / 3].real *= 0.5F;
    options.cut = cusignal::AmbgfunCut::doppler;
    data.changed_waveform_cpu = to_double(
        cusignal::ambgfun_typed_cpu<float>(changed_waveform, options));
    return data;
}

}  // namespace task1::accuracy
