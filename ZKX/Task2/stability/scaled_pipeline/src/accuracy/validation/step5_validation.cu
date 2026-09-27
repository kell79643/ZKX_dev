#include "accuracy/validation/step5_validation.h"

#include <cusignal/runtime/device_array.h>
#include <cusignal/operators/peak_finding/peak_finding_typed.h>

namespace task2::accuracy {

Step5ValidationData collect_step5_cuda_validation(const PipelineState& state)
{
    Step5ValidationData data;
    const cusignal::RelativeExtremaShape shape({state.feature_bundle.size()});
    auto injected = state.feature_bundle;
    const float high = *std::max_element(injected.begin(), injected.end()) + 10.0F;
    injected[50] = high;
    auto d_injected = cusignal::DeviceArray<float>::from_host(injected);
    auto injected_result = cusignal::argrelextrema_device(
        d_injected, shape, cusignal::RelativeExtremaComparator::greater,
        0, 3, "clip");
    data.injected_gpu = injected_result.coordinates.front().to_host();
    injected[50] = state.feature_bundle[50];
    injected[70] = high;
    d_injected = cusignal::DeviceArray<float>::from_host(injected);
    auto moved_result = cusignal::argrelextrema_device(
        d_injected, shape, cusignal::RelativeExtremaComparator::greater,
        0, 3, "clip");
    data.moved_gpu = moved_result.coordinates.front().to_host();
    return data;
}

}  // namespace task2::accuracy
