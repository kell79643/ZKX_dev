#include "accuracy/validation/step5_validation.h"

#include <cusignal/operators/peak_finding/peak_finding_typed.h>

namespace task2::accuracy {

Step5ValidationData collect_step5_cpu_validation(const PipelineState& state)
{
    Step5ValidationData data;
    const cusignal::RelativeExtremaShape shape({state.feature_bundle.size()});
    data.expected = cusignal::argrelextrema_typed_cpu(
        state.feature_bundle, shape, cusignal::RelativeExtremaComparator::greater,
        0, state.config.argrelextrema_order, "clip").coordinates.front();
    data.corrupted_expected = data.expected;
    if (!data.corrupted_expected.empty()) ++data.corrupted_expected.front();

    auto injected = state.feature_bundle;
    const float high = *std::max_element(injected.begin(), injected.end()) + 10.0F;
    injected[50] = high;
    data.injected_cpu = cusignal::argrelextrema_typed_cpu(
        injected, shape, cusignal::RelativeExtremaComparator::greater,
        0, state.config.argrelextrema_order, "clip").coordinates.front();

    injected[50] = state.feature_bundle[50];
    injected[70] = high;
    data.moved_cpu = cusignal::argrelextrema_typed_cpu(
        injected, shape, cusignal::RelativeExtremaComparator::greater,
        0, state.config.argrelextrema_order, "clip").coordinates.front();
    return data;
}

}  // namespace task2::accuracy
