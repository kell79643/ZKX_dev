#include "steps/step5.h"

#include <cusignal/runtime/device_array.h>
#include <cusignal/operators/peak_finding/peak_finding_typed.h>

namespace task2 {

StepEvidence run_step5(PipelineState& state)
{
    StepEvidence evidence;
    evidence.name = "step5";
    const auto total_begin = Clock::now();
    auto begin = Clock::now();
    require(!state.feature_bundle.empty(), "step5 did not receive step4 bundle");
    const cusignal::RelativeExtremaShape shape({state.feature_bundle.size()});
    cusignal::RelativeExtremaDeviceWorkspace workspace(state.feature_bundle.size());
    evidence.prep_ms = milliseconds(begin, Clock::now());
    begin = Clock::now();
    auto d_bundle = cusignal::DeviceArray<float>::from_host(state.feature_bundle);
    evidence.h2d_ms = milliseconds(begin, Clock::now());
    cusignal::RelativeExtremaDeviceResult result;
    evidence.operator_ms["argrelextrema"] = time_gpu([&] {
        result = cusignal::argrelextrema_device(
            d_bundle, shape, cusignal::RelativeExtremaComparator::greater,
            workspace, 0, 3, "clip");
    });
    evidence.compute_ms = evidence.operator_ms["argrelextrema"];
    begin = Clock::now();
    require(result.coordinates.size() == 1, "step5 coordinate rank mismatch");
    state.extrema = result.coordinates.front().to_host();
    evidence.d2h_ms = milliseconds(begin, Clock::now());

    evidence.total_ms = milliseconds(total_begin, Clock::now());
    return evidence;
}

}  // namespace task2
