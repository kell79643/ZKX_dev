#include "steps/step6.h"

#include <cusignal/runtime/device_array.h>
#include <cusignal/operators/estimation/estimation_typed.h>

namespace task2 {
namespace {

std::vector<float> observations_from_feature_point(
    const std::vector<float>& bundle, std::int64_t anchor)
{
    std::vector<float> observations;
    observations.reserve(kKalmanObservationCount);
    for (int radius = 1; radius <= kKalmanObservationCount; ++radius) {
        const int left = std::max(0, static_cast<int>(anchor) - radius);
        const int right = std::min(static_cast<int>(bundle.size()) - 1,
                                   static_cast<int>(anchor) + radius);
        double weighted_index = 0.0;
        double weight = 0.0;
        for (int index = left; index <= right; ++index) {
            const double local_weight = std::max(0.0F, bundle[index]);
            weighted_index += local_weight * index;
            weight += local_weight;
        }
        const double coordinate = weight > 0.0 ? weighted_index / weight : anchor;
        observations.push_back(static_cast<float>(coordinate / (bundle.size() - 1)));
    }
    return observations;
}

}  // namespace

StepEvidence run_step6(PipelineState& state)
{
    StepEvidence evidence;
    evidence.name = "step6";
    const auto total_begin = Clock::now();
    auto begin = Clock::now();
    require(!state.extrema.empty(), "step6 did not receive step5 points");
    const auto strongest = *std::max_element(
        state.extrema.begin(), state.extrema.end(),
        [&](std::int64_t left, std::int64_t right) {
            return state.feature_bundle[static_cast<std::size_t>(left)] <
                   state.feature_bundle[static_cast<std::size_t>(right)];
        });
    state.kalman_anchor = strongest;
    state.kalman_observations = observations_from_feature_point(
        state.feature_bundle, strongest);
    state.truth = (0.5F * static_cast<float>(state.feature_bundle.size() - 1) +
                   static_cast<float>(kTargetDelay)) /
                  static_cast<float>(state.feature_bundle.size() - 1);
    const cusignal::KalmanLayout layout(1, 1, 1);
    const std::vector<float> initial_x{0.0F};
    const std::vector<float> initial_p{1.0F};
    const std::vector<float> f{1.0F}, q{1.0e-4F}, alpha{1.0F};
    const std::vector<float> h{1.0F}, r{kKalmanR};
    evidence.prep_ms = milliseconds(begin, Clock::now());

    begin = Clock::now();
    auto d_initial_x = cusignal::DeviceArray<float>::from_host(initial_x);
    auto d_initial_p = cusignal::DeviceArray<float>::from_host(initial_p);
    auto d_f = cusignal::DeviceArray<float>::from_host(f);
    auto d_q = cusignal::DeviceArray<float>::from_host(q);
    auto d_alpha = cusignal::DeviceArray<float>::from_host(alpha);
    auto d_h = cusignal::DeviceArray<float>::from_host(h);
    auto d_r = cusignal::DeviceArray<float>::from_host(r);
    evidence.h2d_ms = milliseconds(begin, Clock::now());
    cusignal::KalmanDeviceState device_state(layout);
    cusignal::KalmanDeviceWorkspace workspace(layout);
    cusignal::initialize_kalman_device_state(d_initial_x, d_initial_p, device_state);
    float kalman_gpu_ms = 0.0F;
    for (float observation : state.kalman_observations) {
        const std::vector<float> z{observation};
        begin = Clock::now();
        auto d_z = cusignal::DeviceArray<float>::from_host(z);
        evidence.h2d_ms += milliseconds(begin, Clock::now());
        kalman_gpu_ms += time_gpu([&] {
            cusignal::kalman_predict_device(
                device_state, d_f, d_q, d_alpha, workspace);
            cusignal::kalman_update_device(
                device_state, d_h, d_r, d_z, workspace);
        });
    }
    evidence.operator_ms["kalmanfilter_predict_update"] = kalman_gpu_ms;
    evidence.compute_ms = kalman_gpu_ms;
    begin = Clock::now();
    const auto estimated_state = device_state.x.to_host();
    state.kalman_covariance = device_state.p.to_host();
    evidence.d2h_ms = milliseconds(begin, Clock::now());

    begin = Clock::now();
    require(estimated_state.size() == 1 && state.kalman_covariance.size() == 1,
            "step6 state shape mismatch");
    state.estimate = estimated_state.front();
    evidence.post_ms = milliseconds(begin, Clock::now());
    evidence.total_ms = milliseconds(total_begin, Clock::now());
    return evidence;
}

}  // namespace task2
