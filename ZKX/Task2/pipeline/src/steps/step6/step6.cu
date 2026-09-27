#include <task2/steps/step6.h>

#include <cusignal/runtime/device_array.h>
#include <cusignal/operators/estimation/estimation_typed.h>

namespace task2 {
namespace {

std::vector<float> observations_from_feature_point(
    const std::vector<float>& bundle, std::int64_t anchor, int observation_count)
{
    std::vector<float> observations;
    observations.reserve(observation_count);
    for (int radius = 1; radius <= observation_count; ++radius) {
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

void prepare_step6_host_inputs(PipelineState& state)
{
    const auto& config = state.config;
    constexpr std::size_t kSplineTapCount = 9;
    require_task2_upstream(!state.extrema.empty(), "Step6", "Step5.extrema");
    require_task2_upstream(
        !state.feature_bundle.empty(), "Step6", "Step4.feature_bundle");
    require_task2_upstream(
        state.filter_taps.empty() ||
            state.filter_taps.size() == static_cast<std::size_t>(config.filter_taps),
        "Step6", "Step2.filter_taps");
    require_task2_upstream(
        state.spline_weights.empty() || state.spline_weights.size() == kSplineTapCount,
        "Step6", "Step3.spline_weights");
    const auto strongest = *std::max_element(
        state.extrema.begin(), state.extrema.end(),
        [&](std::int64_t left, std::int64_t right) {
            return state.feature_bundle[static_cast<std::size_t>(left)] <
                   state.feature_bundle[static_cast<std::size_t>(right)];
        });
    state.kalman_anchor = strongest;
    state.kalman_observations = observations_from_feature_point(
        state.feature_bundle, strongest, config.kalman_observation_count);
    const float filter_delay_samples =
        (static_cast<float>(config.filter_taps) - 1.0F) / 2.0F;
    const float spline_delay_samples =
        (static_cast<float>(kSplineTapCount) - 1.0F) / 2.0F;
    const float bundle_scale =
        static_cast<float>(state.feature_bundle.size() - 1);
    state.truth =
        (0.5F * bundle_scale + static_cast<float>(config.target_delay_samples) +
         filter_delay_samples + spline_delay_samples) /
        bundle_scale;
}

namespace {

StepEvidence run_step6_impl(PipelineState& state, bool prepare_inputs)
{
    const auto& config = state.config;
    StepEvidence evidence;
    evidence.name = "step6";
    const auto total_begin = Clock::now();
    auto begin = Clock::now();
    if (prepare_inputs) prepare_step6_host_inputs(state);
    else require_task2_upstream(!state.kalman_observations.empty(),
            "Step6", "prepared.kalman_observations");
    const cusignal::KalmanLayout layout(1, 1, 1);
    const std::vector<float> initial_x{0.0F};
    const std::vector<float> initial_p{1.0F};
    const std::vector<float> f{config.kalman_f}, q{config.kalman_q}, alpha{1.0F};
    const std::vector<float> h{config.kalman_h}, r{config.kalman_r};
    evidence.prep_ms = milliseconds(begin, Clock::now());

    begin = Clock::now();
    auto d_initial_x = cusignal::DeviceArray<float>::from_host(initial_x);
    auto d_initial_p = cusignal::DeviceArray<float>::from_host(initial_p);
    auto d_f = cusignal::DeviceArray<float>::from_host(f);
    auto d_q = cusignal::DeviceArray<float>::from_host(q);
    auto d_alpha = cusignal::DeviceArray<float>::from_host(alpha);
    auto d_h = cusignal::DeviceArray<float>::from_host(h);
    auto d_r = cusignal::DeviceArray<float>::from_host(r);
    auto d_observations = cusignal::DeviceArray<float>::from_host(
        state.kalman_observations);
    evidence.h2d_ms = milliseconds(begin, Clock::now());
    cusignal::KalmanDeviceState device_state(layout);
    cusignal::initialize_kalman_device_state(d_initial_x, d_initial_p, device_state);
    const float kalman_gpu_ms = time_gpu([&] {
        cusignal::kalman_predict_update_scalar_sequence_device(
            device_state, d_f, d_q, d_alpha, d_h, d_r, d_observations);
    });
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    evidence.metrics["gpu_used_peak_bytes"] =
        static_cast<double>(total_bytes - free_bytes);
    evidence.operator_ms["kalmanfilter_predict_update"] = kalman_gpu_ms;
    evidence.compute_ms = kalman_gpu_ms;
    evidence.metrics["observation_h2d_batch_count"] = 1.0;
    evidence.metrics["sequence_kernel_count"] = 1.0;
    evidence.metrics["per_observation_event_sync_count"] = 0.0;
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

}  // namespace

StepEvidence run_step6_prepared(PipelineState& state)
{
    return run_step6_impl(state, false);
}

StepEvidence run_step6(PipelineState& state)
{
    return run_step6_impl(state, true);
}

}  // namespace task2
