#include "accuracy/validation/step6_validation.h"

#include <cusignal/operators/estimation/estimation_typed.h>

namespace task2::accuracy {

Step6ValidationData collect_step6_validation(const PipelineState& state)
{
    Step6ValidationData data;
    const cusignal::KalmanLayout layout(1, 1, 1);
    const std::vector<float> initial_x{0.0F}, initial_p{1.0F};
    const std::vector<float> f{state.config.kalman_f}, q{state.config.kalman_q}, alpha{1.0F};
    const std::vector<float> h{state.config.kalman_h}, r{state.config.kalman_r};
    auto cpu_state = cusignal::make_kalman_host_state(initial_x, initial_p, layout);
    for (float observation : state.kalman_observations) {
        const std::vector<float> z{observation};
        cusignal::kalman_predict_typed_cpu(cpu_state, f, q, alpha);
        cusignal::kalman_update_typed_cpu(cpu_state, h, r, z);
    }
    data.state_cpu = cpu_state.x;
    data.covariance_cpu = cpu_state.p;
    auto alternate = cusignal::make_kalman_host_state(initial_x, initial_p, layout);
    const std::vector<float> alternate_r{state.config.kalman_measurement_noise_variant};
    for (float observation : state.kalman_observations) {
        const std::vector<float> z{observation};
        cusignal::kalman_predict_typed_cpu(alternate, f, q, alpha);
        cusignal::kalman_update_typed_cpu(alternate, h, alternate_r, z);
    }
    data.alternate_estimate_cpu = alternate.x.front();
    return data;
}

}  // namespace task2::accuracy
