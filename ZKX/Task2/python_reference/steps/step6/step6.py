from __future__ import annotations

import time

import numpy as np

from task2_common import (
    PipelineState,
    StepEvidence,
    as_host,
    cp,
    cusignal,
    require,
    synchronize,
    time_gpu,
    wall_ms,
)


def _observations_from_feature_point(bundle: np.ndarray, anchor: int, count: int) -> np.ndarray:
    observations: list[float] = []
    for radius in range(1, count + 1):
        left = max(0, anchor - radius)
        right = min(bundle.size - 1, anchor + radius)
        indices = np.arange(left, right + 1, dtype=np.float64)
        weights = np.maximum(bundle[left : right + 1].astype(np.float64), 0.0)
        coordinate = float(np.sum(indices * weights) / np.sum(weights)) if np.sum(weights) > 0 else float(anchor)
        observations.append(coordinate / (bundle.size - 1))
    return np.asarray(observations, dtype=np.float32)


def _estimate_with_noise(observations: np.ndarray, measurement_noise: float, config) -> float:
    kalman = cusignal.KalmanFilter(dim_x=1, dim_z=1, points=1, dtype=cp.float32)
    kalman.x[...] = cp.asarray([[[0.0]]], dtype=cp.float32)
    kalman.P[...] = cp.asarray([[[1.0]]], dtype=cp.float32)
    kalman.F[...] = cp.asarray([[[config.kalman_F]]], dtype=cp.float32)
    kalman.Q[...] = cp.asarray([[[config.kalman_Q]]], dtype=cp.float32)
    kalman.H[...] = cp.asarray([[[config.kalman_H]]], dtype=cp.float32)
    kalman.R[...] = cp.asarray([[[measurement_noise]]], dtype=cp.float32)
    for observation in observations:
        kalman.predict()
        kalman.update(cp.asarray([[[observation]]], dtype=cp.float32))
    return float(as_host(kalman.x).reshape(-1)[0])


def run_step6(state: PipelineState) -> StepEvidence:
    config = state.config
    evidence = StepEvidence("step6")
    total_begin = time.perf_counter()
    formal_begin = time.perf_counter()
    prep_begin = time.perf_counter()
    require(state.extrema is not None and state.extrema.size > 0, "step6 did not receive step5 points")
    strongest_position = int(np.argmax(state.feature_bundle[state.extrema]))
    state.kalman_anchor = int(state.extrema[strongest_position])
    state.kalman_observations = _observations_from_feature_point(
        state.feature_bundle, state.kalman_anchor, config.kalman_observation_count
    )
    state.truth = float(
        (0.5 * (state.feature_bundle.size - 1) + config.target_delay_samples)
        / (state.feature_bundle.size - 1)
    )
    kalman = cusignal.KalmanFilter(dim_x=1, dim_z=1, points=1, dtype=cp.float32)
    evidence.prep_ms = wall_ms(prep_begin)
    h2d_begin = time.perf_counter()
    kalman.x[...] = cp.asarray([[[0.0]]], dtype=cp.float32)
    kalman.P[...] = cp.asarray([[[1.0]]], dtype=cp.float32)
    kalman.F[...] = cp.asarray([[[config.kalman_F]]], dtype=cp.float32)
    kalman.Q[...] = cp.asarray([[[config.kalman_Q]]], dtype=cp.float32)
    kalman.H[...] = cp.asarray([[[config.kalman_H]]], dtype=cp.float32)
    kalman.R[...] = cp.asarray([[[config.kalman_R]]], dtype=cp.float32)
    d_observations = cp.asarray(state.kalman_observations, dtype=cp.float32)
    synchronize()
    evidence.h2d_ms = wall_ms(h2d_begin)
    if hasattr(kalman, "predict_update_sequence"):
        _, kalman_ms = time_gpu(
            lambda: kalman.predict_update_sequence(d_observations))
    else:
        # 非 ZQ500 scalar adapter 的兼容路径仍保持公开 predict/update 语义。
        kalman_ms = 0.0
        for observation in state.kalman_observations:
            d_observation = cp.asarray([[[observation]]], dtype=cp.float32)

            def predict_update():
                kalman.predict()
                kalman.update(d_observation)

            _, iteration_ms = time_gpu(predict_update)
            kalman_ms += iteration_ms
    evidence.operator_ms = {"cusignal.KalmanFilter.predict_update": kalman_ms}
    evidence.compute_ms = kalman_ms
    d2h_begin = time.perf_counter()
    estimated_state = as_host(kalman.x)
    state.kalman_covariance = as_host(kalman.P).reshape(-1).astype(np.float32)
    evidence.d2h_ms = wall_ms(d2h_begin)
    evidence.formal_execution_ms = wall_ms(formal_begin)
    post_begin = time.perf_counter()
    state.estimate = float(estimated_state.reshape(-1)[0])
    estimate_error = abs(state.estimate - state.truth)
    require(np.isfinite(state.estimate), "step6 estimate is non-finite")
    require(np.all(np.isfinite(state.kalman_covariance)), "step6 covariance is non-finite")
    require(estimate_error <= 0.18, "step6 estimate error exceeds approved threshold")
    high_noise_estimate = _estimate_with_noise(
        state.kalman_observations, config.kalman_measurement_noise_variant, config
    )
    measurement_noise_response = abs(high_noise_estimate - state.estimate)
    require(np.isfinite(high_noise_estimate), "step6 noise perturbation is non-finite")
    require(
        measurement_noise_response > 1.0e-6,
        "step6 measurement-noise perturbation did not change estimate",
    )
    evidence.metrics = {
        "kalmanfilter_called": 1.0,
        "step5_to_step6_data_match": 1.0,
        "kalman_anchor": float(state.kalman_anchor),
        "observation_count": float(state.kalman_observations.size),
        "estimate": state.estimate,
        "truth": state.truth,
        "estimate_abs_error": estimate_error,
        "estimate_error_limit": 0.18,
        "final_covariance": float(state.kalman_covariance[0]),
        "high_measurement_noise_estimate": high_noise_estimate,
        "measurement_noise_response": measurement_noise_response,
        "perturbation_checks_pass": 1.0,
    }
    evidence.output_shape = [1]
    evidence.output_dtype = "float32"
    evidence.post_ms = wall_ms(post_begin)
    evidence.total_ms = wall_ms(total_begin)
    return evidence
