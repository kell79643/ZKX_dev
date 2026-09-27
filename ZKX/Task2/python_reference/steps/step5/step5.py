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


def run_step5(state: PipelineState) -> StepEvidence:
    order = state.config.argrelextrema_order
    evidence = StepEvidence("step5")
    total_begin = time.perf_counter()
    formal_begin = time.perf_counter()
    prep_begin = time.perf_counter()
    require(state.feature_bundle is not None and state.feature_bundle.size > 0, "step5 did not receive step4 bundle")
    evidence.prep_ms = wall_ms(prep_begin)
    h2d_begin = time.perf_counter()
    d_bundle = cp.asarray(state.feature_bundle, dtype=cp.float32)
    synchronize()
    evidence.h2d_ms = wall_ms(h2d_begin)
    result, extrema_ms = time_gpu(
        lambda: cusignal.argrelextrema(
            d_bundle, cp.greater, axis=0, order=order, mode="clip"
        )
    )
    require(len(result) == 1, "step5 coordinate rank mismatch")
    evidence.operator_ms = {"cusignal.argrelextrema": extrema_ms}
    evidence.compute_ms = extrema_ms
    d2h_begin = time.perf_counter()
    state.extrema = as_host(result[0]).astype(np.int64)
    evidence.d2h_ms = wall_ms(d2h_begin)
    evidence.formal_execution_ms = wall_ms(formal_begin)
    post_begin = time.perf_counter()
    require(state.extrema.size > 0, "step5 found no feature points")
    require(np.all(np.diff(state.extrema) > 0), "step5 extrema are not ordered unique indices")
    require(
        int(state.extrema[0]) >= 0 and int(state.extrema[-1]) < state.feature_bundle.size,
        "step5 extrema index out of bounds",
    )
    injection_index = 50
    moved_index = 70
    injected = state.feature_bundle.copy()
    injected[injection_index - order : injection_index + order + 1] = 0.0
    injected[injection_index] = float(np.max(state.feature_bundle) + 1.0)
    d_injected = cp.asarray(injected, dtype=cp.float32)
    injected_result = cusignal.argrelextrema(
        d_injected, cp.greater, axis=0, order=order, mode="clip"
    )
    injected_indices = as_host(injected_result[0]).astype(np.int64)
    moved = injected.copy()
    moved[injection_index] = 0.0
    moved[moved_index - order : moved_index + order + 1] = 0.0
    moved[moved_index] = float(np.max(state.feature_bundle) + 1.0)
    d_moved = cp.asarray(moved, dtype=cp.float32)
    moved_result = cusignal.argrelextrema(
        d_moved, cp.greater, axis=0, order=order, mode="clip"
    )
    moved_indices = as_host(moved_result[0]).astype(np.int64)
    require(injection_index in injected_indices, "step5 injected peak was not detected")
    require(moved_index in moved_indices, "step5 moved peak was not detected")
    require(injection_index not in moved_indices, "step5 old injected peak remained after movement")
    evidence.metrics = {
        "argrelextrema_called": 1.0,
        "step4_to_step5_data_match": 1.0,
        "comparator_greater": 1.0,
        "axis": 0.0,
        "order": float(order),
        "mode_clip": 1.0,
        "extrema_count": float(state.extrema.size),
        "injected_peak_index": float(injection_index),
        "moved_peak_index": float(moved_index),
        "injected_peak_followed": 1.0,
        "moved_peak_followed": 1.0,
        "perturbation_checks_pass": 1.0,
    }
    evidence.output_shape = list(state.extrema.shape)
    evidence.output_dtype = str(state.extrema.dtype)
    evidence.post_ms = wall_ms(post_begin)
    evidence.total_ms = wall_ms(total_begin)
    return evidence
