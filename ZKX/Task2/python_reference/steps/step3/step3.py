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
    roughness,
    synchronize,
    time_gpu,
    wall_ms,
)


def run_step3(state: PipelineState) -> StepEvidence:
    config = state.config
    evidence = StepEvidence("step3")
    total_begin = time.perf_counter()
    formal_begin = time.perf_counter()
    prep_begin = time.perf_counter()
    require(state.filtered is not None and state.filtered.size == config.samples, "step3 did not receive step2 output")
    crop = config.step3_crop_each
    cropped = state.filtered[crop:-crop].astype(np.float32, copy=True)
    offsets = np.arange(-2.0, 2.01, 0.5, dtype=np.float32)
    evidence.prep_ms = wall_ms(prep_begin)
    h2d_begin = time.perf_counter()
    d_cropped = cp.asarray(cropped)
    d_offsets = cp.asarray(offsets)
    synchronize()
    evidence.h2d_ms = wall_ms(h2d_begin)
    d_cubic, cubic_ms = time_gpu(lambda: cusignal.cubic(d_offsets))
    d_weights, normalization_ms = time_gpu(lambda: d_cubic / cp.sum(d_cubic))
    d_smoothed, filter_ms = time_gpu(lambda: cusignal.firfilter(d_weights, d_cropped, axis=-1))
    evidence.operator_ms = {
        "cusignal.cubic": cubic_ms,
        "cupy.cubic_normalization": normalization_ms,
        "cusignal.firfilter_b_spline": filter_ms,
    }
    evidence.compute_ms = sum(evidence.operator_ms.values())
    d2h_begin = time.perf_counter()
    state.smoothed = as_host(d_smoothed).astype(np.float32)
    evidence.d2h_ms = wall_ms(d2h_begin)
    handoff_begin = time.perf_counter()
    state.reference_for_correlation = state.target_waveform[
        crop:-crop
    ].astype(np.float32, copy=True)
    evidence.post_ms = wall_ms(handoff_begin)
    evidence.formal_execution_ms = wall_ms(formal_begin)
    post_begin = time.perf_counter()
    rough_before = roughness(cropped)
    rough_after = roughness(state.smoothed)
    peak_ratio = float(
        np.max(np.abs(state.smoothed)) / max(np.max(np.abs(cropped)), 1.0e-12)
    )
    require(rough_after < rough_before, "step3 did not reduce roughness")
    require(peak_ratio > 0.20, "step3 over-smoothed the target structure")
    evidence.metrics = {
        "bsplines_functions": 1.0,
        "step2_to_step3_data_match": 1.0,
        "crop_begin": float(crop),
        "crop_end": float(config.samples - crop),
        "roughness_before": rough_before,
        "roughness_after": rough_after,
        "peak_preservation_ratio": peak_ratio,
    }
    evidence.output_shape = list(state.smoothed.shape)
    evidence.output_dtype = str(state.smoothed.dtype)
    evidence.post_ms += wall_ms(post_begin)
    evidence.total_ms = wall_ms(total_begin)
    return evidence
