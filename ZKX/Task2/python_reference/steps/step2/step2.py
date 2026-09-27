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


def run_step2(state: PipelineState) -> StepEvidence:
    config = state.config
    evidence = StepEvidence("step2")
    total_begin = time.perf_counter()
    formal_begin = time.perf_counter()
    prep_begin = time.perf_counter()
    require(state.echo is not None and state.echo.size == config.samples, "step2 did not receive step1 echo")
    evidence.prep_ms = wall_ms(prep_begin)
    h2d_begin = time.perf_counter()
    d_echo = cp.asarray(state.echo, dtype=cp.float32)
    synchronize()
    evidence.h2d_ms = wall_ms(h2d_begin)
    d_window, window_ms = time_gpu(lambda: cusignal.hamming(config.filter_taps, sym=True))
    d_taps, firwin_ms = time_gpu(
        lambda: cusignal.firwin(
            config.filter_taps, config.filter_cutoff_hz, window="hamming", pass_zero=True,
            scale=True, fs=config.sample_rate_hz, gpupath=True,
        ).astype(cp.float32)
    )
    d_filtered, filter_ms = time_gpu(lambda: cusignal.firfilter(d_taps, d_echo, axis=-1))
    evidence.operator_ms = {
        "cusignal.hamming": window_ms,
        "cusignal.firwin": firwin_ms,
        "cusignal.firfilter": filter_ms,
    }
    evidence.compute_ms = sum(evidence.operator_ms.values())
    d2h_begin = time.perf_counter()
    state.filter_taps = as_host(d_taps).astype(np.float32)
    state.filtered = as_host(d_filtered).astype(np.float32)
    window_host = as_host(d_window)
    evidence.d2h_ms = wall_ms(d2h_begin)
    evidence.formal_execution_ms = wall_ms(formal_begin)
    post_begin = time.perf_counter()
    require(state.filtered.size == config.samples, "step2 output length mismatch")
    require(np.all(np.isfinite(state.filtered)), "step2 produced non-finite output")
    target_filtered = np.convolve(state.target_component, state.filter_taps, mode="full")[:config.samples]
    residual = state.echo - state.target_component
    residual_filtered = np.convolve(residual, state.filter_taps, mode="full")[:config.samples]
    before_snr = 20.0 * np.log10(
        max(np.sqrt(np.mean(state.target_component.astype(np.float64) ** 2)), 1.0e-12)
        / max(np.sqrt(np.mean(residual.astype(np.float64) ** 2)), 1.0e-12)
    )
    after_snr = 20.0 * np.log10(
        max(np.sqrt(np.mean(target_filtered.astype(np.float64) ** 2)), 1.0e-12)
        / max(np.sqrt(np.mean(residual_filtered.astype(np.float64) ** 2)), 1.0e-12)
    )
    require(after_snr - before_snr >= 1.0, "step2 SNR improvement is below 1 dB")
    evidence.metrics = {
        "filter_design_functions": 1.0,
        "filtering_functions": 1.0,
        "window_functions": 1.0,
        "step1_to_step2_data_match": 1.0,
        "taps": float(config.filter_taps),
        "cutoff_hz": config.filter_cutoff_hz,
        "window_nonzero": float(np.max(np.abs(window_host)) > 0.0),
        "snr_before_db": float(before_snr),
        "snr_after_db": float(after_snr),
        "snr_improvement_db": float(after_snr - before_snr),
    }
    evidence.output_shape = list(state.filtered.shape)
    evidence.output_dtype = str(state.filtered.dtype)
    evidence.post_ms = wall_ms(post_begin)
    evidence.total_ms = wall_ms(total_begin)
    return evidence
