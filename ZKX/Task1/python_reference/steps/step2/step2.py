from __future__ import annotations

import time

from task1_common import (
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
    formal_begin = time.perf_counter()
    prep_begin = time.perf_counter()
    require(state.echo is not None and state.echo.shape == (config.num_pulses, config.samples_per_pulse),
            "step2 did not receive the step1 echo")
    require(state.waveform is not None and state.waveform.size == config.pulse_samples,
            "step2 did not receive the step1 waveform")
    evidence.prep_ms = wall_ms(prep_begin)
    h2d_begin = time.perf_counter()
    d_echo = cp.asarray(state.echo, dtype=cp.complex64)
    d_waveform = cp.asarray(state.waveform, dtype=cp.complex64)
    synchronize()
    evidence.h2d_ms = wall_ms(h2d_begin)
    d_compressed, operator_ms = time_gpu(
        lambda: cusignal.pulse_compression(
            d_echo, d_waveform, normalize=True, window=None, nfft=config.samples_per_pulse
        ).astype(cp.complex64, copy=False)
    )
    evidence.operator_ms["cusignal.pulse_compression"] = operator_ms
    evidence.compute_ms = operator_ms
    d2h_begin = time.perf_counter()
    state.compressed = as_host(d_compressed)
    evidence.d2h_ms = wall_ms(d2h_begin)
    evidence.formal_execution_ms = wall_ms(formal_begin)
    return evidence
