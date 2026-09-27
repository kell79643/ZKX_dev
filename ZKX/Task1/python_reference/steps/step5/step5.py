from __future__ import annotations

import time

from task1_common import (
    PipelineState,
    StepEvidence,
    as_host,
    cp,
    require,
    synchronize,
    time_gpu,
    wall_ms,
)
from utils.radartools import task1_ambgfun_2d, task1_ambiguity_cut


def run_step5(state: PipelineState) -> StepEvidence:
    config = state.config
    evidence = StepEvidence("step5")
    formal_begin = time.perf_counter()
    prep_begin = time.perf_counter()
    require(state.waveform is not None and state.waveform.size == config.pulse_samples,
            "step5 did not receive the step1 transmit waveform")
    evidence.prep_ms = wall_ms(prep_begin)
    h2d_begin = time.perf_counter()
    d_waveform = cp.asarray(state.waveform, dtype=cp.complex64)
    synchronize()
    evidence.h2d_ms = wall_ms(h2d_begin)
    d_ambiguity_2d, two_d_ms = time_gpu(
        lambda: task1_ambgfun_2d(d_waveform, config.sample_rate_hz, config.prf_hz)
    )
    d_delay_source, delay_api_ms = time_gpu(
        lambda: task1_ambgfun_2d(d_waveform, config.sample_rate_hz, config.prf_hz)
    )
    d_ambiguity_delay, delay_adapter_ms = time_gpu(
        lambda: task1_ambiguity_cut(
            d_delay_source, config.pulse_samples, config.sample_rate_hz, "doppler", 0
        )
    )
    d_doppler_source, doppler_api_ms = time_gpu(
        lambda: task1_ambgfun_2d(d_waveform, config.sample_rate_hz, config.prf_hz)
    )
    d_ambiguity_doppler, doppler_adapter_ms = time_gpu(
        lambda: task1_ambiguity_cut(
            d_doppler_source, config.pulse_samples, config.sample_rate_hz, "delay", 0
        )
    )
    evidence.operator_ms = {
        "cusignal.ambgfun_2d": two_d_ms,
        "cusignal.ambgfun_2d_for_doppler_cut_adapter": delay_api_ms,
        "utils.ambgfun_doppler_cut_adapter": delay_adapter_ms,
        "cusignal.ambgfun_2d_for_delay_cut_adapter": doppler_api_ms,
        "utils.ambgfun_delay_cut_adapter": doppler_adapter_ms,
    }
    evidence.compute_ms = sum(evidence.operator_ms.values())
    d2h_begin = time.perf_counter()
    state.ambiguity_2d = as_host(d_ambiguity_2d)
    state.ambiguity_delay = as_host(d_ambiguity_delay)
    state.ambiguity_doppler = as_host(d_ambiguity_doppler)
    evidence.d2h_ms = wall_ms(d2h_begin)
    evidence.formal_execution_ms = wall_ms(formal_begin)
    return evidence
