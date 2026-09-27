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
from utils.windows import task1_hamming_window_fp32


def run_step3(state: PipelineState) -> StepEvidence:
    config = state.config
    evidence = StepEvidence("step3")
    formal_begin = time.perf_counter()
    prep_begin = time.perf_counter()
    require(
        state.compressed is not None
        and state.compressed.shape == (config.num_pulses, config.samples_per_pulse),
        "step3 did not receive the step2 compressed signal",
    )
    evidence.prep_ms = wall_ms(prep_begin)
    h2d_begin = time.perf_counter()
    d_compressed = cp.asarray(state.compressed, dtype=cp.complex64)
    synchronize()
    evidence.h2d_ms = wall_ms(h2d_begin)
    d_window, window_ms = time_gpu(
        lambda: task1_hamming_window_fp32(config.num_pulses)
    )
    d_range_doppler, operator_ms = time_gpu(
        lambda: cusignal.pulse_doppler(
            d_compressed, window=d_window, nfft=config.num_pulses
        ).astype(cp.complex64, copy=False)
    )
    evidence.operator_ms = {
        "utils.hamming_fp32_adapter": window_ms,
        "cusignal.pulse_doppler": operator_ms,
    }
    evidence.compute_ms = window_ms + operator_ms
    d2h_begin = time.perf_counter()
    state.range_doppler = as_host(d_range_doppler)
    evidence.d2h_ms = wall_ms(d2h_begin)
    evidence.formal_execution_ms = wall_ms(formal_begin)
    return evidence
