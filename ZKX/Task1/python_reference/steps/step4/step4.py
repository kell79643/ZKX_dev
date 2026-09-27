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
from cusignal.radartools import ca_cfar, cfar_alpha


def run_step4(state: PipelineState) -> StepEvidence:
    config = state.config
    guard_cells = (config.cfar_guard_doppler, config.cfar_guard_range)
    reference_cells = (config.cfar_reference_doppler, config.cfar_reference_range)
    outer = tuple(g + r for g, r in zip(guard_cells, reference_cells))
    reference_count = ((2 * outer[0] + 1) * (2 * outer[1] + 1)
                       - (2 * guard_cells[0] + 1) * (2 * guard_cells[1] + 1))
    evidence = StepEvidence("step4")
    formal_begin = time.perf_counter()
    prep_begin = time.perf_counter()
    require(
        state.range_doppler is not None
        and state.range_doppler.shape == (config.num_pulses, config.samples_per_pulse),
        "step4 did not receive the step3 range-Doppler map",
    )
    evidence.prep_ms = wall_ms(prep_begin)
    h2d_begin = time.perf_counter()
    d_range_doppler = cp.asarray(state.range_doppler, dtype=cp.complex64)
    synchronize()
    evidence.h2d_ms = wall_ms(h2d_begin)
    d_power, power_ms = time_gpu(
        lambda: (cp.abs(d_range_doppler) ** 2).astype(cp.float32)
    )
    alpha_begin = time.perf_counter()
    state.cfar_alpha = float(cfar_alpha(config.pfa, reference_count))
    alpha_ms = wall_ms(alpha_begin)
    outputs, cfar_ms = time_gpu(
        lambda: ca_cfar(
            d_power, guard_cells, reference_cells, pfa=config.pfa
        )
    )
    d_threshold, d_detections = outputs
    evidence.operator_ms = {
        "cupy.range_doppler_power": power_ms,
        "cusignal.cfar_alpha_host": alpha_ms,
        "cusignal.ca_cfar": cfar_ms,
    }
    evidence.compute_ms = power_ms + alpha_ms + cfar_ms
    d2h_begin = time.perf_counter()
    state.power = as_host(d_power)
    state.cfar_threshold = as_host(d_threshold)
    state.detections = as_host(d_detections)
    evidence.d2h_ms = wall_ms(d2h_begin)
    evidence.formal_execution_ms = wall_ms(formal_begin)
    return evidence
