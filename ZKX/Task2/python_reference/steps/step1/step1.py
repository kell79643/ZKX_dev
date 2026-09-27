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


def _noise_bits(indices, seed: int):
    values = cp.uint32(seed) ^ (
        indices.astype(cp.uint32) * cp.uint32(747796405) + cp.uint32(2891336453)
    )
    values ^= values >> cp.uint32(16)
    values *= cp.uint32(2246822519)
    values ^= values >> cp.uint32(13)
    values *= cp.uint32(3266489917)
    return values ^ (values >> cp.uint32(16))


def _mix_echo(
    chirp, gaussian, sawtooth, square, config,
    noise_seed: int | None = None, target_weight: float | None = None,
):
    seed = config.noise_seed if noise_seed is None else noise_seed
    weight = config.target_weight if target_weight is None else target_weight
    indices = cp.arange(config.samples, dtype=cp.int32)
    source = indices - config.target_delay_samples
    target = cp.where(
        source >= 0,
        cp.float32(weight) * chirp[cp.maximum(source, 0)],
        0.0,
    )
    interference = (
        cp.float32(config.gaussian_weight) * gaussian
        + cp.float32(config.sawtooth_weight) * sawtooth.astype(cp.float32)
        + cp.float32(config.square_weight) * square.astype(cp.float32)
    )
    bits = _noise_bits(indices.astype(cp.uint32), seed)
    noise = (
        ((bits & cp.uint32(0xFFFF)).astype(cp.float32) / cp.float32(32767.5) - 1.0)
        * cp.float32(config.noise_amplitude)
    )
    return (
        target.astype(cp.float32),
        interference.astype(cp.float32),
        noise.astype(cp.float32),
        (target + interference + noise).astype(cp.float32),
    )


def run_step1(state: PipelineState) -> StepEvidence:
    config = state.config
    evidence = StepEvidence("step1")
    total_begin = time.perf_counter()
    formal_begin = time.perf_counter()
    prep_begin = time.perf_counter()
    state.time = np.arange(config.samples, dtype=np.float32) / np.float32(config.sample_rate_hz)
    pulse_time = state.time - np.float32(0.45)
    sample_indices = np.arange(config.samples, dtype=np.float64)
    phase = (
        2.0 * np.pi * np.remainder(190.0 * sample_indices / config.sample_rate_hz, 1.0)
    ).astype(np.float32)
    square_phase = (
        2.0 * np.pi * np.remainder(125.0 * sample_indices / config.sample_rate_hz, 1.0)
    ).astype(np.float32)
    evidence.prep_ms = wall_ms(prep_begin)
    h2d_begin = time.perf_counter()
    d_time = cp.asarray(state.time)
    d_pulse_time = cp.asarray(pulse_time)
    d_phase = cp.asarray(phase)
    d_square_phase = cp.asarray(square_phase)
    synchronize()
    evidence.h2d_ms = wall_ms(h2d_begin)
    d_chirp, chirp_ms = time_gpu(
        lambda: cusignal.chirp(
            d_time, f0=20.0, t1=max(config.samples / config.sample_rate_hz, 1.0), f1=70.0, phi=0.0
        )
    )
    d_gaussian, gaussian_ms = time_gpu(
        lambda: cusignal.gausspulse(d_pulse_time, fc=170.0, bw=0.20)
    )
    d_saw, saw_ms = time_gpu(
        lambda: cusignal.sawtooth(d_phase, width=np.float32(0.65))
    )
    d_square, square_ms = time_gpu(
        lambda: cusignal.square(d_square_phase, duty=np.float32(0.35))
    )
    outputs, mix_ms = time_gpu(lambda: _mix_echo(d_chirp, d_gaussian, d_saw, d_square, config))
    d_target, d_interference, d_noise, d_echo = outputs
    evidence.operator_ms = {
        "cusignal.chirp": chirp_ms,
        "cusignal.gausspulse": gaussian_ms,
        "cusignal.sawtooth": saw_ms,
        "cusignal.square": square_ms,
        "cupy.echo_superposition": mix_ms,
    }
    evidence.compute_ms = sum(evidence.operator_ms.values())
    d2h_begin = time.perf_counter()
    state.target_waveform = as_host(d_chirp).astype(np.float32, copy=False)
    state.target_component = as_host(d_target)
    state.interference_component = as_host(d_interference)
    state.noise_component = as_host(d_noise)
    state.echo = as_host(d_echo)
    evidence.d2h_ms = wall_ms(d2h_begin)
    evidence.formal_execution_ms = wall_ms(formal_begin)
    post_begin = time.perf_counter()
    require(np.all(np.isfinite(state.echo)), "step1 produced non-finite echo")
    require(np.max(np.abs(state.target_component)) > 0.1, "step1 target component degenerate")
    require(np.max(np.abs(state.interference_component)) > 0.01, "step1 interference degenerate")
    require(np.max(np.abs(state.noise_component)) > 0.001, "step1 noise degenerate")
    perturbed_weight = _mix_echo(
        d_chirp, d_gaussian, d_saw, d_square, config,
        target_weight=max(0.0, config.target_weight - 0.07),
    )[3]
    perturbed_seed = _mix_echo(
        d_chirp, d_gaussian, d_saw, d_square, config, noise_seed=config.noise_seed + 1
    )[3]
    weight_delta = float(np.max(np.abs(as_host(perturbed_weight) - state.echo)))
    seed_delta = float(np.max(np.abs(as_host(perturbed_seed) - state.echo)))
    require(weight_delta > 1.0e-3, "step1 target-weight perturbation did not change output")
    require(seed_delta > 1.0e-3, "step1 seed perturbation did not change output")
    evidence.metrics = {
        "waveform_functions_selected": 4.0,
        "target_component": 1.0,
        "interference_component": 1.0,
        "noise_component": 1.0,
        "custom_waveform_superposition": 1.0,
        "sample_rate_hz": config.sample_rate_hz,
        "samples": float(config.samples),
        "target_delay_samples": float(config.target_delay_samples),
        "noise_seed": float(config.noise_seed),
        "echo_rms": float(np.sqrt(np.mean(state.echo.astype(np.float64) ** 2))),
        "weight_perturbation_delta": weight_delta,
        "seed_perturbation_delta": seed_delta,
        "perturbation_checks_pass": 1.0,
    }
    evidence.output_shape = list(state.echo.shape)
    evidence.output_dtype = str(state.echo.dtype)
    evidence.post_ms = wall_ms(post_begin)
    evidence.total_ms = wall_ms(total_begin)
    return evidence
