from __future__ import annotations

import time

from task1_common import (
    PipelineState,
    StepEvidence,
    as_host,
    cp,
    cusignal,
    time_gpu,
    wall_ms,
)
from utils.waveforms import ensure_cusignal_chirp_compat


def _noise_bits(indices, seed: int):
    values = cp.uint32(seed) ^ (
        indices.astype(cp.uint32) * cp.uint32(747796405) + cp.uint32(2891336453)
    )
    values ^= values >> cp.uint32(16)
    values *= cp.uint32(2246822519)
    values ^= values >> cp.uint32(13)
    values *= cp.uint32(3266489917)
    return values ^ (values >> cp.uint32(16))


def _generate_waveform(config):
    pulse_width = config.pulse_samples / config.sample_rate_hz
    sample = cp.arange(config.pulse_samples, dtype=cp.float32)
    sample_time = sample / cp.float32(config.sample_rate_hz)
    # cuSignal linear complex chirp equals the centered Task1 LFM after applying
    # the constant phase pi*B*T/4 (degrees: 45*B*T).
    return cusignal.chirp(
        sample_time,
        f0=cp.float32(-0.5 * config.bandwidth_hz),
        t1=cp.float32(pulse_width),
        f1=cp.float32(0.5 * config.bandwidth_hz),
        method="linear",
        phi=cp.float32(45.0 * config.bandwidth_hz * pulse_width),
        type="complex",
    ).astype(cp.complex64, copy=False)


def _simulate_echo(waveform, delay: int, doppler_hz: float, seed: int, config):
    pulse = cp.arange(config.num_pulses, dtype=cp.float32)[:, None]
    sample_i32 = cp.arange(config.samples_per_pulse, dtype=cp.int32)[None, :]
    source = sample_i32 - cp.int32(delay)
    valid = (source >= 0) & (source < config.pulse_samples)
    source_clipped = cp.clip(source, 0, config.pulse_samples - 1)
    base = waveform[source_clipped]
    sample = sample_i32.astype(cp.float32)
    sample_time = pulse / cp.float32(config.prf_hz) + sample / cp.float32(config.sample_rate_hz)
    phase = cp.float32(2.0 * cp.pi * doppler_hz) * sample_time
    modulation = cp.exp(cp.complex64(1j) * phase).astype(cp.complex64)
    noiseless = cp.where(
        valid, cp.complex64(config.target_amplitude) * base * modulation, cp.complex64(0.0)
    ).astype(cp.complex64)
    count = config.num_pulses * config.samples_per_pulse
    component_indices = cp.arange(2 * count, dtype=cp.uint32)
    bits = _noise_bits(component_indices, seed)
    components = (
        ((bits & cp.uint32(0xFFFF)).astype(cp.float32) / cp.float32(32767.5) - 1.0)
        * cp.float32(config.noise_std)
    ).reshape(count, 2)
    noise = (components[:, 0] + cp.complex64(1j) * components[:, 1]).reshape(
        config.num_pulses, config.samples_per_pulse
    ).astype(cp.complex64)
    return noiseless, noise, (noiseless + noise).astype(cp.complex64)


def run_step1(state: PipelineState) -> StepEvidence:
    config = state.config
    evidence = StepEvidence("step1")
    formal_begin = time.perf_counter()
    prep_begin = time.perf_counter()
    doppler_hz = config.doppler_bin * config.prf_hz / config.num_pulses
    evidence.prep_ms = wall_ms(prep_begin)
    compat_begin = time.perf_counter()
    ensure_cusignal_chirp_compat()
    compat_ms = wall_ms(compat_begin)
    d_waveform, lfm_ms = time_gpu(lambda: _generate_waveform(config))
    outputs, echo_ms = time_gpu(
        lambda: _simulate_echo(d_waveform, config.target_delay_samples,
                               doppler_hz, config.noise_seed, config)
    )
    d_noiseless, d_noise, d_echo = outputs
    evidence.operator_ms = {
        "utils.cusignal_chirp_compat": compat_ms,
        "cusignal.chirp": lfm_ms,
        "task_scaffolding.delay_doppler_noise": echo_ms,
    }
    evidence.compute_ms = compat_ms + lfm_ms + echo_ms
    d2h_begin = time.perf_counter()
    state.waveform = as_host(d_waveform)
    state.noiseless_echo = as_host(d_noiseless)
    state.noise = as_host(d_noise)
    state.echo = as_host(d_echo)
    evidence.d2h_ms = wall_ms(d2h_begin)
    evidence.formal_execution_ms = wall_ms(formal_begin)
    return evidence
