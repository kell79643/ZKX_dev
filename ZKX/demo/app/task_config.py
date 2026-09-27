"""Strict dependency-free parser shared by Task1/Task2 Python and demo tools."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def _parse(path: Path, expected: set[str]) -> tuple[dict[str, str], str]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"config is not UTF-8: {path}") from error
    values: dict[str, str] = {}
    for line_number, original in enumerate(text.splitlines(), 1):
        line = original.split("#", 1)[0].strip()
        if not line:
            continue
        if line.count("=") != 1:
            raise ValueError(f"{path}:{line_number}: expected one key=value")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or not value:
            raise ValueError(f"{path}:{line_number}: empty key or value")
        if key not in expected:
            raise ValueError(f"{path}:{line_number}: unknown field {key!r}")
        if key in values:
            raise ValueError(f"{path}:{line_number}: duplicate field {key!r}")
        values[key] = value
    missing = expected - values.keys()
    if missing:
        raise ValueError(f"{path}: missing fields: {', '.join(sorted(missing))}")
    return values, hashlib.sha256(raw).hexdigest()


def _integer(values: Mapping[str, str], key: str) -> int:
    value = values[key]
    if value.strip() != value or not value or any(ch not in "+-0123456789" for ch in value):
        raise ValueError(f"{key} is not a strict integer: {value!r}")
    try:
        return int(value, 10)
    except ValueError as error:
        raise ValueError(f"{key} is not an integer: {value!r}") from error


def _number(values: Mapping[str, str], key: str) -> float:
    try:
        value = float(values[key])
    except ValueError as error:
        raise ValueError(f"{key} is not numeric: {values[key]!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def _power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


@dataclass(frozen=True)
class Task1Config:
    path: Path
    sha256: str
    num_pulses: int
    samples_per_pulse: int
    pulse_samples: int
    sample_rate_hz: float
    prf_hz: float
    bandwidth_hz: float
    carrier_frequency_hz: float
    target_delay_samples: int
    doppler_bin: int
    target_amplitude: float
    noise_std: float
    noise_seed: int
    pfa: float
    cfar_guard_doppler: int
    cfar_guard_range: int
    cfar_reference_doppler: int
    cfar_reference_range: int
    ambiguity_nfreq: int

    @classmethod
    def load(cls, path: str | Path) -> "Task1Config":
        resolved = Path(path).expanduser().resolve(strict=True)
        fields = set(cls.__dataclass_fields__) - {"path", "sha256"}
        raw, digest = _parse(resolved, fields)
        integer_fields = {
            "num_pulses", "samples_per_pulse", "pulse_samples", "target_delay_samples",
            "doppler_bin", "noise_seed", "cfar_guard_doppler", "cfar_guard_range",
            "cfar_reference_doppler", "cfar_reference_range", "ambiguity_nfreq",
        }
        parsed = {key: _integer(raw, key) if key in integer_fields else _number(raw, key) for key in fields}
        result = cls(path=resolved, sha256=digest, **parsed)
        if not _power_of_two(result.num_pulses) or not _power_of_two(result.samples_per_pulse):
            raise ValueError("Task1 FFT dimensions must be powers of two")
        if not (1 <= result.pulse_samples <= result.samples_per_pulse):
            raise ValueError("pulse_samples must be in [1, samples_per_pulse]")
        if not (0 <= result.target_delay_samples <= result.samples_per_pulse - result.pulse_samples):
            raise ValueError("target_delay_samples leaves the pulse outside the receive window")
        if abs(result.doppler_bin) >= result.num_pulses // 2:
            raise ValueError("doppler_bin is outside the unaliased FFT range")
        if min(result.sample_rate_hz, result.prf_hz, result.bandwidth_hz,
               result.carrier_frequency_hz, result.target_amplitude) <= 0 or result.noise_std < 0:
            raise ValueError("Task1 rates, bandwidth, carrier and amplitude must be positive; noise nonnegative")
        if not 0 < result.pfa < 1:
            raise ValueError("pfa must be in (0,1)")
        if min(result.cfar_guard_doppler, result.cfar_guard_range,
               result.cfar_reference_doppler, result.cfar_reference_range) < 0:
            raise ValueError("CFAR cells must be nonnegative")
        ambiguity_rows = 2 * result.pulse_samples - 1
        if result.ambiguity_nfreq != 1 << (ambiguity_rows - 1).bit_length():
            raise ValueError("ambiguity_nfreq must equal next_power_of_two(2*pulse_samples-1)")
        return result


@dataclass(frozen=True)
class Task2Config:
    path: Path
    sha256: str
    samples: int
    sample_rate_hz: float
    target_delay_samples: int
    noise_seed: int
    target_weight: float
    gaussian_weight: float
    sawtooth_weight: float
    square_weight: float
    noise_amplitude: float
    filter_taps: int
    filter_cutoff_hz: float
    step3_crop_each: int
    fusion_weight_fm: float
    fusion_weight_correlation: float
    fusion_weight_spectral: float
    fusion_weight_wavelet: float
    argrelextrema_order: int
    kalman_observation_count: int
    kalman_F: float
    kalman_Q: float
    kalman_H: float
    kalman_R: float
    kalman_measurement_noise_variant: float

    @classmethod
    def load(cls, path: str | Path) -> "Task2Config":
        resolved = Path(path).expanduser().resolve(strict=True)
        fields = set(cls.__dataclass_fields__) - {"path", "sha256"}
        raw, digest = _parse(resolved, fields)
        integer_fields = {"samples", "target_delay_samples", "noise_seed", "filter_taps",
                          "step3_crop_each", "argrelextrema_order", "kalman_observation_count"}
        parsed = {key: _integer(raw, key) if key in integer_fields else _number(raw, key) for key in fields}
        result = cls(path=resolved, sha256=digest, **parsed)
        if not _power_of_two(result.samples) or result.samples < 256:
            raise ValueError("samples must be a power of two >= 256")
        if not 0 <= result.target_delay_samples < result.samples:
            raise ValueError("target_delay_samples is outside the signal")
        if result.sample_rate_hz <= 0 or not 0 < result.filter_cutoff_hz < result.sample_rate_hz / 2:
            raise ValueError("sample rate/cutoff violate Nyquist")
        if result.filter_taps < 3 or result.filter_taps % 2 == 0:
            raise ValueError("filter_taps must be odd and >= 3")
        if result.step3_crop_each < 0 or 2 * result.step3_crop_each >= result.samples - 64:
            raise ValueError("step3_crop_each leaves insufficient output")
        nonnegative = (result.target_weight, result.gaussian_weight, result.sawtooth_weight,
                       result.square_weight, result.noise_amplitude, result.fusion_weight_fm,
                       result.fusion_weight_correlation, result.fusion_weight_spectral,
                       result.fusion_weight_wavelet, result.kalman_Q, result.kalman_R,
                       result.kalman_measurement_noise_variant)
        if any(value < 0 for value in nonnegative):
            raise ValueError("weights, noise and Kalman covariance values must be nonnegative")
        if not math.isclose(sum((result.fusion_weight_fm, result.fusion_weight_correlation,
                                result.fusion_weight_spectral, result.fusion_weight_wavelet)), 1.0,
                            rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("fusion weights must sum to 1")
        if result.argrelextrema_order < 1 or result.argrelextrema_order * 2 >= result.samples:
            raise ValueError("argrelextrema_order is invalid")
        if not 1 <= result.kalman_observation_count <= result.samples:
            raise ValueError("kalman_observation_count is invalid")
        if result.kalman_F == 0 or result.kalman_H == 0 or result.kalman_R == 0:
            raise ValueError("Kalman F, H and R must be nonzero")
        return result
