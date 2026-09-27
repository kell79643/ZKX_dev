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


def _analytic_signal(signal):
    previous = cp.concatenate((signal[:1], signal[:-1]))
    following = cp.concatenate((signal[1:], signal[-1:]))
    return (signal + cp.complex64(0.5j) * (following - previous)).astype(cp.complex64)


def _resample_linear(values, count: int):
    if values.size == count:
        return values.astype(cp.float32)
    old = cp.linspace(0.0, 1.0, values.size, dtype=cp.float32)
    new = cp.linspace(0.0, 1.0, count, dtype=cp.float32)
    return cp.interp(new, old, values.astype(cp.float32))


def _normalize_abs(values):
    magnitudes = cp.abs(values).astype(cp.float32)
    maximum = cp.max(magnitudes)
    return cp.where(maximum > 0.0, magnitudes / maximum, cp.zeros_like(magnitudes))


def _make_bundle(fm, correlation, spectral, wavelet, count: int, config):
    fm_n = _normalize_abs(_resample_linear(fm, count))
    corr_n = _normalize_abs(_resample_linear(correlation, count))
    spec_n = _normalize_abs(_resample_linear(spectral, count))
    wave_n = _normalize_abs(_resample_linear(wavelet, count))
    return (
        cp.float32(config.fusion_weight_fm) * fm_n
        + cp.float32(config.fusion_weight_correlation) * corr_n
        + cp.float32(config.fusion_weight_spectral) * spec_n
        + cp.float32(config.fusion_weight_wavelet) * wave_n
    ).astype(cp.float32)


def run_step4(state: PipelineState) -> StepEvidence:
    config = state.config
    evidence = StepEvidence("step4")
    total_begin = time.perf_counter()
    formal_begin = time.perf_counter()
    prep_begin = time.perf_counter()
    require(state.smoothed is not None and state.smoothed.size > 64, "step4 did not receive step3 output")
    require(
        state.reference_for_correlation is not None
        and state.reference_for_correlation.size == state.smoothed.size,
        "step4 correlation reference mismatch",
    )
    count = int(state.smoothed.size)
    widths = [2, 4, 8, 12]
    evidence.prep_ms = wall_ms(prep_begin)
    h2d_begin = time.perf_counter()
    d_signal = cp.asarray(state.smoothed, dtype=cp.float32)
    d_reference = cp.asarray(state.reference_for_correlation, dtype=cp.float32)
    synchronize()
    evidence.h2d_ms = wall_ms(h2d_begin)
    d_analytic, analytic_ms = time_gpu(lambda: _analytic_signal(d_signal))
    d_fm, fm_ms = time_gpu(lambda: cusignal.fm_demod(d_analytic, axis=-1))
    d_correlation, correlation_ms = time_gpu(
        lambda: cusignal.correlate(d_signal, d_reference, mode="same", method="auto")
    )
    def run_spectrogram():
        return cusignal.spectrogram(
            d_signal,
            fs=config.sample_rate_hz,
            window="boxcar",
            nperseg=64,
            noverlap=32,
            nfft=64,
            detrend=False,
            return_onesided=False,
            scaling="spectrum",
            mode="magnitude",
        )
    spectrogram_outputs, spectral_ms = time_gpu(run_spectrogram)
    _, _, d_spectrogram = spectrogram_outputs
    d_spectral, spectral_envelope_ms = time_gpu(
        lambda: cp.max(cp.abs(d_spectrogram), axis=0).astype(cp.float32)
    )
    d_cwt, cwt_ms = time_gpu(lambda: cusignal.cwt(d_signal, cusignal.ricker, widths))
    d_wavelet, cwt_envelope_ms = time_gpu(
        lambda: cp.max(cp.abs(d_cwt), axis=0).astype(cp.float32)
    )
    d_bundle, bundle_ms = time_gpu(
        lambda: _make_bundle(d_fm, d_correlation, d_spectral, d_wavelet, count, config)
    )
    evidence.operator_ms = {
        "cupy.analytic_glue": analytic_ms,
        "cusignal.fm_demod": fm_ms,
        "cusignal.correlate": correlation_ms,
        "cusignal.spectrogram": spectral_ms,
        "cupy.spectral_envelope": spectral_envelope_ms,
        "cusignal.cwt_ricker": cwt_ms,
        "cupy.cwt_envelope": cwt_envelope_ms,
        "cupy.feature_fusion": bundle_ms,
    }
    evidence.compute_ms = sum(evidence.operator_ms.values())
    d2h_begin = time.perf_counter()
    state.fm_feature = as_host(d_fm).astype(np.float32)
    state.correlation_feature = as_host(d_correlation).astype(np.float32)
    state.spectral_feature = as_host(d_spectral).astype(np.float32)
    state.wavelet_feature = as_host(d_wavelet).astype(np.float32)
    state.feature_bundle = as_host(d_bundle).astype(np.float32)
    evidence.d2h_ms = wall_ms(d2h_begin)
    evidence.formal_execution_ms = wall_ms(formal_begin)
    post_begin = time.perf_counter()
    features = {
        "fm": state.fm_feature,
        "correlation": state.correlation_feature,
        "spectral": state.spectral_feature,
        "wavelet": state.wavelet_feature,
    }
    for name, item in features.items():
        require(np.all(np.isfinite(item)), f"step4 {name} feature is non-finite")
        require(np.max(np.abs(item)) > 0.0, f"step4 {name} feature is degenerate")
    require(np.max(state.feature_bundle) > 0.1, "step4 fused bundle degenerate")
    evidence.metrics = {
        "feature_families": 4.0,
        "demod_functions": 1.0,
        "correlate_functions": 1.0,
        "spectral_functions": 1.0,
        "wavelets_functions": 2.0,
        "step3_to_step4_data_match": 1.0,
        "feature_bundle_complete": 1.0,
        "fusion_weight_fm": config.fusion_weight_fm,
        "fusion_weight_correlation": config.fusion_weight_correlation,
        "fusion_weight_spectral": config.fusion_weight_spectral,
        "fusion_weight_wavelet": config.fusion_weight_wavelet,
        "bundle_samples": float(count),
        "bundle_peak": float(np.max(state.feature_bundle)),
    }
    evidence.output_shape = list(state.feature_bundle.shape)
    evidence.output_dtype = str(state.feature_bundle.dtype)
    evidence.post_ms = wall_ms(post_begin)
    evidence.total_ms = wall_ms(total_begin)
    return evidence
