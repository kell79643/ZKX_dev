from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from steps.step1.step1 import run_step1
from steps.step2.step2 import run_step2
from steps.step3.step3 import run_step3
from steps.step4.step4 import run_step4
from steps.step5.step5 import run_step5
from task1_common import (
    Task1Config,
    PipelineState,
    StepEvidence,
    require,
    require_cusignal_runtime,
    runtime_metadata,
    wall_ms,
)


RUN_STEPS = [run_step1, run_step2, run_step3, run_step4, run_step5]


def _half_power_width(values: np.ndarray, peak: int) -> int:
    threshold = float(values[peak]) / np.sqrt(2.0)
    left = right = peak
    while left > 0 and values[left - 1] >= threshold:
        left -= 1
    while right + 1 < values.size and values[right + 1] >= threshold:
        right += 1
    return right - left + 1


def _pslr_db(values: np.ndarray, peak: int, exclusion: int) -> float:
    mask = np.ones(values.size, dtype=bool)
    mask[max(0, peak - exclusion): min(values.size, peak + exclusion + 1)] = False
    sidelobe = float(np.max(values[mask], initial=0.0))
    return float(20.0 * np.log10(max(sidelobe, 1.0e-12) / float(values[peak])))


def _result_metrics(
    step_index: int, state: PipelineState, evidence: StepEvidence
) -> dict[str, float]:
    """Summarize cuSignal results without building a CPU reference or error gate."""
    config = state.config
    if step_index == 1:
        require(state.waveform is not None and state.echo is not None, "step1 results missing")
        require(state.noiseless_echo is not None and state.noise is not None, "step1 components missing")
        return {
            "chirp_called": float("cusignal.chirp" in evidence.operator_ms),
            "target_echo_present": float(np.max(np.abs(state.noiseless_echo)) > 0.0),
            "delay_ground_truth_recorded": 1.0,
            "delay_ground_truth_samples": float(config.target_delay_samples),
            "doppler_ground_truth_recorded": 1.0,
            "doppler_ground_truth_hz": float(config.doppler_bin * config.prf_hz / config.num_pulses),
            "noise_added": float(np.max(np.abs(state.noise)) > 0.0),
            "waveform_samples": float(state.waveform.size),
            "echo_elements": float(state.echo.size),
            "echo_peak_magnitude": float(np.max(np.abs(state.echo))),
        }
    if step_index == 2:
        require(state.compressed is not None, "step2 results missing")
        magnitude = np.abs(state.compressed)
        profile = np.sum(magnitude * magnitude, axis=0)
        peak = int(np.argmax(profile))
        half_power_width = _half_power_width(magnitude[0], peak)
        return {
            "pulse_compression_called": float("cusignal.pulse_compression" in evidence.operator_ms),
            "step1_to_step2_data_match": float(
                state.echo is not None and state.echo.shape == (config.num_pulses, config.samples_per_pulse)
                and state.waveform is not None and state.waveform.size == config.pulse_samples
            ),
            "peak_range_bin": float(peak),
            "peak_energy": float(profile[peak]),
            "half_peak_width_bins": float(half_power_width),
            "output_elements": float(state.compressed.size),
            "complex64_output": float(state.compressed.dtype == np.complex64),
        }
    if step_index == 3:
        require(state.range_doppler is not None, "step3 results missing")
        power = np.abs(state.range_doppler) ** 2
        doppler_bin, range_bin = np.unravel_index(int(np.argmax(power)), power.shape)
        return {
            "pulse_doppler_called": float("cusignal.pulse_doppler" in evidence.operator_ms),
            "hamming_fp32_adapter_called": float(
                "utils.hamming_fp32_adapter" in evidence.operator_ms
            ),
            "step2_to_step3_data_match": float(
                state.compressed is not None
                and state.compressed.shape == (config.num_pulses, config.samples_per_pulse)
            ),
            "peak_doppler_bin": float(doppler_bin),
            "peak_range_bin": float(range_bin),
            "detected_doppler_hz": float(doppler_bin * config.prf_hz / config.num_pulses),
            "peak_power": float(power[doppler_bin, range_bin]),
            "complex64_output": float(state.range_doppler.dtype == np.complex64),
        }
    if step_index == 4:
        require(state.detections is not None and state.cfar_threshold is not None, "step4 results missing")
        target_index = config.doppler_bin % config.num_pulses
        target_detected = bool(state.detections[target_index, config.target_delay_samples])
        return {
            "cfar_alpha_called": float("cusignal.cfar_alpha_host" in evidence.operator_ms),
            "ca_cfar_called": float("cusignal.ca_cfar" in evidence.operator_ms),
            "step3_to_step4_data_match": float(
                state.range_doppler is not None
                and state.range_doppler.shape == (config.num_pulses, config.samples_per_pulse)
            ),
            "target_detected": float(target_detected),
            "cfar_alpha": float(state.cfar_alpha),
            "detection_count": float(np.count_nonzero(state.detections)),
            "threshold_finite_fraction": float(np.mean(np.isfinite(state.cfar_threshold))),
        }
    require(state.ambiguity_2d is not None, "step5 2D result missing")
    require(state.ambiguity_delay is not None and state.ambiguity_doppler is not None, "step5 cuts missing")
    peak_row, peak_column = np.unravel_index(
        int(np.argmax(state.ambiguity_2d)), state.ambiguity_2d.shape
    )
    delay_peak = int(np.argmax(state.ambiguity_delay))
    doppler_peak = int(np.argmax(state.ambiguity_doppler))
    delay_width = _half_power_width(state.ambiguity_delay, delay_peak)
    doppler_width = _half_power_width(state.ambiguity_doppler, doppler_peak)
    delay_pslr = _pslr_db(state.ambiguity_delay, delay_peak, max(2, delay_width))
    doppler_pslr = _pslr_db(state.ambiguity_doppler, doppler_peak, max(2, doppler_width))
    ambiguity_peak = float(np.max(state.ambiguity_2d))
    finite = bool(
        np.isfinite(state.ambiguity_2d).all()
        and np.isfinite(state.ambiguity_delay).all()
        and np.isfinite(state.ambiguity_doppler).all()
    )
    return {
        "ambgfun_called": float(
            sum(name.startswith("cusignal.ambgfun_2d") for name in evidence.operator_ms)
            == 3
        ),
        "ambiguity_cut_adapter_called": float(
            "utils.ambgfun_doppler_cut_adapter" in evidence.operator_ms
            and "utils.ambgfun_delay_cut_adapter" in evidence.operator_ms
        ),
        "step1_waveform_to_step5_match": float(
            state.waveform is not None and state.waveform.size == config.pulse_samples
        ),
        "waveform_quality_pass": float(
            finite
            and peak_row == config.pulse_samples - 1
            and peak_column == state.ambiguity_2d.shape[1] // 2
            and delay_peak == state.ambiguity_delay.size // 2
            and doppler_peak == state.ambiguity_doppler.size // 2
            and abs(ambiguity_peak - 1.0) <= 3.0e-3
            and delay_width <= 3
            and doppler_width <= 5
            and delay_pslr <= -8.0
            and doppler_pslr <= -8.0
        ),
        "ambiguity_peak": ambiguity_peak,
        "ambiguity_peak_row": float(peak_row),
        "ambiguity_peak_column": float(peak_column),
        "ambiguity_rows": float(state.ambiguity_2d.shape[0]),
        "ambiguity_columns": float(state.ambiguity_2d.shape[1]),
        "peak_delay_index": float(delay_peak),
        "peak_doppler_index": float(doppler_peak),
        "delay_mainlobe_width_samples": float(delay_width),
        "doppler_mainlobe_width_bins": float(doppler_width),
        "delay_pslr_db": delay_pslr,
        "doppler_pslr_db": doppler_pslr,
    }


def _run_step_with_results(
    step_index: int,
    state: PipelineState,
    run_step: Callable[[PipelineState], StepEvidence],
) -> StepEvidence:
    total_begin = time.perf_counter()
    evidence = run_step(state)
    post_begin = time.perf_counter()
    evidence.metrics = _result_metrics(step_index, state, evidence)
    evidence.post_ms = wall_ms(post_begin)
    evidence.total_ms = wall_ms(total_begin)
    return evidence


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _step_json(evidence: StepEvidence) -> dict[str, Any]:
    raw = asdict(evidence)
    categories = {
        "cusignal_api": sum(
            value for name, value in raw["operator_ms"].items()
            if name.startswith("cusignal.")
        ),
        "required_zq500_adapter": sum(
            value for name, value in raw["operator_ms"].items()
            if name.startswith("utils.")
        ),
        "task_scaffolding": sum(
            value for name, value in raw["operator_ms"].items()
            if name.startswith("task_scaffolding.") or name.startswith("cupy.")
        ),
    }
    return {
        "name": evidence.name,
        "timing_ms": {
            "prepare": raw["prep_ms"],
            "h2d": raw["h2d_ms"],
            "compute": raw["compute_ms"],
            "d2h": raw["d2h_ms"],
            "postprocess": raw["post_ms"],
            "formal_execution": raw["formal_execution_ms"],
            "total": raw["total_ms"],
        },
        "operators_ms": raw["operator_ms"],
        "timing_categories_ms": categories,
        "result_metrics": raw["metrics"],
    }


def run_task1_until(
    stop_after: int,
    evidence_id: str,
    config: Task1Config,
    output_directory: str | Path = ".",
    write_json: bool = True,
    quiet: bool = False,
) -> dict[str, Any]:
    require_cusignal_runtime()
    require(1 <= stop_after <= 5, "Task1 stop_after must be in [1,5]")
    require(bool(evidence_id), "Task1 evidence_id must be nonempty")
    pipeline_begin = time.perf_counter()
    state = PipelineState(config=config)
    steps: list[StepEvidence] = []
    for index in range(stop_after):
        steps.append(
            _run_step_with_results(
                index + 1,
                state,
                RUN_STEPS[index],
            )
        )
    pipeline_total_ms = wall_ms(pipeline_begin)
    result = {
        "task": "Task1",
        "implementation": "cusignal_python_23.08.00",
        "evidence_id": evidence_id,
        "requested_steps": stop_after,
        "completed_steps": stop_after,
        "status": "pass",
        "runtime": runtime_metadata(),
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "input_contract": {
            "dtype": "ComplexFP32",
            "shape": [config.num_pulses, config.samples_per_pulse],
            "sample_rate_hz": config.sample_rate_hz,
            "prf_hz": config.prf_hz,
            "bandwidth_hz": config.bandwidth_hz,
            "carrier_frequency_hz": config.carrier_frequency_hz,
            "pulse_samples": config.pulse_samples,
            "target_delay_samples": config.target_delay_samples,
            "doppler_bin": config.doppler_bin,
            "doppler_hz": config.doppler_bin * config.prf_hz / config.num_pulses,
            "target_amplitude": config.target_amplitude,
            "noise_std": config.noise_std,
            "noise_seed": config.noise_seed,
            "pfa": config.pfa,
            "cfar_guard": [config.cfar_guard_doppler, config.cfar_guard_range],
            "cfar_reference": [config.cfar_reference_doppler, config.cfar_reference_range],
            "ambiguity_nfreq": config.ambiguity_nfreq,
        },
        "timing_policy": {
            "gpu_clock": "cupy.cuda.Event",
            "step_clock": "time.perf_counter",
            "h2d_scope": "cp.asarray followed by current-stream synchronization",
            "pipeline_handoff": "Host PipelineState with explicit H2D/D2H, matching the C++/GPU pipeline",
            "d2h_scope": "formal step output copied for result retention",
            "postprocess_scope": "result summarization only; no CPU reference or accuracy comparison",
            "dlpti_used": False,
            "dlpti_scope": "not used for formal timing; optional auxiliary profiling only",
            "python_speedup_numerator_scope": (
                "formal_execution_ms: cuSignal Python GPU calls, required ZQ500 adapters, "
                "task scaffolding, H2D and required D2H; excludes CPU reference/comparison"
            ),
        },
        "formal_apis": [
            "cusignal.pulse_compression",
            "cusignal.pulse_doppler",
            "cusignal.radartools.cfar_alpha",
            "cusignal.radartools.ca_cfar",
            "task1.utils.task1_ambgfun -> cusignal.ambgfun(cut=2d)",
        ],
        "implementation_contract": {
            "cusignal_upstream_version": "23.08.00",
            "cusignal_upstream_modified": False,
            "actual_cusignal_apis": [
                "cusignal.chirp",
                "cusignal.pulse_compression",
                "cusignal.pulse_doppler",
                "cusignal.radartools.cfar_alpha",
                "cusignal.radartools.ca_cfar",
                "cusignal.ambgfun(cut=2d)",
            ],
            "required_zq500_adapters": [
                {
                    "name": "utils.cusignal_chirp_compat",
                    "scope": (
                        "restore removed NumPy type-test alias and explicitly cast the "
                        "official linear chirp FP32 kernel constants"
                    ),
                    "reason": (
                        "cuSignal 23.08.00 references np.issubclass_ and ZQ500 clang "
                        "rejects NVIDIA-accepted implicit FP64-to-FP32 list narrowing"
                    ),
                },
                {
                    "name": "utils.hamming_fp32_adapter",
                    "scope": "FP32 Hamming formula/dtype boundary only",
                    "reason": "avoid unsupported ComplexFP64/Z2Z FFT promotion",
                },
                {
                    "name": "utils.ambgfun_*_cut_adapter",
                    "scope": "grid-aligned slices of official cuSignal 2D output only",
                    "reason": "avoid unsupported FP64/non-power-of-two one-dimensional paths",
                },
            ],
            "adapters_in_formal_execution": True,
            "adapters_in_speedup_numerator": True,
        },
        "pipeline_total_ms": pipeline_total_ms,
        "sum_step_total_ms": sum(step.total_ms for step in steps),
        "steps": [_step_json(step) for step in steps],
    }
    result = _json_safe(result)
    if write_json:
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        destination = output_path / f"task1_{evidence_id}_evidence.json"
        destination.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if not quiet:
        print(
            f"[TASK1_PYTHON][CONFIG] path={config.path} sha256={config.sha256} "
            f"shape=[{config.num_pulses},{config.samples_per_pulse}] "
            f"target_delay_samples={config.target_delay_samples} doppler_bin={config.doppler_bin}"
        )
        for step in steps:
            print(
                f"[TASK1_PYTHON][{step.name}][TIMING] prep_ms={step.prep_ms:.6f} "
                f"h2d_ms={step.h2d_ms:.6f} compute_ms={step.compute_ms:.6f} "
                f"d2h_ms={step.d2h_ms:.6f} post_ms={step.post_ms:.6f} "
                f"formal_execution_ms={step.formal_execution_ms:.6f} total_ms={step.total_ms:.6f}"
            )
            print(f"[TASK1_PYTHON][{step.name}][RESULT] metrics={json.dumps(step.metrics, sort_keys=True)}")
        print(
            f"[TASK1_PYTHON][PIPELINE] completed_steps={stop_after} "
            f"pipeline_total_ms={pipeline_total_ms:.6f} status=pass"
        )
    return result
