"""Runtime-only ZQ500 compatibility helpers for the immutable cuSignal source."""

from __future__ import annotations

import copy
import importlib
from typing import Any

import numpy as np


_INSTALLED = False
_DETAILS: dict[str, Any] = {}


ADAPTER_CONTRACTS: dict[str, dict[str, Any]] = {
    "waveform_source_kernels": {
        "public_apis": ["cusignal.chirp", "cusignal.sawtooth", "cusignal.square"],
        "upstream_backend": "cusignal.waveforms.waveforms ElementwiseKernel symbols",
        "zq500_reason": "upstream wheel fatbin/NVRTC kernel-loading path is unavailable on the private ZQ500 CuPy runtime",
        "adapter_action": "install source-JIT kernels with the same public parameters, equations and upstream output dtypes",
        "semantic_change": False,
        "cpu_fallback": False,
        "device_execution": True,
        "included_in_python_timing_numerator": True,
    },
    "cupy_direct_convolution_backend": {
        "public_apis": ["cusignal.firfilter", "cusignal.correlate", "cusignal.cwt"],
        "upstream_backend": "CuPy/cupyx one-dimensional direct convolve used by cuSignal",
        "zq500_reason": "the private ZQ500 CuPy convolution backend lacks the required working one-dimensional source path",
        "adapter_action": "provide the same full/same/valid direct-convolution primitive on ZQ500 device arrays",
        "semantic_change": False,
        "cpu_fallback": False,
        "device_execution": True,
        "included_in_python_timing_numerator": True,
    },
    "argrelextrema_backend": {
        "public_apis": ["cusignal.argrelextrema"],
        "upstream_backend": "cusignal.peak_finding._boolrelextrema compiled-kernel path",
        "zq500_reason": "the upstream compiled helper depends on NVIDIA/CuPy kernel loading unavailable on ZQ500",
        "adapter_action": "use cuSignal's own generic take/compare algorithm for the Task2 one-dimensional input",
        "semantic_change": False,
        "cpu_fallback": False,
        "device_execution": True,
        "included_in_python_timing_numerator": True,
    },
    "scalar_kalman_backend": {
        "public_apis": ["cusignal.KalmanFilter", "KalmanFilter.predict", "KalmanFilter.update", "KalmanFilter.predict_update_sequence"],
        "upstream_backend": "cusignal.estimation._filters_cuda RawModule specialization",
        "zq500_reason": "ZQ500 CuPy does not support the upstream NVRTC lowered-name/name_expressions path",
        "adapter_action": "evaluate the same scalar predict/update equations with CuPy device operations for dim_x=dim_z=1",
        "semantic_change": False,
        "cpu_fallback": False,
        "device_execution": True,
        "included_in_python_timing_numerator": True,
        "approved_task_shape": {"dim_x": 1, "dim_z": 1, "dim_u": 0},
    },
}

INTERMEDIATE_DOUBLE_AUDIT: dict[str, dict[str, Any]] = {
    "waveform.sawtooth_square_outputs": {
        "business_input_dtype": "FP32",
        "intermediate_output_dtype": "FP64",
        "source": "cuSignal 23.08 waveform public output contract",
        "reason_retained": "changing the public cuSignal output dtype would make the comparison semantically unequal",
        "device_only": True,
        "runtime_evidence": "stock/adapted probe plus Task2 Step1 and pipeline acceptance on ZQ500",
    },
    "step4.cwt_output": {
        "business_input_dtype": "FP32",
        "intermediate_output_dtype": "FP64",
        "source": "cuSignal 23.08 cwt/ricker implementation-selected output",
        "reason_retained": "the Task2 adapter preserves the upstream public API instead of silently narrowing it",
        "device_only": True,
        "runtime_evidence": "Task2 Step4 and pipeline acceptance on ZQ500",
    },
}


class _CupyProxy:
    def __init__(self, cupy_module: Any, convolve_function: Any) -> None:
        self._cupy = cupy_module
        self.convolve = convolve_function

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cupy, name)


def _install_source_convolution(cp: Any) -> Any:
    kernel = cp.ElementwiseKernel(
        "raw T first, raw T second, int32 first_size, int32 second_size, int32 offset",
        "T output",
        """
        const int full_index = i + offset;
        T accumulator {};
        for (int second_index = 0; second_index < second_size; ++second_index) {
            const int first_index = full_index - second_index;
            if (first_index >= 0 && first_index < first_size) {
                accumulator += first[first_index] * second[second_index];
            }
        }
        output = accumulator;
        """,
        "_task2_zq500_convolve_1d",
        options=("-std=c++11",),
    )

    def direct_convolve(first: Any, second: Any, mode: str = "full", method: str = "auto") -> Any:
        del method
        first = cp.asarray(first)
        second = cp.asarray(second)
        if first.ndim == second.ndim == 0:
            return first * second
        if first.ndim != 1 or second.ndim != 1:
            raise ValueError("ZQ500 source convolution adapter supports one-dimensional inputs")
        if mode not in ("full", "same", "valid"):
            raise ValueError("mode must be 'full', 'same', or 'valid'")
        dtype = cp.result_type(first, second)
        first = first.astype(dtype, copy=False)
        second = second.astype(dtype, copy=False)
        full_size = int(first.size + second.size - 1)
        if mode == "full":
            output_size = full_size
            offset = 0
        elif mode == "same":
            output_size = int(first.size)
            offset = (full_size - output_size) // 2
        else:
            output_size = abs(int(first.size) - int(second.size)) + 1
            offset = min(int(first.size), int(second.size)) - 1
        return kernel(
            first,
            second,
            np.int32(first.size),
            np.int32(second.size),
            np.int32(offset),
            size=output_size,
        )

    return direct_convolve


def _install_waveform_kernels(cp: Any) -> None:
    module = importlib.import_module("cusignal.waveforms.waveforms")
    module._sawtooth_kernel = cp.ElementwiseKernel(
        "T t, T w",
        "float64 y",
        """
        double out {};
        const bool invalid { ((w > 1) || (w < 0)) };
        if (invalid) out = nan("0xfff8000000000000ULL");
        const T tmod = fmod(t, 2.0 * M_PI);
        const bool rising { ((1 - invalid) && (tmod < (w * 2.0 * M_PI))) };
        if (rising) out = tmod / (M_PI * w) - 1;
        if ((1 - invalid) && (1 - rising))
            out = (M_PI * (w + 1) - tmod) / (M_PI * (1 - w));
        y = out;
        """,
        "_task2_zq500_sawtooth",
        options=("-std=c++11",),
    )
    module._square_kernel = cp.ElementwiseKernel(
        "T t, T w",
        "float64 y",
        """
        const bool invalid { ((w > 1) || (w < 0)) };
        if (invalid) y = nan("0xfff8000000000000ULL");
        const T tmod = fmod(t, 2.0 * M_PI);
        const bool high { ((1 - invalid) && (tmod < (w * 2.0 * M_PI))) };
        if (high) y = 1;
        if ((1 - invalid) && (1 - high)) y = -1;
        """,
        "_task2_zq500_square",
        options=("-std=c++11",),
    )
    chirp_source = """
        const T beta { (f1 - f0) / t1 };
        const T temp = 2 * M_PI * (f0 * t + 0.5 * beta * t * t);
        phase = cos(temp + phi);
    """
    module._chirp_phase_lin_kernel_real = cp.ElementwiseKernel(
        "T t, T f0, T t1, T f1, T phi",
        "T phase",
        chirp_source,
        "_task2_zq500_chirp_linear_real",
        options=("-std=c++11",),
    )
    module._chirp_phase_lin_kernel_cplx = cp.ElementwiseKernel(
        "T t, T f0, T t1, T f1, T phi",
        "Y phase",
        """
        const T beta { (f1 - f0) / t1 };
        const T temp = 2 * M_PI * (f0 * t + 0.5 * beta * t * t);
        phase = Y(cos(temp + phi), cos(temp + phi + M_PI / 2) * -1);
        """,
        "_task2_zq500_chirp_linear_complex",
        options=("-std=c++11",),
    )


def _install_peak_helper(cp: Any) -> None:
    module = importlib.import_module("cusignal.peak_finding.peak_finding")

    def boolrelextrema(data: Any, comparator: Any, axis: int = 0, order: int = 1, mode: str = "clip") -> Any:
        if int(order) != order or order < 1:
            raise ValueError("Order must be an int >= 1")
        length = data.shape[axis]
        locations = cp.arange(length)
        result = cp.ones(data.shape, dtype=bool)
        main = cp.take(data, locations, axis=axis)
        for shift in range(1, order + 1):
            if mode == "clip":
                plus_locations = cp.clip(locations + shift, None, length - 1)
                minus_locations = cp.clip(locations - shift, 0, None)
            elif mode == "wrap":
                plus_locations = (locations + shift) % length
                minus_locations = (locations - shift) % length
            else:
                raise NotImplementedError("CuPy `take` doesn't support mode='raise'.")
            result &= comparator(main, cp.take(data, plus_locations, axis=axis))
            result &= comparator(main, cp.take(data, minus_locations, axis=axis))
        return result

    module._boolrelextrema = boolrelextrema


def _install_scalar_kalman(cp: Any, cusignal: Any) -> None:
    klass = cusignal.KalmanFilter
    original_init = klass.__init__
    original_predict = klass.predict
    original_update = klass.update
    sequence_kernel = cp.ElementwiseKernel(
        "raw float32 observations, int32 observation_count, raw float32 initial_x, "
        "raw float32 initial_p, raw float32 f, raw float32 q, raw float32 alpha_sq, "
        "raw float32 h, raw float32 r",
        "float32 next_x, float32 next_p",
        """
        float state = initial_x[0];
        float covariance = initial_p[0];
        const float transition = f[0];
        const float process_noise = q[0];
        const float alpha = alpha_sq[0];
        const float measurement = h[0];
        const float measurement_noise = r[0];
        for (int index = 0; index < observation_count; ++index) {
            state = transition * state;
            covariance = alpha * transition * covariance * transition + process_noise;
            const float innovation = observations[index] - measurement * state;
            const float pht = covariance * measurement;
            const float gain = pht / (measurement * pht + measurement_noise);
            state += gain * innovation;
            const float i_kh = 1.0F - gain * measurement;
            covariance = i_kh * covariance * i_kh + gain * measurement_noise * gain;
        }
        next_x = state;
        next_p = covariance;
        """,
        "_task2_zq500_kalman_scalar_sequence",
        options=("-std=c++11",),
    )

    def init(self: Any, dim_x: int, dim_z: int, dim_u: int = 0, points: int = 1, dtype: Any = None) -> None:
        dtype = cp.float32 if dtype is None else dtype
        if dim_x != 1 or dim_z != 1 or dim_u != 0:
            original_init(self, dim_x, dim_z, dim_u, points, dtype)
            return
        self.points = points
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u
        shape = (points, 1, 1)
        self.x = cp.zeros(shape, dtype=dtype)
        self.P = cp.ones(shape, dtype=dtype)
        self.Q = cp.ones(shape, dtype=dtype)
        self.B = None
        self.F = cp.ones(shape, dtype=dtype)
        self.H = cp.zeros(shape, dtype=dtype)
        self.R = cp.ones(shape, dtype=dtype)
        self._alpha_sq = cp.ones(shape, dtype=dtype)
        self.z = cp.empty(shape, dtype=dtype)
        self._task2_zq500_scalar_backend = True

    def predict(self: Any, u: Any = None, B: Any = None, F: Any = None, Q: Any = None) -> None:
        if not getattr(self, "_task2_zq500_scalar_backend", False):
            original_predict(self, u, B, F, Q)
            return
        if u is not None:
            raise NotImplementedError("Control Matrix implementation in process")
        del B
        F = self.F if F is None else cp.asarray(F)
        if Q is None:
            Q = self.Q
        elif cp.isscalar(Q):
            Q = cp.ones_like(self.Q) * Q
        else:
            Q = cp.asarray(Q)
        self.x[...] = F * self.x
        self.P[...] = self._alpha_sq * F * self.P * F + Q

    def update(self: Any, z: Any, R: Any = None, H: Any = None) -> None:
        if not getattr(self, "_task2_zq500_scalar_backend", False):
            original_update(self, z, R, H)
            return
        if z is None:
            return
        H = self.H if H is None else cp.asarray(H)
        if R is None:
            R = self.R
        elif cp.isscalar(R):
            R = cp.ones_like(self.R) * R
        else:
            R = cp.asarray(R)
        self.z[...] = cp.asarray(z)
        residual = self.z - H * self.x
        innovation = H * self.P * H + R
        gain = self.P * H / innovation
        identity_minus_gain_h = cp.ones_like(self.P) - gain * H
        self.x[...] = self.x + gain * residual
        self.P[...] = (
            identity_minus_gain_h * self.P * identity_minus_gain_h
            + gain * R * gain
        )

    def predict_update_sequence(self: Any, observations: Any) -> None:
        if not getattr(self, "_task2_zq500_scalar_backend", False):
            raise NotImplementedError(
                "predict_update_sequence is only defined for the approved scalar Kalman layout")
        values = cp.asarray(observations, dtype=cp.float32).reshape(-1)
        if values.size == 0:
            return
        next_x, next_p = sequence_kernel(
            values,
            np.int32(values.size),
            self.x,
            self.P,
            self.F,
            self.Q,
            self._alpha_sq,
            self.H,
            self.R,
            size=1,
        )
        self.x = next_x.reshape((1, 1, 1))
        self.P = next_p.reshape((1, 1, 1))
        self.z = values[-1:].reshape((1, 1, 1))

    klass.__init__ = init
    klass.predict = predict
    klass.update = update
    klass.predict_update_sequence = predict_update_sequence


def install(cusignal: Any, cp: Any) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    direct_convolve = _install_source_convolution(cp)
    _install_waveform_kernels(cp)
    filtering = importlib.import_module("cusignal.filtering.filtering")
    filtering.cp = _CupyProxy(cp, direct_convolve)
    wavelets = importlib.import_module("cusignal.wavelets.wavelets")
    wavelets.convolve = direct_convolve
    correlate = importlib.import_module("cusignal.convolution.correlate")
    correlate.convolve = direct_convolve
    _install_peak_helper(cp)
    _install_scalar_kalman(cp, cusignal)
    _DETAILS.update(
        implementation="cusignal_python_gpu_plus_required_zq500_adapter",
        public_cusignal_api_preserved=True,
        adapter_overhead_included_in_timing_numerator=True,
        cpu_fallback=False,
        waveform_kernels=True,
        direct_convolution=True,
        peak_helper=True,
        scalar_kalman=True,
        active_adapters=copy.deepcopy(ADAPTER_CONTRACTS),
        intermediate_double_audit=copy.deepcopy(INTERMEDIATE_DOUBLE_AUDIT),
        vendored_source_modified=False,
    )
    _INSTALLED = True


def adapter_contract() -> dict[str, dict[str, Any]]:
    return copy.deepcopy(ADAPTER_CONTRACTS)


def adapter_status() -> dict[str, Any]:
    return {"installed": _INSTALLED, **_DETAILS}
