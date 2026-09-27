# Copyright (c) 2020-2021, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cupy as cp
import numpy as np
import pytest

from cusignal.radartools import ambgfun, cfar_alpha, pulse_compression, pulse_doppler
from cusignal.testing.utils import array_equal


def cpu_pulse_compression(x, template, normalize=False, window=None, nfft=None):
    num_pulses, samples_per_pulse = x.shape

    if nfft is None:
        nfft = samples_per_pulse

    if window is not None:
        Nx = len(template)
        if callable(window):
            W = window(np.fft.fftfreq(Nx))
        elif isinstance(window, np.ndarray):
            if window.shape != (Nx,):
                raise ValueError("window must have the same length as data")
            W = window
        else:
            from scipy.signal import get_window

            W = get_window(window, Nx, False)

        template = np.multiply(template, W)

    if normalize is True:
        template = np.divide(template, np.linalg.norm(template))

    fft_x = np.fft.fft(x, nfft)
    fft_template = np.conj(np.tile(np.fft.fft(template, nfft), (num_pulses, 1)))
    compressedIQ = np.fft.ifft(np.multiply(fft_x, fft_template), nfft)

    return compressedIQ


def cpu_pulse_doppler(x, window=None, nfft=None):
    num_pulses, samples_per_pulse = x.shape

    if nfft is None:
        nfft = num_pulses

    if window is not None:
        Nx = num_pulses
        if callable(window):
            W = window(np.fft.fftfreq(Nx))
        elif isinstance(window, np.ndarray):
            if window.shape != (Nx,):
                raise ValueError("window must have the same length as data")
            W = window
        else:
            from scipy.signal import get_window

            W = get_window(window, Nx, False)[np.newaxis]

        pd_dataMatrix = np.fft.fft(
            np.multiply(x, np.tile(W.T, (1, samples_per_pulse)), nfft, axis=0)
        )
    else:
        pd_dataMatrix = np.fft.fft(x, nfft, axis=0)

    return pd_dataMatrix


class TestPulseCompression:
    @pytest.mark.parametrize(
        "num_pulses, samples_per_pulse, template_len, normalize, nfft",
        [
            (10, 100, 50, False, None),
            (20, 200, 100, True, None),
            (15, 150, 75, False, 256),
            (5, 50, 25, True, 128),
        ],
    )
    def test_pulse_compression_basic(
        self, num_pulses, samples_per_pulse, template_len, normalize, nfft
    ):
        np.random.seed(42)
        cpu_x = np.random.randn(num_pulses, samples_per_pulse) + 1j * np.random.randn(
            num_pulses, samples_per_pulse
        )
        gpu_x = cp.asarray(cpu_x)

        cpu_template = np.random.randn(template_len) + 1j * np.random.randn(template_len)
        gpu_template = cp.asarray(cpu_template)

        cpu_result = cpu_pulse_compression(cpu_x, cpu_template, normalize, nfft=nfft)
        gpu_result = pulse_compression(gpu_x, gpu_template, normalize, nfft=nfft)

        array_equal(gpu_result, cpu_result)

    @pytest.mark.parametrize("window", ["hamming", "hann", "blackman"])
    def test_pulse_compression_with_window(self, window):
        np.random.seed(42)
        num_pulses = 10
        samples_per_pulse = 100
        template_len = 50

        cpu_x = np.random.randn(num_pulses, samples_per_pulse) + 1j * np.random.randn(
            num_pulses, samples_per_pulse
        )
        gpu_x = cp.asarray(cpu_x)

        cpu_template = np.random.randn(template_len) + 1j * np.random.randn(template_len)
        gpu_template = cp.asarray(cpu_template)

        cpu_result = cpu_pulse_compression(cpu_x, cpu_template, window=window)
        gpu_result = pulse_compression(gpu_x, gpu_template, window=window)

        array_equal(gpu_result, cpu_result)

    @pytest.mark.cpu
    def test_pulse_compression_cpu_version(self):
        np.random.seed(42)
        num_pulses = 10
        samples_per_pulse = 100
        template_len = 50

        cpu_x = np.random.randn(num_pulses, samples_per_pulse) + 1j * np.random.randn(
            num_pulses, samples_per_pulse
        )
        cpu_template = np.random.randn(template_len) + 1j * np.random.randn(template_len)

        result = cpu_pulse_compression(cpu_x, cpu_template, normalize=True)

        assert result.shape == (num_pulses, samples_per_pulse)
        assert result.dtype == np.complex128


class TestPulseDoppler:
    @pytest.mark.parametrize(
        "num_pulses, samples_per_pulse, nfft",
        [(10, 100, None), (20, 200, 256), (15, 150, 128)],
    )
    def test_pulse_doppler_basic(self, num_pulses, samples_per_pulse, nfft):
        np.random.seed(42)
        cpu_x = np.random.randn(num_pulses, samples_per_pulse) + 1j * np.random.randn(
            num_pulses, samples_per_pulse
        )
        gpu_x = cp.asarray(cpu_x)

        cpu_result = cpu_pulse_doppler(cpu_x, nfft=nfft)
        gpu_result = pulse_doppler(gpu_x, nfft=nfft)

        array_equal(gpu_result, cpu_result)

    @pytest.mark.parametrize("window", ["hamming", "hann", "blackman"])
    def test_pulse_doppler_with_window(self, window):
        np.random.seed(42)
        num_pulses = 10
        samples_per_pulse = 100

        cpu_x = np.random.randn(num_pulses, samples_per_pulse) + 1j * np.random.randn(
            num_pulses, samples_per_pulse
        )
        gpu_x = cp.asarray(cpu_x)

        cpu_result = cpu_pulse_doppler(cpu_x, window=window)
        gpu_result = pulse_doppler(gpu_x, window=window)

        array_equal(gpu_result, cpu_result)

    @pytest.mark.cpu
    def test_pulse_doppler_cpu_version(self):
        np.random.seed(42)
        num_pulses = 10
        samples_per_pulse = 100

        cpu_x = np.random.randn(num_pulses, samples_per_pulse) + 1j * np.random.randn(
            num_pulses, samples_per_pulse
        )

        result = cpu_pulse_doppler(cpu_x)

        assert result.shape == (num_pulses, samples_per_pulse)
        assert result.dtype == np.complex128


class TestAmbgfun:
    @pytest.mark.parametrize("x_len, fs, prf", [(50, 1000, 100), (100, 2000, 200)])
    def test_ambgfun_2d(self, x_len, fs, prf):
        np.random.seed(42)
        cpu_x = np.random.randn(x_len) + 1j * np.random.randn(x_len)
        gpu_x = cp.asarray(cpu_x)

        cpu_result = ambgfun(cpu_x, fs, prf, cut="2d")
        gpu_result = ambgfun(gpu_x, fs, prf, cut="2d")

        array_equal(gpu_result, cpu_result)

    @pytest.mark.parametrize("x_len, fs, prf, cutValue", [(50, 1000, 100, 0), (100, 2000, 200, 5)])
    def test_ambgfun_delay(self, x_len, fs, prf, cutValue):
        np.random.seed(42)
        cpu_x = np.random.randn(x_len) + 1j * np.random.randn(x_len)
        gpu_x = cp.asarray(cpu_x)

        cpu_result = ambgfun(cpu_x, fs, prf, cut="delay", cutValue=cutValue)
        gpu_result = ambgfun(gpu_x, fs, prf, cut="delay", cutValue=cutValue)

        array_equal(gpu_result, cpu_result)

    @pytest.mark.parametrize("x_len, fs, prf, cutValue", [(50, 1000, 100, 0), (100, 2000, 200, 10)])
    def test_ambgfun_doppler(self, x_len, fs, prf, cutValue):
        np.random.seed(42)
        cpu_x = np.random.randn(x_len) + 1j * np.random.randn(x_len)
        gpu_x = cp.asarray(cpu_x)

        cpu_result = ambgfun(cpu_x, fs, prf, cut="doppler", cutValue=cutValue)
        gpu_result = ambgfun(gpu_x, fs, prf, cut="doppler", cutValue=cutValue)

        array_equal(gpu_result, cpu_result)

    def test_ambgfun_with_y(self):
        np.random.seed(42)
        x_len = 50
        y_len = 50
        fs = 1000
        prf = 100

        cpu_x = np.random.randn(x_len) + 1j * np.random.randn(x_len)
        cpu_y = np.random.randn(y_len) + 1j * np.random.randn(y_len)
        gpu_x = cp.asarray(cpu_x)
        gpu_y = cp.asarray(cpu_y)

        cpu_result = ambgfun(cpu_x, fs, prf, y=cpu_y, cut="2d")
        gpu_result = ambgfun(gpu_x, fs, prf, y=gpu_y, cut="2d")

        array_equal(gpu_result, cpu_result)

    def test_ambgfun_invalid_cut(self):
        np.random.seed(42)
        x_len = 50
        fs = 1000
        prf = 100

        cpu_x = np.random.randn(x_len) + 1j * np.random.randn(x_len)
        gpu_x = cp.asarray(cpu_x)

        with pytest.raises(ValueError):
            ambgfun(gpu_x, fs, prf, cut="invalid")

    @pytest.mark.cpu
    def test_ambgfun_cpu_version(self):
        np.random.seed(42)
        x_len = 50
        fs = 1000
        prf = 100

        cpu_x = np.random.randn(x_len) + 1j * np.random.randn(x_len)

        result_2d = ambgfun(cpu_x, fs, prf, cut="2d")
        result_delay = ambgfun(cpu_x, fs, prf, cut="delay", cutValue=0)
        result_doppler = ambgfun(cpu_x, fs, prf, cut="doppler", cutValue=0)

        assert result_2d.shape[0] == 2 * x_len - 1
        assert result_delay.shape[0] == 2 ** np.ceil(np.log2(2 * x_len - 1))
        assert result_doppler.shape[0] == 2 * x_len - 1


class TestCfarAlpha:
    @pytest.mark.parametrize("pfa, N", [(1e-3, 10), (1e-6, 20), (1e-9, 50)])
    def test_cfar_alpha(self, pfa, N):
        cpu_result = cfar_alpha(pfa, N)
        gpu_result = cfar_alpha(pfa, N)

        np.testing.assert_allclose(cpu_result, gpu_result)

    @pytest.mark.cpu
    def test_cfar_alpha_cpu_version(self):
        pfa = 1e-3
        N = 10

        result = cfar_alpha(pfa, N)

        assert result > 0
        assert isinstance(result, float)