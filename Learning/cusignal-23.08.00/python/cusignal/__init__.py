# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

from cusignal.acoustics.cepstrum import (
    complex_cepstrum,
    inverse_complex_cepstrum,
    minimum_phase,
    real_cepstrum,
)
# <学习注释：把 quadratic 提升到 cusignal 顶层命名空间，使用户可以直接调用 cusignal.quadratic；原始语句还导出另外两个符号。>
# <学习注释：从 bsplines.bsplines 模块导入 cubic、gauss_spline 和 quadratic，使 cubic 可通过顶层 cusignal.cubic 访问。>
from cusignal.bsplines.bsplines import cubic, gauss_spline, quadratic
from cusignal.convolution.convolve import (
    choose_conv_method,
    convolve,
    convolve1d2o,
    convolve1d3o,
    convolve2d,
    fftconvolve,
)
# <学习注释：把 correlate 和 correlate2d 从实现模块提升到 cusignal 顶层命名空间，因此用户可以直接调用 cusignal.correlate2d。>
from cusignal.convolution.correlate import correlate, correlate2d
# <学习注释：把 fm_demod 提升到 cusignal 顶层命名空间，使用户可直接调用 cusignal.fm_demod。>
from cusignal.demod.demod import fm_demod
from cusignal.estimation.filters import KalmanFilter
# <学习注释：从滤波器设计实现模块导入公开符号，使 firwin2 可以通过 cusignal.firwin2 直接调用。>
from cusignal.filter_design.fir_filter_design import (
    cmplx_sort,
    # <学习注释：把 firwin 提升到 cusignal 顶层命名空间，因此公开调用路径是 cusignal.firwin。>
    firwin,
    # <学习注释：这一项把 firwin2 绑定到 cusignal 顶层命名空间。>
    firwin2,
    kaiser_atten,
    kaiser_beta,
)
from cusignal.filtering.filtering import (
    channelize_poly,
    detrend,
    filtfilt,
    # <学习注释：把 filtering.py 中的 firfilter 提升到 cusignal 顶层命名空间，用户可直接调用 cusignal.firfilter。>
    firfilter,
    # <学习注释：把 firfilter2 继续提升到 cusignal 顶层，因此用户可以直接调用 cusignal.firfilter2。>
    firfilter2,
    firfilter_zi,
    freq_shift,
    # <学习注释：把 filtering.py 中的 hilbert 导出到 cusignal 顶层命名空间，使用户可直接使用 cusignal.hilbert。>
    hilbert,
    # <学习注释：把 filtering 子包导出的 hilbert2 再暴露为顶层 cusignal.hilbert2。>
    hilbert2,
    lfilter,
    lfilter_zi,
    # <学习注释：顶层包再次导出该符号，因此用户可直接调用 cusignal.sosfilt。>
    sosfilt,
    wiener,
)
from cusignal.filtering.resample import decimate, resample, resample_poly, upfirdn
from cusignal.io.reader import read_bin, read_sigmf, unpack_bin
from cusignal.io.writer import pack_bin, write_bin, write_sigmf
# <学习注释：把 peak_finding.py 中的 argrelextrema 提升到 cusignal 顶层命名空间，公开调用路径因此可以写成 cusignal.argrelextrema。>
from cusignal.peak_finding.peak_finding import argrelextrema, argrelmax, argrelmin
from cusignal.radartools.beamformers import mvdr
from cusignal.radartools.radartools import ambgfun, pulse_compression, pulse_doppler
from cusignal.spectral_analysis.spectral import (
    coherence,
    csd,
    istft,
    lombscargle,
    periodogram,
    spectrogram,
    stft,
    # <学习注释：顶层重新导出 vectorstrength，使用户可以直接调用 cusignal.vectorstrength。>
    vectorstrength,
    welch,
)
from cusignal.utils.arraytools import (
    from_pycuda,
    get_pinned_array,
    get_pinned_mem,
    get_shared_array,
    get_shared_mem,
)
from cusignal.waveforms.waveforms import (
    # <学习注释：把 waveforms.py 中定义的 chirp 暴露到 cusignal 顶层命名空间，因此用户可直接调用 cusignal.chirp。>
    chirp,
    gausspulse,
    sawtooth,
    # <学习注释：把 square 暴露到 cusignal 顶层命名空间，因此示例可以直接调用 cusignal.square(...)。>
    square,
    # <学习注释：把 unit_impulse 暴露到顶层 cusignal 命名空间，使用户可直接调用 cusignal.unit_impulse。>
    unit_impulse,
)
# <学习注释：把 qmf 重新导出到 cusignal 顶层命名空间，因此公开调用形式是 cusignal.qmf(hk)。>
# <学习注释：把 wavelets.py 中的 ricker 提升到 cusignal 顶层命名空间，因此用户可以直接调用 cusignal.ricker。>
# <学习注释：把 cwt 与同模块的小波函数导出到 cusignal 顶层，因此用户可以调用 cusignal.cwt(...)。>
from cusignal.wavelets.wavelets import cwt, morlet, morlet2, qmf, ricker
from cusignal.windows.windows import (
    barthann,
    bartlett,
    blackman,
    blackmanharris,
    bohman,
    boxcar,
    chebwin,
    cosine,
    exponential,
    flattop,
    gaussian,
    general_cosine,
    general_gaussian,
    general_hamming,
    get_window,
    hamming,
    hann,
    # <学习注释：把 windows 子包中的 kaiser 提升到顶层命名空间，使用户可直接调用 cusignal.kaiser。>
    kaiser,
    nuttall,
    # <学习注释：把 windows.py 中的 parzen 提升到 cusignal 顶层命名空间，使用户可直接调用 cusignal.parzen。>
    parzen,
    # <学习注释：把 windows 子包导出的 taylor 再提升到 cusignal 顶层命名空间。>
    taylor,
    triang,
    tukey,
)

# Versioneer
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
