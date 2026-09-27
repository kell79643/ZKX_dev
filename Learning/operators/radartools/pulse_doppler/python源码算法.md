# pulse_doppler Python 源码算法

> 本文件对应阶段二:cuSignal 23.08.00 Python 源码算法。

## 代码定位与接口概览

| 项目 | 内容 |
|------|------|
| 公开导入路径 | `cusignal.pulse_doppler` |
| 顶层包导出 | `cusignal/__init__.py`:59（只读基准行号，下同） |
| 子模块导出 | `cusignal/radartools/__init__.py`:20 |
| 函数定义文件 | `cusignal/radartools/radartools.py`:78-124 |
| 符号名 | `pulse_doppler` |
| 直接依赖 | `cusignal/windows/windows.get_window`（定义于 `windows.py`:2010） |
| 底层依赖 | `cupy.fft.fft`、`cupy.fft.fftfreq`、`cupy.newaxis` |

**调用链：**

```
cusignal.pulse_doppler(x, window, nfft)
  └─ cusignal.radartools.radartools.pulse_doppler(x, window, nfft)
       ├─ cusignal.windows.windows.get_window(window, Nx, False)   [仅 window 不为 None 时]
       ├─ cupy.fft.fftfreq(Nx)                                     [仅 window 为可调用对象时]
       ├─ cupy.multiply(x, W[:, cp.newaxis])                       [仅 window 不为 None 时]
       └─ cupy.fft.fft(x, nfft, axis=0)                            [核心计算]
```

---

## 当前算子的完整相关源码

### 1. 文件头部导入（radartools.py:1-18）

```python
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

from math import ceil, log2

import cupy as cp

from ..windows.windows import get_window
```

### 2. pulse_doppler 函数完整定义（radartools.py:78-124）

```python
def pulse_doppler(x, window=None, nfft=None):
    """
    Pulse doppler processing yields a range/doppler data matrix that represents
    moving target data that's separated from clutter. An estimation of the
    doppler shift can also be obtained from pulse doppler processing. FFT taken
    across slow-time (pulse) dimension.

    Parameters
    ----------
    x : ndarray
        Received signal, assume 2D array with [num_pulses, sample_per_pulse]

    window : array_like, callable, string, float, or tuple, optional
        Specifies the window applied to the signal in the Fourier
        domain.

    nfft : int, size of FFT for pulse compression. Default is number of
        samples per pulse

    Returns
    -------
    pd_dataMatrix : ndarray
        Pulse-doppler output (range/doppler matrix)
    """
    [num_pulses, samples_per_pulse] = x.shape

    if nfft is None:
        nfft = num_pulses

    if window is not None:
        Nx = num_pulses
        if callable(window):
            W = window(cp.fft.fftfreq(Nx))
        elif isinstance(window, cp.ndarray):
            if window.shape != (Nx,):
                raise ValueError("window must have the same length as data")
            W = window
        else:
            W = get_window(window, Nx, False)

        pd_dataMatrix = cp.fft.fft(
            x * W[:, cp.newaxis], nfft, axis=0
        )
    else:
        pd_dataMatrix = cp.fft.fft(x, nfft, axis=0)

    return pd_dataMatrix
```

### 3. 直接依赖：get_window 函数签名（windows.py:2010-2083）

仅摘录签名与参数说明（完整实现过长且服务多个算子）：

```python
def get_window(window, Nx, fftbins=True):
    r"""
    Return a window of a given length and type.

    Parameters
    ----------
    window : string, float, or tuple
        The type of window to create.
    Nx : int
        The number of samples in the window.
    fftbins : bool, optional
        If True (default), create a "periodic" window, ready to use with
        `ifftshift` and be multiplied by the result of an FFT.
        If False, create a "symmetric" window, for use in filter design.

    Returns
    -------
    get_window : ndarray
        Returns a window of length `Nx` and type `window`
    """
```

### 4. 导出入口

**cusignal/radartools/\_\_init\_\_.py:14-21：**

```python
from cusignal.radartools.beamformers import mvdr
from cusignal.radartools.radartools import (
    ambgfun,
    ca_cfar,
    cfar_alpha,
    pulse_compression,
    pulse_doppler,
)
```

**cusignal/\_\_init\_\_.py:59：**

```python
from cusignal.radartools.radartools import ambgfun, pulse_compression, pulse_doppler
```

---

## docstring 逐行翻译与解释

| 行号（只读基准） | 原文 | 翻译与解释 |
|---|---|---|
| 80-83 | `Pulse doppler processing yields a range/doppler data matrix that represents moving target data that's separated from clutter. An estimation of the doppler shift can also be obtained from pulse doppler processing. FFT taken across slow-time (pulse) dimension.` | 脉冲多普勒处理产生一个距离/多普勒数据矩阵，其中运动目标数据与杂波分离。从脉冲多普勒处理中也可获得多普勒频移的估计。FFT 沿慢时间（脉冲）维度执行。 |
| 87-88 | `x : ndarray` / `Received signal, assume 2D array with [num_pulses, sample_per_pulse]` | x：CuPy ndarray，接收信号，假设为 2D 数组，形状为 [脉冲数, 每脉冲采样数]。行 = 慢时间轴，列 = 快时间轴。 |
| 90-92 | `window : array_like, callable, string, float, or tuple, optional` / `Specifies the window applied to the signal in the Fourier domain.` | window：可选窗函数，支持多种类型（数组、可调用对象、字符串、浮点数或元组）。docstring 说"在 Fourier 域施加"，但实际实现是在**时域**（慢时间维）乘以窗函数序列后做 FFT。此描述有误：应为"applied to the signal in the slow-time (pulse) domain before FFT"。 |
| 94-95 | `nfft : int, size of FFT for pulse compression. Default is number of samples per pulse` | nfft：FFT 长度。**docstring 有误**：写的是"Default is number of samples per pulse"（每脉冲采样数），但实际默认值是 `num_pulses`（脉冲数），即慢时间维长度。且描述写"for pulse compression"应为"for pulse doppler"。 |
| 99-100 | `pd_dataMatrix : ndarray` / `Pulse-doppler output (range/doppler matrix)` | 返回值：CuPy ndarray，脉冲多普勒输出，即距离-多普勒矩阵。形状为 [nfft, samples_per_pulse]。 |

**docstring 错误汇总：**
1. 第 92 行：窗函数描述"applied to the signal in the Fourier domain"不准确，实际是时域加窗。
2. 第 94 行：`nfft` 描述"for pulse compression"应为"for pulse doppler processing"。
3. 第 95 行：`nfft` 默认值描述"Default is number of samples per pulse"应为"Default is number of pulses"。

---

## 源代码逐行解释

### 文件头部导入（radartools.py:1-18）

| 行号 | 代码 | 解释 |
|---|---|---|
| 1-12 | `# Copyright ...` | Apache 2.0 版权声明，NVIDIA CORPORATION 2020-2021 |
| 14 | `from math import ceil, log2` | 从 Python 标准库 `math` 导入 `ceil`（向上取整）和 `log2`（以 2 为底的对数）。这两个函数被同文件中的 `ambgfun` 使用，`pulse_doppler` 本身不使用。 |
| 16 | `import cupy as cp` | 导入 CuPy 库并取别名 `cp`。CuPy 是 NVIDIA GPU 上的 NumPy/CUDA 替代库，提供 GPU 加速的数组运算和 FFT。`pulse_doppler` 中所有数组操作都通过 `cp` 执行。 |
| 18 | `from ..windows.windows import get_window` | 从上级包的 `windows/windows.py` 导入 `get_window` 函数。相对导入 `..` 指向 `cusignal/` 包。`get_window` 用于将字符串/元组形式的窗函数名转换为窗函数序列数组。 |

### 函数签名与 docstring（radartools.py:78-101）

| 行号 | 代码 | 解释 |
|---|---|---|
| 78 | `def pulse_doppler(x, window=None, nfft=None):` | 函数定义。三个参数：`x`（接收信号，位置参数，无默认值）、`window`（窗函数，关键字参数，默认 `None` 即不加窗）、`nfft`（FFT 长度，关键字参数，默认 `None` 后在函数体内设为 `num_pulses`）。 |
| 79-101 | `"""..."""` | 完整 docstring，内容见上方"docstring 逐行翻译与解释"小节。 |

### 函数体：维度获取与 nfft 默认值（radartools.py:102-105）

| 行号 | 代码 | 解释 |
|---|---|---|
| 102 | `[num_pulses, samples_per_pulse] = x.shape` | 将输入数组 `x` 的形状解构为两个变量。`x.shape` 返回元组 `(M, N)`，通过列表解构赋值给 `num_pulses`（行数 = 慢时间维长度 = CPI 内脉冲数 $M$）和 `samples_per_pulse`（列数 = 快时间维长度 = 每脉冲距离门数 $N$）。**要求 `x` 必须是 2D 数组**，1D 或 3D 输入会导致 `ValueError`。 |
| 104-105 | `if nfft is None:` / `nfft = num_pulses` | 若用户未指定 FFT 长度，默认设为 `num_pulses`（$M$）。这意味着慢时间 FFT 使用与脉冲数相同的点数，不做零填充。数学含义：DFT 频率分辨率为 $\Delta f_d = \text{PRF}/M$。若用户指定更大的 `nfft`，则零填充至 `nfft` 点，频谱插值但不提高分辨率。 |

### 函数体：窗函数处理分支（radartools.py:107-120）

| 行号 | 代码 | 解释 |
|---|---|---|
| 107 | `if window is not None:` | 判断是否需要加窗。`window is not None` 是 Python 中判断可选参数是否被用户传入的惯用写法，避免了 `window` 为合法 falsy 值时误判的问题。 |
| 108 | `Nx = num_pulses` | 将窗函数长度设为 `num_pulses`（慢时间维长度 $M$）。与 `pulse_compression` 中 `Nx = len(template)` 不同，这里的窗长度由脉冲数而非距离门数决定。物理含义：窗函数沿慢时间（脉冲间）维度施加，长度等于待变换序列的长度。 |
| 109-110 | `if callable(window):` / `W = window(cp.fft.fftfreq(Nx))` | 分支 1：若 `window` 是可调用对象（如自定义 lambda 或函数），则调用它并传入归一化频率数组 `cp.fft.fftfreq(Nx)`。`fftfreq(Nx)` 返回 $[0, 1/N, 2/N, \ldots, (N/2-1)/N, -N/2/N, \ldots, -1/N]$（对 $N$ 点 FFT 的频率 bin 中心），允许用户定义频域窗函数。返回的 `+` 窗函数序列 `W` 是长度为 `Nx` 的 1D CuPy 数组。 |
| 111-113 | `elif isinstance(window, cp.ndarray):` / `if window.shape != (Nx,):` / `raise ValueError("window must have the same length as data")` | 分支 2：若 `window` 已经是 CuPy 数组，直接用作窗函数。首先检查其形状是否为 `(Nx,)`（1D 且长度匹配），不匹配则抛出 `ValueError`。这确保窗函数与慢时间序列等长。 |
| 114 | `W = window` | 通过检查后，将用户提供的数组直接赋值给 `W`。 |
| 115-116 | `else:` / `W = get_window(window, Nx, False)` | 分支 3（默认）：若 `window` 是字符串（如 `"hamming"`）、浮点数（Kaiser 窗的 beta 参数）或元组（如 `("kaiser", 4.0)`），调用 `get_window` 生成窗函数序列。第二个参数 `Nx = num_pulses` 指定窗长，第三个参数 `False` 表示生成"对称窗"（`fftbins=False` → `sym=True`），用于频谱分析而非滤波器设计。 |
| 118-120 | `pd_dataMatrix = cp.fft.fft(` / `x * W[:, cp.newaxis], nfft, axis=0` / `)` | **核心计算（有窗分支）**。先执行逐元素乘法 `x * W[:, cp.newaxis]`，再对结果做 FFT。`W[:, cp.newaxis]` 将 1D 窗函数 `W`（形状 `(M,)`）扩展为 2D 列向量（形状 `(M, 1)`），利用 NumPy/CuPy 广播机制：`x` 形状 `(M, N)` 乘以 `W[:, newaxis]` 形状 `(M, 1)` → 结果形状 `(M, N)`，等效于对每一列（每个距离门）的慢时间序列乘以相同的窗函数 $w[m]$。然后 `cp.fft.fft(..., nfft, axis=0)` 沿行方向（慢时间轴，`axis=0`）做 `nfft` 点 FFT。数学表达：$S[k,n] = \sum_{m=0}^{M-1} x[m,n] \cdot w[m] \cdot e^{-j2\pi km/N_{\text{fft}}}$。 |

### 函数体：无窗分支（radartools.py:121-122）

| 行号 | 代码 | 解释 |
|---|---|---|
| 121-122 | `else:` / `pd_dataMatrix = cp.fft.fft(x, nfft, axis=0)` | **核心计算（无窗分支）**。直接对输入 `x` 沿慢时间轴（`axis=0`）做 `nfft` 点 FFT，等效于使用矩形窗（$w[m] = 1$）。数学表达：$S[k,n] = \sum_{m=0}^{M-1} x[m,n] \cdot e^{-j2\pi km/N_{\text{fft}}}$。 |

### 函数体：返回（radartools.py:124）

| 行号 | 代码 | 解释 |
|---|---|---|
| 124 | `return pd_dataMatrix` | 返回距离-多普勒矩阵。形状为 `(nfft, samples_per_pulse)`，dtype 与输入 `x` 的复数类型相同（若输入为 float 会自动提升为 complex）。 |

---

## 调用链与算法总结

### 完整调用链

```
用户调用
  cusignal.pulse_doppler(x, window="hamming", nfft=256)
    │
    ▼
cusignal/__init__.py:59  →  cusignal.radartools.radartools.pulse_doppler
    │
    ▼
radartools.py:102        →  解构 x.shape 为 [num_pulses, samples_per_pulse]
    │
    ▼
radartools.py:104-105    →  若 nfft 为 None，设 nfft = num_pulses
    │
    ▼
radartools.py:107        →  判断 window 是否为 None
    ├── 不为 None ──────────────────────────────────────────┐
    │   radartools.py:108  →  Nx = num_pulses               │
    │   radartools.py:109  →  callable? → W = window(...)    │
    │   radartools.py:111  →  ndarray?  → 长度检查 → W       │
    │   radartools.py:115  →  else     → W = get_window(...) │
    │   radartools.py:118  →  fft(x * W[:, newaxis], nfft,  │
    │                            axis=0)                     │
    ├── 为 None ─────────────────────────────────────────────┤
    │   radartools.py:122  →  fft(x, nfft, axis=0)          │
    │                                                        │
    ▼                                                        ▼
radartools.py:124         →  return pd_dataMatrix
```

### 算法总结

`pulse_doppler` 的核心算法极为简洁：**沿 2D 输入矩阵的慢时间轴（行方向）执行一维 FFT**，可选地在 FFT 前对慢时间序列施加窗函数。这实现了脉冲多普勒雷达中从脉冲回波数据到距离-多普勒图的转换。

数学表达：

$$\text{pd\_dataMatrix}[k, n] = \sum_{m=0}^{M-1} x[m, n] \cdot w[m] \cdot e^{-j2\pi km / N_{\text{fft}}}$$

其中 $w[m] = 1$（无窗）或由 `get_window` 生成的窗函数序列（有窗）。

---

## 数学映射、边界、复杂度和阅读检查

### 数学公式与代码的逐项映射

| 数学概念 | 代码实现 | 行号 |
|----------|----------|------|
| 慢时间序列 $x[m, n]$（第 $n$ 个距离门的 $M$ 个脉冲采样） | 输入数组 `x` 的第 $n$ 列 `x[:, n]`，形状 `(M,)` | 102 |
| 窗函数 $w[m]$（长度 $M$） | `W = get_window(window, num_pulses, False)` 或用户提供的数组/可调用对象结果 | 108-116 |
| 加窗 $x_w[m, n] = x[m, n] \cdot w[m]$ | `x * W[:, cp.newaxis]`，广播乘法 | 119 |
| DFT $S[k] = \sum_{m} x_w[m] \cdot e^{-j2\pi km/N}$ | `cp.fft.fft(x_w, nfft, axis=0)` | 118-120 |
| 零填充（$N_{\text{fft}} > M$） | `cp.fft.fft` 的 `nfft` 参数自动对输入零填充 | 104-105 |

### 边界处理与异常检查

| 情况 | 处理方式 | 行号 |
|------|----------|------|
| `x` 非 2D | `x.shape` 解构会抛出 `ValueError: not enough values to unpack` | 102 |
| `window` 为 CuPy 数组但长度不匹配 | 抛出 `ValueError("window must have the same length as data")` | 112-113 |
| `window` 为字符串但不被 `get_window` 识别 | `get_window` 内部抛出 `ValueError` | 116 |
| `nfft` 小于 `num_pulses` | CuPy FFT 允许，等效于对输入截断后做 FFT，物理上无意义但不会报错 | 104-105 |
| `nfft` 为 0 或负数 | CuPy FFT 抛出 `ValueError` | — |
| 输入 `x` 为实数类型 | CuPy FFT 自动输出复数类型（float32→complex64，float64→complex128） | 118/122 |

### 数值稳定性与 dtype 转换

- `cp.fft.fft` 对实数输入自动提升为对应复数类型，输出始终为复数。
- 窗函数 `W` 由 `get_window` 生成，dtype 通常为 float64。
- 乘法 `x * W[:, cp.newaxis]` 涉及复数×实数广播，CuPy 遵循 NumPy 的类型提升规则：complex64 × float64 → complex128。
- 无显式 dtype 转换代码，用户需注意输入精度。

### 时间复杂度与空间复杂度

| 操作 | 时间复杂度 | 说明 |
|------|------------|------|
| 窗函数生成 `get_window` | $O(M)$ | $M$ = num_pulses |
| 加窗乘法 `x * W[:, newaxis]` | $O(M \times N)$ | 逐元素乘，$N$ = samples_per_pulse |
| FFT `cp.fft.fft(..., axis=0)` | $O(N \times N_{\text{fft}} \log N_{\text{fft}})$ | 对 $N$ 列分别做 $N_{\text{fft}}$ 点 FFT |
| **总计** | $O(N \times N_{\text{fft}} \log N_{\text{fft}})$ | FFT 占主导 |

空间复杂度：输出数组 `pd_dataMatrix` 大小为 $N_{\text{fft}} \times N$，加窗时的临时数组同大小，总额外空间 $O(N_{\text{fft}} \times N)$。

**性能瓶颈：** FFT 是计算密集型操作，cuSignal 使用 CuPy 的 GPU FFT 实现（基于 cuFFT），相比 NumPy 的 CPU FFT 有显著加速。当 `nfft` 很大时，GPU 显存可能成为瓶颈。

### 与 CPU 参考实现的差异

测试文件中的 `cpu_pulse_doppler`（`test_radartools_cpu.py`:53-78）是 NumPy 参考实现，与 GPU 实现的主要差异：

| 差异点 | GPU 版本（cuSignal） | CPU 版本（测试参考） |
|--------|----------------------|----------------------|
| 数组库 | CuPy (`cp`) | NumPy (`np`) |
| 窗函数来源 | `cusignal.windows.windows.get_window` | `scipy.signal.get_window` |
| 加窗广播方式 | `x * W[:, cp.newaxis]`（CuPy newaxis 广播） | `np.multiply(x, np.tile(W.T, (1, samples_per_pulse)))`（显式 tile 复制） |
| FFT | `cp.fft.fft`（GPU cuFFT） | `np.fft.fft`（CPU FFTPACK/FFTW） |
| 可调用窗函数频率 | `cp.fft.fftfreq(Nx)`（CuPy） | `np.fft.fftfreq(Nx)`（NumPy） |

数学上 GPU 和 CPU 实现完全等价，差异仅在工程层面（CuPy vs NumPy、广播 vs tile）。

---

## 建议阅读顺序

1. **先看函数签名与 docstring**（radartools.py:78-101）：理解输入输出约定。注意 docstring 中的三处错误。
2. **看维度解构与 nfft 默认值**（radartools.py:102-105）：理解数据矩阵的行列含义和 FFT 长度选择。
3. **看窗函数三分支**（radartools.py:107-116）：理解 `callable`/`ndarray`/`else` 三种窗函数传入方式，特别注意 `Nx = num_pulses`（窗长度等于脉冲数，不是距离门数）。
4. **看核心 FFT 调用**（radartools.py:118-122）：这是算法核心。有窗时先乘后 FFT，无窗时直接 FFT。`axis=0` 是关键——沿慢时间轴。
5. **看广播 `W[:, cp.newaxis]`**：理解 1D 窗函数如何通过广播扩展为与 `x` 相同形状的 2D 数组。
6. **对照 `数学物理原理.md` 的原理 3**：将 DFT 公式与代码中的 `cp.fft.fft(..., axis=0)` 逐项对应。
7. **看测试代码**（test_radartools_cpu.py:146-192）：理解典型用法和正确性验证方式。

**阅读时应注意的问题：**
- 为什么 FFT 沿 `axis=0`（行方向）而不是 `axis=1`（列方向）？——因为行方向是慢时间维（脉冲间），列方向是快时间维（距离门间）。
- 窗函数长度为什么是 `num_pulses` 而不是 `samples_per_pulse`？——窗函数在慢时间维施加，长度等于待变换序列长度。
- `nfft` 默认值为什么是 `num_pulses` 而非 `samples_per_pulse`？——慢时间 FFT 的长度由脉冲数决定，docstring 此处有误。
