# hilbert Python 源码算法

> 当前阶段：阶段二（cuSignal 23.08.00 Python 源码算法）  
> 学习对象：`hilbert`，不包含相邻的 `hilbert2`  
> 核对日期：2026-08-06

## 1. 代码定位与接口概览

### 1.1 完整定位

| 层次 | 共享根目录 `ZKX_dev/` 下的路径 | 符号 | 只读基准行号 | 带注释学习副本行号 |
| --- | --- | --- | --- | --- |
| 顶层公开导出 | `ZKX/cusignal-23.08.00/python/cusignal/__init__.py` / `Learning/cusignal-23.08.00/python/cusignal/__init__.py` | 导入项 `hilbert` | 39-53；`hilbert` 在 47 | 41-59；`hilbert` 在 52 |
| 子包公开导出 | `ZKX/cusignal-23.08.00/python/cusignal/filtering/__init__.py` / `Learning/cusignal-23.08.00/python/cusignal/filtering/__init__.py` | 导入项 `hilbert` | 14-28；`hilbert` 在 22 | 14-32；`hilbert` 在 25 |
| 直接 kernel | `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py` / `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py` | `_hilbert_kernel` | 756-784 | 852-893 |
| API 定义 | 同上 | `hilbert` | 787-892 | 897-1015 |

学习副本行号会因其他算子的合法学习注释而增加；只读基准行号保持不变。本表是写文档时重新映射的结果。

### 1.2 公开调用方式

```python
import cusignal
xa = cusignal.hilbert(x, N=None, axis=-1)
```

也可从子包导入：

```python
from cusignal.filtering import hilbert
xa = hilbert(x)
```

### 1.3 签名、参数和返回值

```python
hilbert(x, N=None, axis=-1)
```

| 项目 | 语义 |
| --- | --- |
| `x` | `array_like` 实值信号。函数先用 `cp.asarray` 转成 CuPy 数组；复数 dtype 会抛出 `ValueError`。可为一维或多维。 |
| `N` | 可选 FFT 长度。`None` 时取 `x.shape[axis]`；`N > 原长度` 时由 FFT 尾部补零，`N < 原长度` 时截断；`N <= 0` 被拒绝。 |
| `axis` | 执行一维 Hilbert/解析信号构造的轴，默认 `-1`，即最后一轴。负索引遵循 Python/CuPy 轴索引规则。 |
| 返回值 | 复值 CuPy `ndarray`。除目标轴长度变为 `N` 外，其余维度保持不变；实部对应截断或补零后的输入，虚部是有限 DFT 周期模型下的 Hilbert 变换。 |
| dtype | `h` 明确使用 `x.dtype`；FFT/IFFT 的复数输出 dtype 由 CuPy FFT 对输入 dtype 的规则决定。源码没有手动强制成某个固定复数 dtype。 |

### 1.4 进入源码阶段前的完整性门禁

阶段开始时得到：

- `filtering/filtering.py`：学习副本与只读基准 SHA-256 完全相同；
- `filtering/__init__.py`：完全相同；
- 顶层 `cusignal/__init__.py`：存在其他算子的既有学习注释；只忽略格式严格匹配 `# <学习注释：...>` 的独立注释行后，与只读基准逐行相同；
- 插入本算子注释后，三个文件再次执行同一检查，均满足“剔除合法学习注释后与基准逐行相同”。

## 2. 当前算子的完整相关源码

以下摘录全部来自只读基准 `ZKX/cusignal-23.08.00`，保持原文、空行和顺序；没有用省略号删减 `hilbert` 的 docstring、示例或函数体。相邻算子不在本次边界内。

### 2.1 顶层公开导出

基准：`ZKX/cusignal-23.08.00/python/cusignal/__init__.py:39`

```python
from cusignal.filtering.filtering import (
    channelize_poly,
    detrend,
    filtfilt,
    firfilter,
    firfilter2,
    firfilter_zi,
    freq_shift,
    hilbert,
    hilbert2,
    lfilter,
    lfilter_zi,
    sosfilt,
    wiener,
)
```

### 2.2 filtering 子包公开导出

基准：`ZKX/cusignal-23.08.00/python/cusignal/filtering/__init__.py:14`

```python
from cusignal.filtering.filtering import (
    channelize_poly,
    detrend,
    filtfilt,
    firfilter,
    firfilter2,
    firfilter_zi,
    freq_shift,
    hilbert,
    hilbert2,
    lfilter,
    lfilter_zi,
    sosfilt,
    wiener,
)
```

### 2.3 直接依赖 `_hilbert_kernel`

基准：`ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:756`

```python
_hilbert_kernel = cp.ElementwiseKernel(
    "",
    "T h",
    """
    if ( !odd ) {
        if ( ( i == 0 ) || ( i == bend ) ) {
            h = 1.0;
        } else if ( i > 0 && i < bend ) {
            h = 2.0;
        } else {
            h = 0.0;
        }
    } else {
        if ( i == 0 ) {
            h = 1.0;
        } else if ( i > 0 && i < bend) {
            h = 2.0;
        } else {
            h = 0.0;
        }
    }
    """,
    "_hilbert_kernel",
    options=("-std=c++11",),
    loop_prep="const bool odd { _ind.size() & 1 }; \
               const int bend = odd ? \
                   static_cast<int>( 0.5 * ( _ind.size()  + 1 ) ) : \
                   static_cast<int>( 0.5 * _ind.size() );",
)
```

### 2.4 `hilbert` 完整定义、docstring、示例与函数体

基准：`ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:787`

```python
def hilbert(x, N=None, axis=-1):
    """
    Compute the analytic signal, using the Hilbert transform.

    The transformation is done along the last axis by default.

    Parameters
    ----------
    x : array_like
        Signal data.  Must be real.
    N : int, optional
        Number of Fourier components.  Default: ``x.shape[axis]``
    axis : int, optional
        Axis along which to do the transformation.  Default: -1.

    Returns
    -------
    xa : ndarray
        Analytic signal of `x`, of each 1-D array along `axis`

    Notes
    -----
    The analytic signal ``x_a(t)`` of signal ``x(t)`` is:

    .. math:: x_a = F^{-1}(F(x) 2U) = x + i y

    where `F` is the Fourier transform, `U` the unit step function,
    and `y` the Hilbert transform of `x`. [1]_

    In other words, the negative half of the frequency spectrum is zeroed
    out, turning the real-valued signal into a complex signal.  The Hilbert
    transformed signal can be obtained from ``cp.imag(hilbert(x))``, and the
    original signal from ``cp.real(hilbert(x))``.

    Examples
    ---------
    In this example we use the Hilbert transform to determine the amplitude
    envelope and instantaneous frequency of an amplitude-modulated signal.

    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt
    >>> from cusignal import hilbert, chirp

    >>> duration = 1.0
    >>> fs = 400.0
    >>> samples = int(fs*duration)
    >>> t = cp.arange(samples) / fs

    We create a chirp of which the frequency increases from 20 Hz to 100 Hz and
    apply an amplitude modulation.

    >>> signal = chirp(t, 20.0, t[-1], 100.0)
    >>> signal *= (1.0 + 0.5 * cp.sin(2.0*cp.pi*3.0*t) )

    The amplitude envelope is given by magnitude of the analytic signal. The
    instantaneous frequency can be obtained by differentiating the
    instantaneous phase in respect to time. The instantaneous phase corresponds
    to the phase angle of the analytic signal.

    >>> analytic_signal = hilbert(signal)
    >>> amplitude_envelope = cp.abs(analytic_signal)
    >>> instantaneous_phase = cp.unwrap(cp.angle(analytic_signal))
    >>> instantaneous_frequency = (cp.diff(instantaneous_phase) /
    ...                            (2.0*cp.pi) * fs)

    >>> fig = plt.figure()
    >>> ax0 = fig.add_subplot(211)
    >>> ax0.plot(cp.asnumpy(t), cp.asnumpy(signal), label='signal')
    >>> ax0.plot(cp.asnumpy(t), cp.asnumpy(amplitude_envelope), \
        label='envelope')
    >>> ax0.set_xlabel("time in seconds")
    >>> ax0.legend()
    >>> ax1 = fig.add_subplot(212)
    >>> ax1.plot(t[1:], instantaneous_frequency)
    >>> ax1.set_xlabel("time in seconds")
    >>> ax1.set_ylim(0.0, 120.0)

    References
    ----------
    .. [1] Wikipedia, "Analytic signal".
           https://en.wikipedia.org/wiki/Analytic_signal
    .. [2] Leon Cohen, "Time-Frequency Analysis", 1995. Chapter 2.
    .. [3] Alan V. Oppenheim, Ronald W. Schafer. Discrete-Time Signal
           Processing, Third Edition, 2009. Chapter 12.
           ISBN 13: 978-1292-02572-8

    """
    x = cp.asarray(x)
    if cp.iscomplexobj(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")

    Xf = cp.fft.fft(x, N, axis=axis)

    h = cp.empty((N,), dtype=x.dtype)
    _hilbert_kernel(h)

    if x.ndim > 1:
        ind = [cp.newaxis] * x.ndim
        ind[axis] = slice(None)
        h = h[tuple(ind)]
    x = cp.fft.ifft(Xf * h, axis=axis)
    return x
```

## 3. docstring 逐行翻译与解释

空行只负责分段，按规则不单列。定位格式为“基准行 / 学习副本行”。

#### 文档字符串

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:898` 至 `:899`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:788` 至 `:789`。

```python
    """
    Compute the analytic signal, using the Hilbert transform.
```

逐行对应说明：

- 第 1 行：开始 Python docstring；它是函数对象的 `__doc__`，不是普通注释。
- 第 2 行：“使用 Hilbert 变换计算解析信号。”明确返回目标是解析信号。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:901` 至 `:901`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:791` 至 `:791`。

```python
    The transformation is done along the last axis by default.
```

逐行对应说明：

- 第 1 行：默认沿最后一个轴逐条变换，对应 `axis=-1`。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:903` 至 `:910`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:793` 至 `:800`。

```python
    Parameters
    ----------
    x : array_like
        Signal data.  Must be real.
    N : int, optional
        Number of Fourier components.  Default: ``x.shape[axis]``
    axis : int, optional
        Axis along which to do the transformation.  Default: -1.
```

逐行对应说明：

- 第 1 行：NumPy/SciPy 风格文档的参数章节标题。
- 第 2 行：reStructuredText 标题下划线。
- 第 3 行：参数 `x` 接受可转换为数组的对象。
- 第 4 行：信号必须是实值；函数体会显式检查。
- 第 5 行：`N` 是可选整数，表示 FFT 长度。
- 第 6 行：缺省值取目标轴长度；双反引号是文档中的行内代码标记。
- 第 7 行：`axis` 是可选整数轴索引。
- 第 8 行：默认 `-1`，即最后一轴。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:912` 至 `:915`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:802` 至 `:805`。

```python
    Returns
    -------
    xa : ndarray
        Analytic signal of `x`, of each 1-D array along `axis`
```

逐行对应说明：

- 第 1 行：返回值章节标题。
- 第 2 行：返回值标题下划线。
- 第 3 行：返回数组命名为 `xa`；实际是 CuPy `ndarray`。
- 第 4 行：对沿目标轴看到的每一条一维序列分别构造解析信号。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:917` 至 `:919`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:807` 至 `:809`。

```python
    Notes
    -----
    The analytic signal ``x_a(t)`` of signal ``x(t)`` is:
```

逐行对应说明：

- 第 1 行：数学说明章节。
- 第 2 行：Notes 标题下划线。
- 第 3 行：引入原信号 $x(t)$ 与解析信号 $x_a(t)$；双反引号标识代码/数学名称。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:921` 至 `:921`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:811` 至 `:811`。

```python
    .. math:: x_a = F^{-1}(F(x) 2U) = x + i y
```

逐行对应说明：

- 第 1 行：reStructuredText 数学指令：频谱乘 $2U$ 后逆变换，等于原信号加 $i$ 倍 Hilbert 分量。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:923` 至 `:924`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:813` 至 `:814`。

```python
    where `F` is the Fourier transform, `U` the unit step function,
    and `y` the Hilbert transform of `x`. [1]_
```

逐行对应说明：

- 第 1 行：$F$ 表示 Fourier 变换，$U$ 表示单位阶跃。
- 第 2 行：$y$ 是 $x$ 的 Hilbert 变换；`[1]_` 引用文末第一条资料。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:926` 至 `:929`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:816` 至 `:819`。

```python
    In other words, the negative half of the frequency spectrum is zeroed
    out, turning the real-valued signal into a complex signal.  The Hilbert
    transformed signal can be obtained from ``cp.imag(hilbert(x))``, and the
    original signal from ``cp.real(hilbert(x))``.
```

逐行对应说明：

- 第 1 行：开始用频谱语言解释：负频率部分被清零。
- 第 2 行：上句续行：实信号由此变成复解析信号。
- 第 3 行：返回值虚部才是纯 Hilbert 变换。
- 第 4 行：返回值实部恢复原信号（按 `N` 截断/补零后的版本）。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:931` 至 `:934`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:821` 至 `:824`。

```python
    Examples
    ---------
    In this example we use the Hilbert transform to determine the amplitude
    envelope and instantaneous frequency of an amplitude-modulated signal.
```

逐行对应说明：

- 第 1 行：示例章节标题。
- 第 2 行：Examples 标题下划线。
- 第 3 行：示例目标之一是求幅度包络。
- 第 4 行：另一个目标是求调幅信号的瞬时频率。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:936` 至 `:938`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:826` 至 `:828`。

```python
    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt
    >>> from cusignal import hilbert, chirp
```

逐行对应说明：

- 第 1 行：doctest 提示符 `>>>`；导入 CuPy 并命名为 `cp`。
- 第 2 行：导入绘图库接口为 `plt`。
- 第 3 行：从顶层公开入口导入本算子和用于造信号的 `chirp`。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:940` 至 `:943`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:830` 至 `:833`。

```python
    >>> duration = 1.0
    >>> fs = 400.0
    >>> samples = int(fs*duration)
    >>> t = cp.arange(samples) / fs
```

逐行对应说明：

- 第 1 行：设持续时间 1 s。
- 第 2 行：设采样率 400 Hz。
- 第 3 行：样本数为 $400\times1=400$；`int` 转为整数。
- 第 4 行：生成 $0,1,\ldots,399$ 并除以采样率，得到秒单位时间轴。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:945` 至 `:946`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:835` 至 `:836`。

```python
    We create a chirp of which the frequency increases from 20 Hz to 100 Hz and
    apply an amplitude modulation.
```

逐行对应说明：

- 第 1 行：说明将创建从 20 Hz 增至 100 Hz 的 chirp；句子在下一行继续。
- 第 2 行：再叠加幅度调制。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:948` 至 `:949`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:838` 至 `:839`。

```python
    >>> signal = chirp(t, 20.0, t[-1], 100.0)
    >>> signal *= (1.0 + 0.5 * cp.sin(2.0*cp.pi*3.0*t) )
```

逐行对应说明：

- 第 1 行：在时间 `t` 上生成 20→100 Hz chirp；`t[-1]` 是扫频终止时刻。
- 第 2 行：原地乘以 $1+0.5\sin(2\pi\cdot3t)$，形成 3 Hz、调制度 0.5 的 AM 包络。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:951` 至 `:954`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:841` 至 `:844`。

```python
    The amplitude envelope is given by magnitude of the analytic signal. The
    instantaneous frequency can be obtained by differentiating the
    instantaneous phase in respect to time. The instantaneous phase corresponds
    to the phase angle of the analytic signal.
```

逐行对应说明：

- 第 1 行：说明解析信号的模用于估计幅度包络。
- 第 2 行：瞬时频率来自相位对时间的导数；句子续行。
- 第 3 行：瞬时相位是解析信号相位；句子续行。
- 第 4 行：完成上一句。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:956` 至 `:960`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:846` 至 `:850`。

```python
    >>> analytic_signal = hilbert(signal)
    >>> amplitude_envelope = cp.abs(analytic_signal)
    >>> instantaneous_phase = cp.unwrap(cp.angle(analytic_signal))
    >>> instantaneous_frequency = (cp.diff(instantaneous_phase) /
    ...                            (2.0*cp.pi) * fs)
```

逐行对应说明：

- 第 1 行：调用本函数得到复解析信号。
- 第 2 行：复数模 $\sqrt{\Re^2+\Im^2}$ 得包络估计。
- 第 3 行：`angle` 求主值相位，`unwrap` 消除相邻 $2\pi$ 跳变。
- 第 4 行：`cp.diff` 做相邻相位差；左括号允许表达式跨行。
- 第 5 行：除 $2\pi$ 将 rad 转为周，再乘采样率转为 Hz；`...` 是 doctest 续行提示符。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:962` 至 `:972`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:852` 至 `:862`。

```python
    >>> fig = plt.figure()
    >>> ax0 = fig.add_subplot(211)
    >>> ax0.plot(cp.asnumpy(t), cp.asnumpy(signal), label='signal')
    >>> ax0.plot(cp.asnumpy(t), cp.asnumpy(amplitude_envelope), \
        label='envelope')
    >>> ax0.set_xlabel("time in seconds")
    >>> ax0.legend()
    >>> ax1 = fig.add_subplot(212)
    >>> ax1.plot(t[1:], instantaneous_frequency)
    >>> ax1.set_xlabel("time in seconds")
    >>> ax1.set_ylim(0.0, 120.0)
```

逐行对应说明：

- 第 1 行：新建 Matplotlib Figure。
- 第 2 行：创建 2 行 1 列中的第 1 个子图。
- 第 3 行：`cp.asnumpy` 把 GPU 数组移到主机 NumPy 数组后绘制原信号。
- 第 4 行：开始绘制包络；反斜杠显式续行。
- 第 5 行：结束调用并设置图例标签。注意该行在 docstring 中不是 `>>>`，依靠上一行续行。
- 第 6 行：设置第一幅图横轴为秒。
- 第 7 行：显示信号和包络图例。
- 第 8 行：创建第 2 个子图。
- 第 9 行：相位差少一个样本，所以时间轴也从第 2 个点开始。源码示例这里未显式 `asnumpy`。
- 第 10 行：设置第二幅图横轴标签。
- 第 11 行：把频率显示范围设为 0–120 Hz。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:974` 至 `:981`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:864` 至 `:871`。

```python
    References
    ----------
    .. [1] Wikipedia, "Analytic signal".
           https://en.wikipedia.org/wiki/Analytic_signal
    .. [2] Leon Cohen, "Time-Frequency Analysis", 1995. Chapter 2.
    .. [3] Alan V. Oppenheim, Ronald W. Schafer. Discrete-Time Signal
           Processing, Third Edition, 2009. Chapter 12.
           ISBN 13: 978-1292-02572-8
```

逐行对应说明：

- 第 1 行：参考资料章节。
- 第 2 行：References 标题下划线。
- 第 3 行：reStructuredText 第一条脚注定义。
- 第 4 行：第一条资料链接。
- 第 5 行：第二条资料：Cohen 的时频分析教材第 2 章。
- 第 6 行：第三条资料第一行：Oppenheim 与 Schafer 的离散时间信号处理教材。
- 第 7 行：第三条资料续行：第三版、第 12 章。
- 第 8 行：给出书籍 ISBN。

#### 文档字符串

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:983` 至 `:983`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:873` 至 `:873`。

```python
    """
```

逐行对应说明：

- 第 1 行：结束 docstring；下一行开始真正执行的函数体。


## 4. 源代码逐行解释

### 4.1 两处公开导出逐行解释

两个导出块结构相同。下表仍逐行列出，避免把 `hilbert` 周围的原始导入语句省略掉。

#### 公开导入或依赖导入

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/__init__.py:46` 至 `:61`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:39` 至 `:50`。

```python
from cusignal.filtering.filtering import (
    channelize_poly,
    detrend,
    filtfilt,
    firfilter,
    firfilter2,
    firfilter_zi,
    freq_shift,
    hilbert,
    hilbert2,
    lfilter,
    lfilter_zi,
```

逐行对应说明：

- 第 1 行：从实现模块开始一个括号包围的多行导入；括号允许隐式续行。
- 第 2 行：同块导入 `channelize_poly`；尾逗号分隔名称。
- 第 3 行：同块导入 `detrend`。
- 第 4 行：同块导入 `filtfilt`。
- 第 5 行：同块导入 `firfilter`；学习行号因其上方既有注释偏移。
- 第 6 行：同块导入 `firfilter2`。
- 第 7 行：同块导入 `firfilter_zi`。
- 第 8 行：同块导入 `freq_shift`。
- 第 9 行：本算子的关键导出：分别建立 `cusignal.hilbert` 和 `cusignal.filtering.hilbert` 名称。它不包装函数，只绑定同一对象。
- 第 10 行：同块导入相邻二维 API；不属于本次算法边界。
- 第 11 行：同块导入 `lfilter`。
- 第 12 行：同块导入 `lfilter_zi`。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/__init__.py:63` 至 `:65`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:51` 至 `:53`。

```python
    sosfilt,
    wiener,
)
```

逐行对应说明：

- 第 1 行：同块导入 `sosfilt`。
- 第 2 行：同块导入 `wiener`。
- 第 3 行：结束多行导入语句。


### 4.2 `_hilbert_kernel` 逐行解释

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:852` 至 `:857`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:756` 至 `:759`。

```python
_hilbert_kernel = cp.ElementwiseKernel(
    "",
    "T h",
    """
```

逐行对应说明：

- 第 1 行：调用 CuPy 工厂创建逐元素 GPU kernel，并把对象绑定到模块私有名称。前导下划线表示非公开 helper。
- 第 2 行：`in_params` 为空：没有显式输入数组。kernel 的元素个数由输出 `h` 决定。
- 第 3 行：`out_params` 声明输出 `h`，`T` 是由实际数组 dtype 推导的模板占位类型。
- 第 4 行：开始传给运行时编译器的多行 C/CUDA 操作字符串。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:859` 至 `:859`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:760` 至 `:760`。

```python
    if ( !odd ) {
```

逐行对应说明：

- 第 1 行：C/C++ 条件；`!` 取逻辑非。`odd=false` 即 $N$ 为偶数。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:861` 至 `:871`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:761` 至 `:768`。

```python
        if ( ( i == 0 ) || ( i == bend ) ) {
            h = 1.0;
        } else if ( i > 0 && i < bend ) {
            h = 2.0;
        } else {
            h = 0.0;
        }
    } else {
```

逐行对应说明：

- 第 1 行：
- 第 2 行：对 DC/Nyquist 写 1，保持这些自共轭频点。分号结束 C/C++ 语句。
- 第 3 行：`&&` 是逻辑与；$0<i<N/2$ 对应严格正频率。前 `}` 关闭上一个分支。
- 第 4 行：正频率权重写 2，补偿随后删除的共轭负频率。
- 第 5 行：其余偶数长度索引进入兜底分支。
- 第 6 行：$i>N/2$ 的负频率权重置零。
- 第 7 行：关闭偶数长度的内层条件。
- 第 8 行：关闭 `!odd` 分支并开始奇数长度分支。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:873` 至 `:889`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:769` 至 `:780`。

```python
        if ( i == 0 ) {
            h = 1.0;
        } else if ( i > 0 && i < bend) {
            h = 2.0;
        } else {
            h = 0.0;
        }
    }
    """,
    "_hilbert_kernel",
    options=("-std=c++11",),
    loop_prep="const bool odd { _ind.size() & 1 }; \
```

逐行对应说明：

- 第 1 行：奇数长度时先识别唯一的 DC bin。
- 第 2 行：DC 保持权重 1。
- 第 3 行：$0<i<(N+1)/2$，即索引 $1$ 到 $(N-1)/2$，均为严格正频率。`bend)` 前少一个空格只是风格，不改变语法。
- 第 4 行：奇数长度的正频率权重写 2。
- 第 5 行：其余奇数长度索引进入负频率分支。
- 第 6 行：负频率权重写 0。
- 第 7 行：关闭奇数长度内层条件。
- 第 8 行：关闭最外层奇偶条件。
- 第 9 行：结束操作字符串，并以逗号分隔下一个 Python 实参。
- 第 10 行：`name` 参数，决定生成 kernel 的名字和缓存标识。
- 第 11 行：传给后端编译器 C++11 选项；单元素 tuple 必须有尾逗号。
- 第 12 行：开始 `loop_prep` 字符串。`_ind.size()` 是元素总数 $N$；位与 `& 1` 检测最低位，转换为 `bool` 得奇偶性。反斜杠续接 Python 源码行。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:890` 至 `:893`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:781` 至 `:784`。

```python
               const int bend = odd ? \
                   static_cast<int>( 0.5 * ( _ind.size()  + 1 ) ) : \
                   static_cast<int>( 0.5 * _ind.size() );",
)
```

逐行对应说明：

- 第 1 行：声明常量整数 `bend`；`?:` 是 C/C++ 三元运算符，条件为 `odd`。
- 第 2 行：奇数分支计算 $(N+1)/2$ 并显式转成 `int`；冒号引出偶数分支。
- 第 3 行：偶数分支计算 $N/2$，结束 C++ 语句和 Python 字符串。
- 第 4 行：结束 `cp.ElementwiseKernel(...)` 调用；模块导入时 kernel 对象即被创建。


### 4.3 `hilbert` 函数体逐行解释

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:897` 至 `:897`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:787` 至 `:787`。

```python
def hilbert(x, N=None, axis=-1):
```

逐行对应说明：

- 第 1 行：用 `def` 定义函数。`x` 必填；`N` 默认 `None`；`axis` 默认 `-1`；冒号开始缩进函数体。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:985` 至 `:985`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:874` 至 `:874`。

```python
    x = cp.asarray(x)
```

逐行对应说明：

- 第 1 行：将输入变为 CuPy 数组并重新绑定局部变量 `x`。已有兼容数组可避免额外复制；主机数据通常需传入设备。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:987` 至 `:988`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:875` 至 `:876`。

```python
    if cp.iscomplexobj(x):
        raise ValueError("x must be real.")
```

逐行对应说明：

- 第 1 行：判断 dtype 是否为复数类型；结果为真时执行下一缩进块。
- 第 2 行：主动抛出值错误并立即终止函数，保证输入为实值语义。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:990` 至 `:991`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:877` 至 `:878`。

```python
    if N is None:
        N = x.shape[axis]
```

逐行对应说明：

- 第 1 行：用身份比较判断调用者是否省略 `N`。`is None` 不会调用相等运算重载。
- 第 2 行：从 shape tuple 取目标轴长度作为默认 FFT 点数；非法轴由此处或 FFT 后端报错。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:993` 至 `:994`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:879` 至 `:880`。

```python
    if N <= 0:
        raise ValueError("N must be positive.")
```

逐行对应说明：

- 第 1 行：检查 FFT 长度是否非正；源码没有额外写出整数类型检查。
- 第 2 行：非正时抛出 `ValueError`。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:997` 至 `:997`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:882` 至 `:882`。

```python
    Xf = cp.fft.fft(x, N, axis=axis)
```

逐行对应说明：

- 第 1 行：沿 `axis` 做 $N$ 点复 FFT。`N` 大于轴长时尾部补零，小于时截断；结果绑定为 `Xf`。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1000` 至 `:1002`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:884` 至 `:885`。

```python
    h = cp.empty((N,), dtype=x.dtype)
    _hilbert_kernel(h)
```

逐行对应说明：

- 第 1 行：在设备上分配一维未初始化数组。`(N,)` 的逗号表示一维 shape tuple；dtype 跟随输入。
- 第 2 行：把 `h` 作为输出调用 kernel；每个元素根据自身索引被写成 0、1 或 2，所以 `empty` 的旧内容不会遗留。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1005` 至 `:1013`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:887` 至 `:891`。

```python
    if x.ndim > 1:
        ind = [cp.newaxis] * x.ndim
        ind[axis] = slice(None)
        h = h[tuple(ind)]
    x = cp.fft.ifft(Xf * h, axis=axis)
```

逐行对应说明：

- 第 1 行：只有多维输入才需显式调整 mask 形状；一维时 `h.shape == (N,)` 已匹配。
- 第 2 行：创建长度为维数的列表，每项为 `None/newaxis`。列表乘法重复引用即可，因为元素不可变。
- 第 3 行：把目标轴项改为完整切片 `:`；负轴索引可直接索引 Python 列表。
- 第 4 行：将列表转 tuple 后做基础索引。例如三维、`axis=1` 得形状 `(1,N,1)`，便于广播。
- 第 5 行：先逐元素广播相乘，再沿同一轴 IFFT。频域 mask 删除负频、加倍正频，结果覆盖局部变量 `x`。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:1015` 至 `:1015`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:892` 至 `:892`。

```python
    return x
```

逐行对应说明：

- 第 1 行：返回复解析信号并结束函数。


## 5. 调用链与算法总结

### 5.1 调用链

```text
cusignal.hilbert / cusignal.filtering.hilbert
    ↓（两个 __init__.py 只是名称导出）
filtering.py::hilbert(x, N, axis)
    ↓ cp.asarray + 实值/N 边界检查
cp.fft.fft(x, N, axis)
    ↓
cp.empty((N,), dtype=x.dtype)
    ↓
filtering.py::_hilbert_kernel(h)
    ↓ 生成奇偶长度对应的 [1, 2, ..., 1/0, ..., 0] mask
Xf * h（多维时广播）
    ↓
cp.fft.ifft(..., axis)
    ↓
复值解析信号
```

公开导入没有额外 wrapper，因此三个名字最终引用同一个 Python 函数对象。

### 5.2 按执行顺序总结

1. 将 `x` 转为设备数组。
2. 拒绝复数输入，确定并检查 $N$。
3. 沿目标轴计算 $N$ 点 DFT：$X[k]=\operatorname{FFT}_N\{x[n]\}$。
4. kernel 生成 $h[k]$：DC 为 1，严格正频为 2，负频为 0，偶数 $N$ 的 Nyquist 为 1。
5. 多维输入时把 $h$ reshape 成可沿 `axis` 广播的形状。
6. 计算 $x_a[n]=\operatorname{IFFT}\{X[k]h[k]\}$。

### 5.3 `_hilbert_kernel` 的线程职责

`ElementwiseKernel` 把输出数组的每个元素交给一个逻辑迭代位置 `i`。每个位置只写 `h[i]`，无跨线程依赖、归约或显式同步。`loop_prep` 先为一次 kernel 执行计算 `odd` 与 `bend`，逐元素主体再根据 `i` 选择 0、1、2。

这里能从源码确认的是“逐元素 kernel 的索引职责”。具体 CUDA grid/block 配置由 CuPy `ElementwiseKernel` 后端决定，当前项目源码没有显式指定，不能把某个固定 block 大小写成代码事实。

## 6. 数学映射、边界、复杂度和阅读检查

### 6.1 数学公式与代码逐项映射

| 数学对象 | 代码位置（基准） | 对应关系 |
| --- | --- | --- |
| $X[k]=\operatorname{FFT}_N(x)$ | `filtering.py:882` | `cp.fft.fft(x, N, axis=axis)` |
| 偶数 $N$ 的 DC/Nyquist 权重 1 | `filtering.py:760-762` | `!odd` 且 `i==0 || i==bend` |
| 偶数 $N$ 的正频权重 2 | `filtering.py:763-764` | $0<i<N/2$ |
| 偶数 $N$ 的负频权重 0 | `filtering.py:765-766` | $i>N/2$ |
| 奇数 $N$ 的 DC 权重 1 | `filtering.py:768-770` | `odd` 且 `i==0` |
| 奇数 $N$ 的正频权重 2 | `filtering.py:771-772` | $0<i<(N+1)/2$ |
| 奇数 $N$ 的负频权重 0 | `filtering.py:773-774` | 其余索引 |
| $x_a=\operatorname{IFFT}(X\odot h)$ | `filtering.py:891` | `cp.fft.ifft(Xf * h, axis=axis)` |
| $\Re(x_a)=x$，$\Im(x_a)=\mathcal H\{x\}$ | `filtering.py:816-819, 891` | docstring 给出语义，mask 与 IFFT 落实计算 |

### 6.2 shape 与广播例子

若 `x.shape == (B, C, T)`：

- `axis=-1`：FFT 结果目标轴为 `N`，`h` 被索引成 `(1, 1, N)`；
- `axis=1`：`h` 被索引成 `(1, N, 1)`；
- 逐点乘时，单例维自动广播到所有 batch/channel；
- 输出分别为 `(B, C, N)` 或 `(B, N, T)`。

一维输入不进入 reshape 分支，`Xf.shape == h.shape == (N,)`。

### 6.3 边界处理与异常

- 复数输入：显式 `ValueError("x must be real.")`。
- `N <= 0`：显式 `ValueError("N must be positive.")`。
- `N` 非整数、`axis` 越界、输入 dtype 不受 FFT 支持：本函数没有单独处理，交给 Python/CuPy 后端报错。
- 空目标轴且 `N is None`：得到 `N=0`，随后触发正数检查。
- 多维负轴：`x.shape[axis]`、FFT 和 `ind[axis]` 使用一致的 Python 负索引语义。
- 有限记录边界：没有镜像延拓、加窗或 overlap 处理；DFT 周期延拓假设可能导致边缘振铃。
- 偶数/奇数：`bend` 公式保证只有偶数长度保留独立 Nyquist bin。

### 6.4 dtype 与数值稳定性

- `cp.iscomplexobj` 按 dtype 判断复数，不是检查数值虚部是否恰好为零；即使复数组虚部全零也会被拒绝。
- `h` 使用 `x.dtype`。常规浮点输入可精确表示 0、1、2；整数也能表示这些权重，但最终 FFT dtype 仍由 CuPy 决定。
- 布尔输入值得警惕：布尔 `h` 无法区分数值 1 与 2，而源码没有禁止 bool。这是由代码可推导的边界风险，不等同于已经执行测试确认的缺陷。
- FFT/IFFT 有有限精度舍入；理论上为零的负频可能留下机器误差量级残差。
- 当解析信号幅值接近零时，后续 `angle` 和瞬时频率差分不稳定；这是示例后处理的数学性质，不是本函数内部异常。

### 6.5 时间与空间复杂度

设目标轴长度为 $N$，其余维度乘积为 $M$：

- FFT + IFFT：通常为 $O(MN\log N)$；
- mask kernel：$O(N)$；
- 频谱逐点乘：$O(MN)$；
- 主要额外设备空间：复频谱 `Xf`、复输出、长度 $N$ 的 `h`，总体为 $O(MN)$；
- 性能通常由两次 FFT、设备内存分配以及中间乘法数组主导，mask 生成相对较小；
- 若输入最初在主机，`cp.asarray` 的 Host→Device 传输也可能显著，但已在设备上的输入通常没有这项成本。

源码没有原地复用 `Xf` 或显式 FFT plan 缓存控制；CuPy 后端是否缓存 plan 属第三方运行时行为，不应从本函数直接断言。

### 6.6 建议亲自阅读的顺序

1. 先读基准 `filtering.py:787-805`：只确认签名、输入、输出，不看实现。
2. 再读 `809-819`：把 `x + i y` 与“正频加倍、负频置零”连起来。
3. 跳到函数体 `874-885`：观察输入检查、FFT、mask 分配与 kernel 调用。
4. 回读 `_hilbert_kernel` 的 `756-784`：分别手算 $N=4$ 和 $N=5$ 的 `odd`、`bend`、`h`。
5. 最后读 `887-892`：用三维 shape 手推广播，再确认 IFFT 返回值。
6. 阅读两个 `__init__.py`：确认它们只是公开名称绑定，不是算法包装层。

### 6.7 阅读检查

1. 为什么 `hilbert` 返回复数，而不是只返回实值 Hilbert 分量？
2. $N=4$ 时 `odd`、`bend` 和四个 `h[i]` 分别是什么？
3. $N=5$ 时为什么不存在权重为 1 的 Nyquist bin？
4. `loop_prep` 与逐元素 operation 各自执行什么工作？
5. 为什么 `h` 多维广播时目标轴用 `slice(None)`，其余轴用 `newaxis`？
6. `N` 大于、小于输入长度时，哪一行实际完成补零或截断？
7. 哪些异常由本函数抛出，哪些交给 CuPy？
8. 示例中为什么 `instantaneous_frequency` 比 `signal` 少一个样本？
9. 顶层 `cusignal.hilbert` 和实现模块中的 `hilbert` 是否经过了额外包装？
10. 如何验证新增学习注释没有改动任何原始源码行？

#### 参考答案

1. 返回的是解析信号，因为基准 `filtering.py:891-892` 返回 `ifft(Xf * h)`；纯 Hilbert 分量是返回值的 `imag`，见 docstring `:816-819`。
2. $N=4$ 时 `odd=false`、`bend=2`，所以 `h=[1,2,1,0]`。
3. $N=5$ 时 `bend=3`，索引 1、2 是正频，3、4 是负频，没有独立 Nyquist bin，所以 `h=[1,2,2,0,0]`。
4. `loop_prep` 在逐元素循环前计算一次 `odd` 和 `bend`；operation 再让每个逻辑索引 `i` 独立写一个 `h[i]`。源码依据是 `_hilbert_kernel` 基准 `:756-784`。
5. `slice(None)` 保留 `h` 的长度 $N$，`newaxis` 在其他维插入长度 1；例如三维 `axis=1` 得 `(1,N,1)`，可对其余维广播。
6. 基准 `filtering.py:882` 的 `cp.fft.fft(x, N, axis=axis)` 完成补零或截断，不是 `_hilbert_kernel`。
7. 本函数显式抛出复数输入和 `N<=0` 两类 `ValueError`（`:875-880`）；非整数 `N`、非法 axis、后端不支持 dtype 等交给 CuPy。
8. `cp.diff` 对相邻相位做差，长度从 $N$ 变成 $N-1$，所以示例用 `t[1:]` 对齐（`:849-850, 860`）。
9. 没有额外包装。两个 `__init__.py` 只是 `from ... import hilbert`，三个名字绑定同一个函数对象。
10. 过滤且只过滤格式严格匹配 `# <学习注释：...>` 或 kernel 字符串内 `// <学习注释：...>` 的独立行，再与 `ZKX/cusignal-23.08.00` 对应文件逐行比较；本阶段复核结果完全一致。

## 7. 本阶段未执行的事项

- 未运行 Python、CuPy、CUDA 或硬件测试；Learning 规则明确要求本学习任务默认不运行这些测试。
- 未读取或修改 `cusignal_cpp`；阶段三尚未开始。
- 未把 `hilbert2`、其他 filtering 算子或第三方 CuPy 内部实现扩展为本次学习范围。

