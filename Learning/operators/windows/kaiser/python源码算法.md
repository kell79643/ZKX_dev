# kaiser 算子学习——Python 源码算法

## 一、代码定位与接口概览

### 1.1 当前阶段与源码边界

本阶段只学习 cuSignal 23.08.00 的 `kaiser` Python/CuPy 实现。直接相关源码限定为：

1. `kaiser` 的两级公开导出；
2. `_kaiser_kernel` 的完整定义；
3. `kaiser(M, beta, sym=True)` 的完整签名、docstring、全部示例和函数体；
4. `cp.ElementwiseKernel`、`cyl_bessel_i0`、CuPy 数组构造和切片的当前调用语义。

没有阅读或摘录其他窗函数的实现，也没有进入 CuPy 第三方库内部源码。

### 1.2 公开导入路径

```text
cusignal.kaiser
  ← cusignal/__init__.py 中从 windows 子包导出
  ← cusignal.windows.kaiser
  ← cusignal/windows/__init__.py 中从 windows.py 导出
  ← cusignal/windows/windows.py: kaiser
  ← _kaiser_kernel
```

| 层级 | 学习副本位置 | 只读基准位置 | 符号 |
| --- | --- | --- | --- |
| 顶层 API | `Learning/cusignal-23.08.00/python/cusignal/__init__.py:125` | `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:104` | `kaiser` |
| `windows` 子包 API | `Learning/cusignal-23.08.00/python/cusignal/windows/__init__.py:33` | `ZKX/cusignal-23.08.00/python/cusignal/windows/__init__.py:32` | `kaiser` |
| 设备计算定义 | `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1269` | `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1215` | `_kaiser_kernel` |
| 公开函数定义 | `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1292` | `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1229` | `kaiser` |

行号差异来自学习副本已有的合法 `<学习注释：...>` 行；基准文件没有修改。

### 1.3 函数签名

```python
def kaiser(M, beta, sym=True):
```

| 参数 | 默认值 | 接口语义 | 源码实际行为 |
| --- | --- | --- | --- |
| `M` | 必填 | 输出点数，文档称 `int` | `M < 1` 返回空数组；`M == 1` 返回一个 1；其余值参与 `%`、比较和 `size=M`，没有统一整数类型检查 |
| `beta` | 必填 | Kaiser 形状参数，控制主瓣宽度与旁瓣电平 | 传给固定 `float64` kernel 输入，并分别进入分子、分母的 `cyl_bessel_i0` |
| `sym` | `True` | `True` 为对称窗，`False` 为周期窗 | 只有 `sym=False` 且原始 `M` 为偶数时扩为 `M+1` 再截断；奇数 `M` 不扩展 |

### 1.4 返回值、dtype 与 shape

| 条件 | 返回对象 | dtype | shape |
| --- | --- | --- | --- |
| `M < 1` | `cp.array([])` | 通常为 `float64` | `(0,)` |
| `M == 1` | `cp.ones(1, "d")` | `float64` | `(1,)` |
| `M > 1` 一般分支 | `_kaiser_kernel` 输出 | 固定 `float64` | 最终为 `(M_original,)` |

一般分支输出位于 GPU 设备内存。函数不会自动执行 `cp.asnumpy`，示例只在交给 Matplotlib 绘图时显式转为 NumPy 数组。

---

## 二、当前算子的完整相关源码

以下摘录全部来自只读基准原文，未加入学习注释，也未使用省略号。完整函数边界是基准 `windows.py:1215-1348`。

### 2.1 两级公开导出中的当前算子行

这两个 `kaiser,` 分别位于各自已有的括号式 `from ... import (...)` 语句内。为避免把同一导入表中的其他算子混入当前学习范围，只摘录当前算子的原始导出项；它们不是新的实现。

```python
    kaiser,
```

- `ZKX/cusignal-23.08.00/python/cusignal/windows/__init__.py:32`
- `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:104`

### 2.2 `_kaiser_kernel` 与 `kaiser` 的完整原文

```python
_kaiser_kernel = cp.ElementwiseKernel(
    "float64 beta",
    "float64 w",
    """
    const double temp { ( i - alpha ) / alpha };
    w = cyl_bessel_i0( beta * sqrt( 1.0 - ( temp * temp ) ) ) /
        cyl_bessel_i0( beta );
    """,
    "_kaiser_kernel",
    options=("-std=c++11",),
    loop_prep="const double alpha { 0.5 * ( _ind.size() - 1 ) };",
)


def kaiser(M, beta, sym=True):
    r"""
    Return a Kaiser window.

    The Kaiser window is a taper formed by using a Bessel function.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    beta : float
        Shape parameter, determines trade-off between main-lobe width and
        side lobe level. As beta gets large, the window narrows.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The Kaiser window is defined as

    .. math::  w(n) = I_0\left( \beta \sqrt{1-\frac{4n^2}{(M-1)^2}}
               \right)/I_0(\beta)

    with

    .. math:: \quad -\frac{M-1}{2} \leq n \leq \frac{M-1}{2},

    where :math:`I_0` is the modified zeroth-order Bessel function.

    The Kaiser was named for Jim Kaiser, who discovered a simple approximation
    to the DPSS window based on Bessel functions.
    The Kaiser window is a very good approximation to the Digital Prolate
    Spheroidal Sequence, or Slepian window, which is the transform which
    maximizes the energy in the main lobe of the window relative to total
    energy.

    The Kaiser can approximate other windows by varying the beta parameter.
    (Some literature uses alpha = beta/pi.) [4]_

    ====  =======================
    beta  Window shape
    ====  =======================
    0     Rectangular
    5     Similar to a Hamming
    6     Similar to a Hann
    8.6   Similar to a Blackman
    ====  =======================

    A beta value of 14 is probably a good starting point. Note that as beta
    gets large, the window narrows, and so the number of samples needs to be
    large enough to sample the increasingly narrow spike, otherwise NaNs will
    be returned.

    Most references to the Kaiser window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] J. F. Kaiser, "Digital Filters" - Ch 7 in "Systems analysis by
           digital computer", Editors: F.F. Kuo and J.F. Kaiser, p 218-285.
           John Wiley and Sons, New York, (1966).
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
           University of Alberta Press, 1975, pp. 177-178.
    .. [3] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [4] F. J. Harris, "On the use of windows for harmonic analysis with the
           discrete Fourier transform," Proceedings of the IEEE, vol. 66,
           no. 1, pp. 51-83, Jan. 1978. :doi:`10.1109/PROC.1978.10837`.

    Examples
    --------
    Plot the window and its frequency response:

    >>> import cusignal
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = cusignal.kaiser(51, beta=14)
    >>> plt.plot(cp.asnumpy(window))
    >>> plt.title(r"Kaiser window ($\beta$=14)")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cp.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * cp.log10(cp.abs(fftshift(A / cp.abs(A).max())))
    >>> plt.plot(cp.asnumpy(freq), cp.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title(r"Frequency response of the Kaiser window ($\beta$=14)")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if M < 1:
        return cp.array([])
    if M == 1:
        return cp.ones(1, "d")
    odd = M % 2
    if not sym and not odd:
        M = M + 1

    w = _kaiser_kernel(beta, size=M)

    if not sym and not odd:
        w = w[:-1]
    return w
```

---

## 三、docstring 逐行翻译与解释

表中每一行都对应基准 docstring 的一个非空原文行，顺序与源码一致。`L` 表示学习副本行号，`B` 表示只读基准行号。

### 3.1 简介、参数与返回值

#### raw docstring 起始符

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1293` 至 `:1293`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1230` 至 `:1230`。

```python
    r"""
```

逐行对应说明：

- 第 1 行：开始 raw docstring。前缀 `r` 让反斜杠按原样保留，适合其中的 LaTeX 命令。

#### 一句话功能摘要

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1294` 至 `:1294`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1231` 至 `:1231`。

```python
    Return a Kaiser window.
```

逐行对应说明：

- 第 1 行：返回一个 Kaiser 窗；这是函数的一句话摘要。

#### Bessel 渐缩窗概述

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1296` 至 `:1296`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1233` 至 `:1233`。

```python
    The Kaiser window is a taper formed by using a Bessel function.
```

逐行对应说明：

- 第 1 行：Kaiser 窗是利用 Bessel 函数构成的渐缩窗；这里的具体函数随后说明为 $I_0$。

#### 参数章节：`M`、`beta` 与 `sym`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1298` 至 `:1309`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1235` 至 `:1246`。

```python
    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    beta : float
        Shape parameter, determines trade-off between main-lobe width and
        side lobe level. As beta gets large, the window narrows.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.
```

逐行对应说明：

- 第 1 行：参数章节标题。
- 第 2 行：NumPy/Sphinx docstring 的标题下划线。
- 第 3 行：声明 `M` 应为整数窗长。函数体本身没有统一调用整数校验 helper。
- 第 4 行：`M` 是输出点数；若小于等于零则返回空数组。句子在下一行继续。
- 第 5 行：补全上一行，和函数体 `if M < 1` 一致。
- 第 6 行：声明 `beta` 是浮点形状参数。kernel 输入进一步固定为 `float64`。
- 第 7 行：`beta` 决定主瓣宽度与旁瓣水平之间的折中。
- 第 8 行：补全参数说明；“窗变窄”指时域轮廓更集中，频域主瓣反而变宽。
- 第 9 行：`sym` 是可选布尔参数，默认值由签名给出为 `True`。
- 第 10 行：`True` 请求对称窗，适用于滤波器设计；句子下一行结束。
- 第 11 行：补全“用于滤波器设计”。对称 FIR 系数可形成线性相位。
- 第 12 行：`False` 文档上请求周期窗，用于频谱分析；奇数长度实现存在后述差异。

#### 返回值与偶数对称窗峰值

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1311` 至 `:1315`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1248` 至 `:1252`。

```python
    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).
```

逐行对应说明：

- 第 1 行：返回值章节标题。
- 第 2 行：NumPy/Sphinx 标题下划线。
- 第 3 行：返回名为 `w` 的数组；在实现中是 `cupy.ndarray`。
- 第 4 行：连续窗中心峰值归一化为 1；括号说明在下一行继续。
- 第 5 行：偶数长度对称窗的中心位于两个样本之间，因此数组不含恰好为 1 的样本。


### 3.2 数学定义与背景

#### Notes 章节与公式引导

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1317` 至 `:1319`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1254` 至 `:1256`。

```python
    Notes
    -----
    The Kaiser window is defined as
```

逐行对应说明：

- 第 1 行：注意事项章节标题。
- 第 2 行：标题下划线。
- 第 3 行：引出 Kaiser 窗的数学定义。

#### Kaiser 核心公式

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1321` 至 `:1322`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1258` 至 `:1259`。

```python
    .. math::  w(n) = I_0\left( \beta \sqrt{1-\frac{4n^2}{(M-1)^2}}
               \right)/I_0(\beta)
```

逐行对应说明：

- 第 1 行：Sphinx 数学指令的第一行，给出分子 $I_0(\beta\sqrt{1-4n^2/(M-1)^2})$。这里的 $n$ 是中心化坐标。
- 第 2 行：补全右括号并除以 $I_0(\beta)$，实现中心归一化。

#### 中心坐标范围引导词

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1324` 至 `:1324`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1261` 至 `:1261`。

```python
    with
```

逐行对应说明：

- 第 1 行：引出中心坐标 $n$ 的允许范围。

#### 中心坐标取值范围

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1326` 至 `:1326`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1263` 至 `:1263`。

```python
    .. math:: \quad -\frac{M-1}{2} \leq n \leq \frac{M-1}{2},
```

逐行对应说明：

- 第 1 行：$n$ 从左端的 $-(M-1)/2$ 变化到右端的 $(M-1)/2$。源码 kernel 的索引 `i` 会先转换成这个中心位置。

#### $I_0$ 符号定义

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1328` 至 `:1328`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1265` 至 `:1265`。

```python
    where :math:`I_0` is the modified zeroth-order Bessel function.
```

逐行对应说明：

- 第 1 行：$I_0$ 是第一类零阶修正 Bessel 函数；kernel 用 `cyl_bessel_i0` 求值。

#### Kaiser 与 DPSS 的近似关系

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1330` 至 `:1335`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1267` 至 `:1272`。

```python
    The Kaiser was named for Jim Kaiser, who discovered a simple approximation
    to the DPSS window based on Bessel functions.
    The Kaiser window is a very good approximation to the Digital Prolate
    Spheroidal Sequence, or Slepian window, which is the transform which
    maximizes the energy in the main lobe of the window relative to total
    energy.
```

逐行对应说明：

- 第 1 行：说明命名来源，并指出它是一种简单近似；句子下一行继续。
- 第 2 行：补全：它用 Bessel 函数近似 DPSS 窗。不是说两者严格相等。
- 第 3 行：再次强调 Kaiser 是 Digital Prolate... 的良好近似。
- 第 4 行：补全名称 DPSS/Slepian window，并开始描述其能量集中性质。
- 第 5 行：DPSS 优化带内/主瓣能量占总能量的比例；句子下一行结束。
- 第 6 行：补全“相对于总能量”。cuSignal `kaiser` 并没有求解 DPSS 特征值问题。

#### `beta` 可调性与另一参数化

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1337` 至 `:1338`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1274` 至 `:1275`。

```python
    The Kaiser can approximate other windows by varying the beta parameter.
    (Some literature uses alpha = beta/pi.) [4]_
```

逐行对应说明：

- 第 1 行：改变 `beta` 可得到与其他固定窗相近的轮廓和频谱折中。
- 第 2 行：提醒另一参数化 $\alpha=\beta/\pi$；`[4]_` 是 Sphinx 引用。源码 kernel 里的局部变量 `alpha` 却表示半窗长，不能混淆。

#### `beta` 与相似窗形状表

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1340` 至 `:1347`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1277` 至 `:1284`。

```python
    ====  =======================
    beta  Window shape
    ====  =======================
    0     Rectangular
    5     Similar to a Hamming
    6     Similar to a Hann
    8.6   Similar to a Blackman
    ====  =======================
```

逐行对应说明：

- 第 1 行：reStructuredText 简单表格的上边界。
- 第 2 行：表头：`beta` 与近似窗形状。
- 第 3 行：表头分隔线。
- 第 4 行：$\beta=0$ 时精确退化为矩形窗，因为 $I_0(0)/I_0(0)=1$。
- 第 5 行：$\beta=5$ 与 Hamming 窗相似，不是同一公式。
- 第 6 行：$\beta=6$ 与 Hann 窗相似。
- 第 7 行：$\beta=8.6$ 与 Blackman 窗相似。
- 第 8 行：表格结束边界。

#### 大 `beta` 的采样与 `NaN` 风险

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1349` 至 `:1352`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1286` 至 `:1289`。

```python
    A beta value of 14 is probably a good starting point. Note that as beta
    gets large, the window narrows, and so the number of samples needs to be
    large enough to sample the increasingly narrow spike, otherwise NaNs will
    be returned.
```

逐行对应说明：

- 第 1 行：文档给出 $\beta=14$ 的经验起点，并开始提醒大参数风险。它不是普适最优值。
- 第 2 行：$\beta$ 越大，时域峰越窄，必须增加采样数才能描绘它。
- 第 3 行：若点数不足或 Bessel 数值求值溢出，中间结果可能产生 `NaN`。
- 第 4 行：补全上一行的警告。函数体没有显式检查或修复 `NaN`。

#### apodization 与 tapering 用途

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1354` 至 `:1358`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1291` 至 `:1295`。

```python
    Most references to the Kaiser window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.
```

逐行对应说明：

- 第 1 行：Kaiser 窗主要见于信号处理文献。
- 第 2 行：它是多种平滑窗之一；句子继续。
- 第 3 行：它也叫 apodization 或 taper；这里开始解释 apodization。
- 第 4 行：直观上是削弱观测段开头、结尾的突变。
- 第 5 行：补全说明：通过渐缩边缘减少有限截断造成的频谱泄漏。


### 3.3 参考资料行

#### 原始参考资料列表

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1360` 至 `:1371`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1297` 至 `:1308`。

```python
    References
    ----------
    .. [1] J. F. Kaiser, "Digital Filters" - Ch 7 in "Systems analysis by
           digital computer", Editors: F.F. Kuo and J.F. Kaiser, p 218-285.
           John Wiley and Sons, New York, (1966).
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
           University of Alberta Press, 1975, pp. 177-178.
    .. [3] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [4] F. J. Harris, "On the use of windows for harmonic analysis with the
           discrete Fourier transform," Proceedings of the IEEE, vol. 66,
           no. 1, pp. 51-83, Jan. 1978. :doi:`10.1109/PROC.1978.10837`.
```

逐行对应说明：

- 第 1 行：参考资料章节标题。
- 第 2 行：标题下划线。
- 第 3 行：引用 1 第一行：Kaiser 的 “Digital Filters” 章节。
- 第 4 行：给出书名、编辑者和页码范围。
- 第 5 行：给出出版社、地点和年份。
- 第 6 行：引用 2 第一行：Kanasewich 的地球物理时间序列著作。
- 第 7 行：给出出版社、年份和页码。
- 第 8 行：引用 3：Wikipedia 的窗函数条目。
- 第 9 行：引用 3 的 URL。
- 第 10 行：引用 4 第一行：Harris 的经典窗函数论文。
- 第 11 行：给出论文题名后半与期刊卷号。
- 第 12 行：给出期号、页码、日期和 DOI；`:doi:` 是 Sphinx 角色。


### 3.4 示例逐行解释

#### 示例目标说明

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1373` 至 `:1375`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1310` 至 `:1312`。

```python
    Examples
    --------
    Plot the window and its frequency response:
```

逐行对应说明：

- 第 1 行：示例章节标题。
- 第 2 行：标题下划线。
- 第 3 行：示例目标：绘制时域窗及频率响应。

#### 示例依赖导入

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1377` 至 `:1380`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1314` 至 `:1317`。

```python
    >>> import cusignal
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt
```

逐行对应说明：

- 第 1 行：导入 cuSignal 顶层包，以使用 `cusignal.kaiser`。
- 第 2 行：将 CuPy 导入为 `cp`，用于 GPU 数组运算和转回 NumPy。
- 第 3 行：导入 GPU FFT 和频谱中心移动函数。
- 第 4 行：导入 Matplotlib 绘图接口；Matplotlib 通常接收 CPU NumPy 数据。

#### 生成并绘制时域窗

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1382` 至 `:1386`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1319` 至 `:1323`。

```python
    >>> window = cusignal.kaiser(51, beta=14)
    >>> plt.plot(cp.asnumpy(window))
    >>> plt.title(r"Kaiser window ($\beta$=14)")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")
```

逐行对应说明：

- 第 1 行：生成长度 51、$\beta=14$ 的对称 Kaiser 窗。`sym` 使用默认 `True`。
- 第 2 行：用 `cp.asnumpy` 把 GPU 窗复制到 CPU 后绘制时域幅度。
- 第 3 行：设置标题；raw 字符串保留 LaTeX 的 `\beta`。
- 第 4 行：纵轴为幅度。
- 第 5 行：横轴为样本索引。

#### 计算并绘制归一化频率响应

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1388` 至 `:1396`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1325` 至 `:1333`。

```python
    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cp.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * cp.log10(cp.abs(fftshift(A / cp.abs(A).max())))
    >>> plt.plot(cp.asnumpy(freq), cp.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title(r"Frequency response of the Kaiser window ($\beta$=14)")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")
```

逐行对应说明：

- 第 1 行：新建一张图，避免频率响应覆盖时域图。
- 第 2 行：对窗做 2048 点零填充 FFT，并按 $M/2$ 缩放；零填充只加密显示采样。
- 第 3 行：构造约从 $-0.5$ 到 $0.5$ cycles/sample 的显示频率轴。两端都包含，因此与 FFT bin 的严格坐标略有差别，但用于示意。
- 第 4 行：先按 FFT 最大幅度归一化，再 `fftshift` 移到零频居中，取幅值、对数并换算成 dB。
- 第 5 行：把 GPU 频率轴和响应复制到 CPU 后绘图。
- 第 6 行：显示范围限制为全归一化频带和 $[-120,0]$ dB。
- 第 7 行：设置频率响应图标题。
- 第 8 行：纵轴是归一化幅值的 dB 表示。
- 第 9 行：横轴单位为 cycles/sample。

#### raw docstring 结束符

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1398` 至 `:1398`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1335` 至 `:1335`。

```python
    """
```

逐行对应说明：

- 第 1 行：结束 raw docstring；随后才开始真正执行函数体。


---

## 四、源代码逐行解释

本节覆盖导出、kernel、签名和函数体的每一行非空原文。docstring 的每一行已经在第三节逐行解释，不在此重复。

### 4.1 公开导出

#### 导出到 `cusignal.windows`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/__init__.py:33` 至 `:33`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/__init__.py:32` 至 `:32`。

```python
    kaiser,
```

逐行对应说明：

- 第 1 行：这是 `from .windows import (...)` 中的一个导入项，使 `kaiser` 进入 `cusignal.windows` 子包命名空间。结尾逗号允许多行括号式导入。

#### 导出到顶层 `cusignal`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/__init__.py:125` 至 `:125`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:104` 至 `:104`。

```python
    kaiser,
```

逐行对应说明：

- 第 1 行：这是 `from cusignal.windows import (...)` 中的一个导入项，把同一函数再提升到 `cusignal.kaiser`。没有包装或复制函数。


### 4.2 `_kaiser_kernel` 每行解释

#### 定义完整 `_kaiser_kernel`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1269` 至 `:1288`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1215` 至 `:1226`。

```python
_kaiser_kernel = cp.ElementwiseKernel(
    "float64 beta",
    "float64 w",
    """
    const double temp { ( i - alpha ) / alpha };
    w = cyl_bessel_i0( beta * sqrt( 1.0 - ( temp * temp ) ) ) /
        cyl_bessel_i0( beta );
    """,
    "_kaiser_kernel",
    options=("-std=c++11",),
    loop_prep="const double alpha { 0.5 * ( _ind.size() - 1 ) };",
)
```

逐行对应说明：

- 第 1 行：模块导入时调用 CuPy 的逐元素 kernel 构造器，并把可调用对象绑定到 `_kaiser_kernel`。前导下划线表示模块内部实现。
- 第 2 行：输入参数声明字符串：一个名为 `beta` 的双精度标量/广播输入。调用时 Python 数值会按该类型转换。
- 第 3 行：输出参数声明字符串：每个逻辑元素产生一个双精度 `w`。
- 第 4 行：开始传给即时编译器的多行 C++ kernel body 字符串。
- 第 5 行：C++11 列表初始化常量 `temp`。`i` 是 CuPy 提供的线性元素索引；`alpha=(M'-1)/2`，所以 `temp=2i/(M'-1)-1`。
- 第 6 行：计算分子 $I_0(\beta\sqrt{1-temp^2})$ 并开始除法表达式。`sqrt` 是平方根，`cyl_bessel_i0` 是零阶修正 Bessel 函数。
- 第 7 行：给出分母 $I_0(\beta)$ 并以分号结束 C++ 赋值语句，完成 Kaiser 归一化公式。
- 第 8 行：结束 kernel body 字符串；逗号分隔下一个构造参数。
- 第 9 行：指定生成 kernel 的内部名字，供编译缓存和诊断使用。
- 第 10 行：传递编译选项。`("-std=c++11",)` 是单元素 tuple，尾逗号不可省略。
- 第 11 行：在元素循环前执行一次的 C++ 片段。`_ind.size()` 是当前输出总元素数 $M'$；`alpha` 是半窗长，不是文献参数 $\beta/\pi$。
- 第 12 行：结束 `cp.ElementwiseKernel` 调用。构造结果保存于 `_kaiser_kernel`。


### 4.3 签名与函数体每行解释

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1292` 至 `:1292`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1229` 至 `:1229`。

```python
def kaiser(M, beta, sym=True):
```

逐行对应说明：

- 第 1 行：定义公开 Python 函数。`M`、`beta` 必填，`sym` 默认 `True`；冒号开始函数体。

#### 非正长度提前返回

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1400` 至 `:1402`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1336` 至 `:1337`。

```python
    if M < 1:
        return cp.array([])
```

这是一个完整的 `if` 分支：第 1 行比较 `M < 1` 并以冒号开启缩进块；条件为真时，第 2 行创建空 CuPy 数组并用 `return` 立即结束函数。因为条件包含所有负数，当前实现对负 `M` 也返回空数组，而不是抛出异常。

#### 单点窗提前返回

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1404` 至 `:1406`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1338` 至 `:1339`。

```python
    if M == 1:
        return cp.ones(1, "d")
```

只有非正长度分支没有返回时才执行这里。第 1 行用值相等运算符 `==` 识别单点窗；第 2 行生成长度 1、值为 1 的 CuPy 数组。dtype 字符 `"d"` 表示双精度 `float64`。这个分支避免后续公式出现半窗长 `alpha=(M-1)/2=0` 的除零。

#### 保存原始长度奇偶性

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1408`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1340`。

```python
    odd = M % 2
```

`%` 求除以 2 的余数：偶数得到 0，奇数得到 1。该值在可能修改 `M` 之前保存原始长度的奇偶性，后面的扩展和截断使用同一标志。

#### 偶数周期窗扩展一点

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1410` 至 `:1412`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1341` 至 `:1342`。

```python
    if not sym and not odd:
        M = M + 1
```

逐行对应说明：

- 第 1 行：`not` 做布尔取反，`and` 要求两边都真。条件等价于“请求非对称/周期窗且 `M` 为偶数”。
- 第 2 行：把局部变量 `M` 扩大 1。原始奇偶状态仍保存在 `odd`，后面据此决定是否截断。

#### 调用 kernel 生成实际窗值

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1415` 至 `:1415`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1344` 至 `:1344`。

```python
    w = _kaiser_kernel(beta, size=M)
```

逐行对应说明：

- 第 1 行：启动逐元素 GPU kernel。`beta` 对应声明输入，关键字 `size=M` 决定输出元素数并触发隐式输出分配。

#### 删除偶数周期窗的扩展末点

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1418` 至 `:1420`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1346` 至 `:1347`。

```python
    if not sym and not odd:
        w = w[:-1]
```

逐行对应说明：

- 第 1 行：复用扩展时的同一条件；若此前扩展过，这里进入截断分支。
- 第 2 行：Python 切片从开头取到最后一个元素之前，删除扩展窗的末点；结果通常是 CuPy view，最终长度恢复为原始 `M`。

#### 返回最终 GPU 数组

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:1422` 至 `:1422`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1348` 至 `:1348`。

```python
    return w
```

逐行对应说明：

- 第 1 行：返回最终 GPU 数组。对普通对称窗直接返回 kernel 输出；对偶数周期窗返回截断结果。


---

## 五、调用链与算法总结

### 5.1 完整调用链

```text
用户调用 cusignal.kaiser(M, beta, sym)
  └─ 顶层导出指向 cusignal.windows.kaiser
      └─ windows.py: kaiser
          ├─ M < 1 → cp.array([])
          ├─ M == 1 → cp.ones(1, "d")
          ├─ odd = M % 2
          ├─ sym=False 且 M 偶数 → M 临时加 1
          ├─ _kaiser_kernel(beta, size=M)
          │   ├─ loop_prep: alpha = (size-1)/2
          │   └─ 每个 i:
          │       temp = (i-alpha)/alpha
          │       w[i] = I0(beta*sqrt(1-temp^2))/I0(beta)
          ├─ 若曾扩展 → w = w[:-1]
          └─ return w
```

### 5.2 按执行顺序理解

1. **Python 边界层**先处理空窗和单点窗，避免 `alpha=0` 导致除零。
2. **Python 形状层**记录奇偶性，并为偶数周期窗临时扩长。
3. **CuPy 调度层**根据 `size=M` 分配输出并启动 kernel。
4. **device 计算层**让每个逻辑元素独立求一个 Bessel 比值。
5. **Python 返回层**按需要删除扩展端点并返回设备数组。

### 5.3 对称分支

设实际 kernel 长度为 $M'$，则

$$
\alpha_c=\frac{M'-1}{2},\qquad
t_i=\frac{i-\alpha_c}{\alpha_c}=\frac{2i}{M'-1}-1.
$$

每个元素计算

$$
w[i]=\frac{I_0\!\left(\beta\sqrt{1-t_i^2}\right)}{I_0(\beta)}.
$$

当 `sym=True` 时 $M'=M$，这就是标准长度 $M$ 的对称 Kaiser 窗。

### 5.4 偶数周期分支

原始 $M$ 为偶数且 `sym=False` 时，$M'=M+1$，所以

$$
t_i=\frac{2i}{M}-1,\qquad 0\le i\le M.
$$

删除 $i=M$ 的最后一点后保留 $0\le i\le M-1$，得到一个周期内不重复的 $M$ 个样本。

### 5.5 奇数周期分支的源码差异

原始 $M$ 为奇数时 `odd=1`，`not odd=False`。即使 `sym=False`，代码也不会扩展，仍使用 $M-1$ 分母并保留左右相等端点。这与通常对任意长度都生成 $M+1$ 点后截断的周期窗约定不一致。

当前只做静态源码学习，没有运行测试，因此结论限定为：

- 可以确认分支条件和所得公式；
- 可以确认它与 docstring 的一般周期窗描述存在差异；
- 不能在未检查直接测试和版本历史前断言设计动机。

---

## 六、Python、CuPy 与 kernel 语法要点

### 6.1 `cp.ElementwiseKernel`

构造器在这里接收：

1. 输入参数声明 `"float64 beta"`；
2. 输出参数声明 `"float64 w"`；
3. operation C++ 字符串；
4. kernel 名称；
5. 编译选项；
6. 元素循环前执行的 `loop_prep`。

调用 `_kaiser_kernel(beta, size=M)` 时没有显式输出数组，因此 CuPy 根据 `size` 和输出声明分配 `float64` 数组。`i` 是当前逻辑元素的线性索引，`_ind.size()` 是总逻辑元素数。

### 6.2 C++11 初始化语法

```cpp
const double temp { expression };
```

花括号是 C++11 列表初始化。`const` 表示赋值后不可修改，`double` 对应双精度。kernel 的 `options=("-std=c++11",)` 明确启用这种语法。

### 6.3 Python 布尔表达式

```python
if not sym and not odd:
```

- `not sym`：调用者请求 `sym=False`；
- `not odd`：`odd` 为 0，即 $M$ 为偶数；
- `and` 短路求值，两者同时成立才扩展或截断。

### 6.4 切片 `w[:-1]`

省略起点表示从索引 0 开始，终点 `-1` 表示最后一个元素的位置且不包含终点。因此保留所有元素，唯独删除末点。CuPy 切片通常建立 view，而不是复制整个数组；其底层内存生命周期由 CuPy 数组引用管理。

### 6.5 dtype 字符 `"d"`

NumPy/CuPy dtype 字符 `d` 表示双精度浮点数。它让 `M==1` 分支与一般 kernel 的 `float64` 输出保持一致。

---

## 七、数学公式与代码逐项映射

| 数学量或步骤 | 源码落实 | 学习 / 基准位置 | 说明 |
| --- | --- | --- | --- |
| 实际窗长 $M'$ | `_ind.size()`、`size=M` | `1246/1225`、`1374/1344` | `M'` 可能等于原始 $M$ 或偶数周期分支中的 $M+1$ |
| 半窗长 $\alpha_c=(M'-1)/2$ | `0.5 * (_ind.size() - 1)` | `1246 / 1225` | 源码名 `alpha`，不是文献的 $\beta/\pi$ |
| 归一化坐标 $t_i=(i-\alpha_c)/\alpha_c$ | `( i - alpha ) / alpha` | `1235 / 1219` | 把索引映射到 $[-1,1]$ |
| 根号项 $\sqrt{1-t_i^2}$ | `sqrt(1.0 - (temp * temp))` | `1237 / 1220` | 控制从中心到端点的轮廓 |
| 分子 $I_0(\beta\sqrt{1-t_i^2})$ | 第一个 `cyl_bessel_i0(...)` | `1237 / 1220` | 每个输出元素不同 |
| 分母 $I_0(\beta)$ | 第二个 `cyl_bessel_i0(beta)` | `1239 / 1221` | 同一调用内各元素相同，但源码写在逐元素 operation 中 |
| $\beta=0$ 的矩形极限 | 同一比值公式 | `1237-1239 / 1220-1221` | 分子、分母均为 $I_0(0)=1$ |
| 偶数周期窗的 $M$ 分母 | 先 `M=M+1`，后 `w[:-1]` | `1371/1342`、`1379/1347` | 使 kernel 的 $M'-1$ 等于原始 $M$ |
| 空窗边界 | `if M < 1` | `1359-1361 / 1336-1337` | 不进入公式 |
| 单点窗边界 | `if M == 1` | `1363-1365 / 1338-1339` | 避免 $M'-1=0$ |

---

## 八、底层机制、边界与数值行为

### 8.1 并行职责

逻辑上每个 GPU 元素只负责一个索引 $i$：读取同一个 `beta`，使用共享预计算的 `alpha`，写一个 `w[i]`。元素之间无数据依赖、无归约、无同步需求，也没有临时全局缓冲区。

`cp.ElementwiseKernel` 隐藏了具体 CUDA grid、block 和线程索引计算；当前 Python 源码只表达逐元素逻辑，不能从这里声称固定 block 大小。

### 8.2 未使用的机制

- 不使用 FFT 生成窗；FFT 只出现在 docstring 绘图示例。
- 不使用卷积、插值或线性代数。
- 不求 DPSS eigenvector。
- 不调用 `kaiser_beta` 或 `kaiser_atten`。
- 不调用文件顶部已有的 `_len_guards`、`_extend`、`_truncate` helper。

### 8.3 参数和异常边界

| 情况 | 实际代码行为 | 风险或含义 |
| --- | --- | --- |
| `M < 1` | 返回空数组 | 包括负数，和部分现代 SciPy 版本的负数报错行为不同 |
| `M == 1` | 返回 `[1.0]` | 避免 kernel 中 `alpha=0` |
| 非整数 `M` | 没有前置统一校验 | `%`、比较可能成功，但 `size=M` 可能拒绝不合适类型；接口要求仍是 `int` |
| `beta < 0` | 没有拒绝 | 数学上 $I_0$ 为偶函数，理想结果与 $|\beta|$ 相同 |
| 大 `beta` | 分子、分母分别求 `I_0` | 两者可能溢出后形成 `inf/inf → NaN` |
| `sym=False` 且奇数 `M` | 不扩展、不截断 | 与通用周期窗公式和 docstring 存在差异 |

### 8.4 数值稳定性

核心比值直接写成

$$
\frac{I_0(a)}{I_0(\beta)}.
$$

源码没有采用 scaled Bessel 函数或对数域计算。对较大 $\beta$，即使最终比值有限，分别求值仍可能溢出。docstring 的 `NaN` 警告同时提到窄峰欠采样；从代码上还能看到没有任何 `isfinite` 检查或后处理。

### 8.5 精度和类型转换

- kernel 输入和输出声明都固定为 `float64`；
- `temp`、`alpha` 也是 C++ `double`；
- `M==1` 用 `"d"` 保持 `float64`；
- `M<1` 的空列表默认通常推断为 `float64`；
- 没有 `float32` 专用分支或模板分发。

---

## 九、复杂度与性能瓶颈

### 9.1 时间复杂度

- $M<1$ 或 $M=1$：$O(1)$；
- 一般分支：总工作量 $O(M)$；
- 每个元素包含常数次算术、一个平方根和两次 `cyl_bessel_i0` 求值。

若有 $P$ 个并行执行资源，理想 kernel 时间可粗略看作 $O(M/P)$，但实际还包括 kernel 启动、Bessel 特殊函数吞吐和内存写入开销。

### 9.2 空间复杂度

- 输出数组：$O(M)$；
- 每个元素只需常数个标量临时量，总额按线程计但不另建 $O(M)$ 辅助数组；
- 偶数周期窗先生成 $M+1$ 点，再返回前 $M$ 点 view，底层分配仍约为 $O(M)$。

### 9.3 可能瓶颈

1. 两次 `cyl_bessel_i0` 是昂贵的特殊函数计算；分母对每个元素相同，但当前 operation 表达式中逐元素重复出现，实际编译器是否提升不能仅凭 Python 源码保证。
2. 很小的 $M$ 时，GPU kernel 启动开销可能高于算术本身。
3. `float64` 的吞吐取决于具体 GPU 平台。
4. 大 $\beta$ 的溢出不只是性能问题，也会直接影响正确性。

---

## 十、阅读顺序与检查清单

### 10.1 建议阅读顺序

1. 先看基准 `windows.py:1229` 的签名，记住三个输入。
2. 看 `windows.py:1336-1342`，画出空窗、单点、普通对称、偶数周期、奇数周期五条路径。
3. 回到 `windows.py:1225`，把 `_ind.size()` 换成 $M'$，算出 `alpha`。
4. 看 `windows.py:1219`，手推 `temp=2i/(M'-1)-1`。
5. 看 `windows.py:1220-1221`，逐项映射阶段一的 Kaiser 公式。
6. 最后看 `windows.py:1344-1348`，确认设备输出、截断和返回。
7. 再看两级 `__init__.py` 导出，理解为什么同一个函数能以两条路径调用。

### 10.2 阶段二自检问题

1. `alpha` 为什么等于半窗长？为什么不是文献中的 $\alpha=\beta/\pi$？
2. `temp` 在第一个、中心和最后一个样本分别是多少？
3. 为什么分母 $I_0(\beta)$ 能归一化中心值？
4. 偶数 `M` 且 `sym=True` 时为什么数组中没有精确的 1？
5. 偶数周期窗为何需要 `M+1` 后再删一点？
6. 奇数周期窗分支与通用定义有何差异？
7. docstring 示例中的 FFT 是否参与窗的生成？
8. 哪些行在 CPU/Python 端执行，哪些表达式进入 GPU kernel？
9. 为什么这是 $O(M)$ 总工作量，却适合逐元素并行？
10. 大 $\beta$ 为什么可能产生 `NaN`？

### 10.3 自检参考答案

1. **答案：**kernel 的 `loop_prep` 在学习副本 `windows.py:1287`、基准 `windows.py:1225` 定义 `alpha=0.5*(_ind.size()-1)`，所以它是实际计算长度 $M'$ 的半窗长 $(M'-1)/2$。文献中的另一参数化是 $\alpha_{literature}=\beta/\pi$；两者只是同名，物理含义不同。
2. **答案：**`temp=(i-alpha)/alpha`（学习副本 `:1276`、基准 `:1219`）。第一个样本 $i=0$ 时 `temp=-1`；若存在精确中心样本 $i=\alpha$，则 `temp=0`；最后一个实际计算样本 $i=M'-1=2\alpha$ 时 `temp=1`。
3. **答案：**中心 `temp=0`，分子变为 $I_0(\beta\sqrt{1})=I_0(\beta)$，再除以相同分母 $I_0(\beta)$ 得 1。对应基准 `windows.py:1220-1221`。
4. **答案：**偶数 $M$ 的半窗长 $(M-1)/2$ 是半整数，设备索引 `i` 只能取整数，没有任何样本满足 `i=alpha`，因此没有 `temp=0`，离散最大值略小于连续中心值 1。
5. **答案：**原始偶数 $M$ 且 `sym=False` 时，基准 `windows.py:1341-1342` 把计算长度改为 $M+1$，使公式分母 $M'-1$ 等于原始 $M$；基准 `:1346-1347` 再删除末点，留下一个周期内的 $M$ 个不重复样本。
6. **答案：**奇数 $M$ 时 `odd=1`，`not odd` 为假，所以 `sym=False` 也不会扩展或截断，仍按 $M-1$ 分母生成端点相等的对称采样；通常的周期窗应按 $M$ 分母采样。源码只能确认这个差异，不能无历史或测试证据断言其动机。
7. **答案：**不参与。窗值在基准 `windows.py:1344` 调用 `_kaiser_kernel` 时已经生成；FFT 仅位于 docstring 示例 `:1325-1333`，用于把返回窗转换成可绘制的频率响应。
8. **答案：**Python/Host 端执行 `M` 守卫、奇偶判断、可能的扩展、kernel 调用、切片和返回（基准 `:1336-1348`）。传入即时编译设备代码的是 `_kaiser_kernel` 的 `temp` 与 Bessel 比值表达式（基准 `:1218-1222`）；`loop_prep` 也属于生成的 kernel 代码上下文。
9. **答案：**每个输出位置只依赖 `i`、共享 `beta` 和 `alpha`，彼此无数据依赖，适合并行；但总共仍要计算并写入 $M$ 个元素，所以总工作量和输出空间都是 $O(M)$。
10. **答案：**源码分别计算分子、分母的 `cyl_bessel_i0`，没有 scaled Bessel 或对数域比值。大 $\beta$ 时二者可能先溢出为 `inf`，随后 `inf/inf` 产生 `NaN`；docstring 基准 `:1286-1289` 还提醒窄峰需要足够采样点。

---

## 十一、学习副本完整性与行号复核

### 11.1 注释规则检查

本阶段只在学习副本插入独立的合法学习注释行：

- Python 层使用 `# <学习注释：...>`；
- kernel C++ 字符串内部使用 `// <学习注释：...>`；
- 没有改写、删除、移动、格式化原代码、原注释、空行或顺序；
- `ZKX/cusignal-23.08.00` 只读基准没有修改。

### 11.2 完整性复核结果

从以下学习文件中过滤全部符合固定格式的学习注释后，与对应只读基准逐行比较，差异数均为 0：

| 文件 | 学习副本原行数 | 合法学习注释数 | 过滤后行数 | 基准行数 | 差异数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `python/cusignal/windows/windows.py` | 2248 | 134（含其他既有算子注释） | 2114 | 2114 | 0 |
| `python/cusignal/windows/__init__.py` | 41 | 3（含其他既有算子注释） | 38 | 38 | 0 |
| `python/cusignal/__init__.py` | 139 | 23（含其他既有算子注释） | 116 | 116 | 0 |

### 11.3 当前算子关键行号对照

| 符号或语句 | 学习副本 | 只读基准 |
| --- | ---: | ---: |
| `cusignal.windows` 导出 `kaiser` | 33 | 32 |
| 顶层导出 `kaiser` | 125 | 104 |
| `_kaiser_kernel` 定义 | 1269 | 1215 |
| `temp` | 1276 | 1219 |
| Bessel 分子 | 1278 | 1220 |
| Bessel 分母 | 1280 | 1221 |
| `loop_prep` | 1287 | 1225 |
| `kaiser` 定义 | 1292 | 1229 |
| `if M < 1` | 1400 | 1336 |
| `if M == 1` | 1404 | 1338 |
| `odd = M % 2` | 1408 | 1340 |
| 周期扩展条件 | 1410 | 1341 |
| kernel 调用 | 1415 | 1344 |
| 周期截断条件 | 1418 | 1346 |
| `return w` | 1422 | 1348 |

