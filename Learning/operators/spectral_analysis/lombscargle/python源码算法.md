# lombscargle 算子 Python 源码算法

> 本文件对应阶段二:cuSignal 23.08.00 Python 源码算法。
> 算子编号:官方 53 项中的第 71 项,模块 `spectral_analysis`,名称 `lombscargle`。
> 本阶段深入到代码语法和执行逻辑,逐行解释当前算子的完整实现。

---

## 一、代码定位与接口概览

### 1.1 公开导入路径

**模块导出入口**: `cusignal.spectral_analysis.__init__.py:18`

```python
from cusignal.spectral_analysis.spectral import (
    ...
    lombscargle,
    ...
)
```

**用户调用方式**:
```python
import cusignal
pgram = cusignal.lombscargle(x, y, freqs)
```

或:
```python
from cusignal import lombscargle
pgram = lombscargle(x, y, freqs)
```

### 1.2 定义文件与符号名

**主函数定义文件**: `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py`
- 函数名: `lombscargle`
- 起始行号: 第 24 行(学习副本与只读基准一致)

**CUDA kernel 调用文件**: `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/_spectral_cuda.py`
- 包装类: `_cupy_lombscargle_wrapper`(第 21-52 行)
- 调度函数: `_lombscargle`(第 77-94 行)

**CUDA kernel 源码**: `/spectral_analysis/_spectral.fatbin`(预编译的二进制文件)
- Kernel 名称: `_cupy_lombscargle_float64`(假设输入为 float64)

**辅助函数**:
- `_get_tpb_bpg()`: 计算线程块配置(`helper_tools.py:55-61`)
- `_get_function()`: 加载 CUDA kernel(`helper_tools.py:64-71`)
- `_populate_kernel_cache()`: 缓存 kernel(`_spectral_cuda.py:54-66`)
- `_get_backend_kernel()`: 获取缓存的 kernel wrapper(`_spectral_cuda.py:68-74`)

### 1.3 数据类型与精度

**强制数据类型**: float64(双精度浮点)
- 输入数组 `x`, `y`, `freqs` 自动转换为 `cp.float64`
- 输出数组 `pgram` 预分配为 `cp.float64`
- 支持的数据类型: `float32`, `float64`(见 `_spectral_cuda.py:18`)

---

## 二、当前算子的完整相关源码

### 2.1 主函数 `lombscargle` 完整源码

**文件**: `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:24-150`

```python
def lombscargle(
    x,
    y,
    freqs,
    precenter=False,
    normalize=False,
):
    """
    lombscargle(x, y, freqs)
    Computes the Lomb-Scargle periodogram.
    The Lomb-Scargle periodogram was developed by Lomb [1]_ and further
    extended by Scargle [2]_ to find, and test the significance of weak
    periodic signals with uneven temporal sampling.
    When *normalize* is False (default) the computed periodogram
    is unnormalized, it takes the value ``(A**2) * N/4`` for a harmonic
    signal with amplitude A for sufficiently large N.
    When *normalize* is True the computed periodogram is normalized by
    the residuals of the data around a constant reference model (at zero).
    Input arrays should be one-dimensional and will be cast to float64.

    Parameters
    ----------
    x : array_like
        Sample times.
    y : array_like
        Measurement values.
    freqs : array_like
        Angular frequencies for output periodogram.
    precenter : bool, optional
        Pre-center amplitudes by subtracting the mean.
    normalize : bool, optional
        Compute normalized periodogram.

    Returns
    -------
    pgram : array_like
        Lomb-Scargle periodogram.

    Raises
    ------
    ValueError
        If the input arrays `x` and `y` do not have the same shape.

    Notes
    -----
    This subroutine calculates the periodogram using a slightly
    modified algorithm due to Townsend [3]_ which allows the
    periodogram to be calculated using only a single pass through
    the input arrays for each frequency.
    The algorithm running time scales roughly as O(x * freqs) or O(N^2)
    for a large number of samples and frequencies.

    References
    ----------
    .. [1] N.R. Lomb "Least-squares frequency analysis of unequally spaced
           data", Astrophysics and Space Science, vol 39, pp. 447-462, 1976
    .. [2] J.D. Scargle "Studies in astronomical time series analysis. II -
           Statistical aspects of spectral analysis of unevenly spaced data",
           The Astrophysical Journal, vol 263, pp. 835-853, 1982
    .. [3] R.H.D. Townsend, "Fast calculation of the Lomb-Scargle
           periodogram using graphics processing units.", The Astrophysical
           Journal Supplement Series, vol 191, pp. 247-253, 2010

    See Also
    --------
    istft: Inverse Short Time Fourier Transform
    check_COLA: Check whether the Constant OverLap Add (COLA) constraint is met
    welch: Power spectral density by Welch's method
    spectrogram: Spectrogram by Welch's method
    csd: Cross spectral density by Welch's method

    Examples
    --------
    >>> import cusignal
    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt
    First define some input parameters for the signal:
    >>> A = 2.
    >>> w = 1.
    >>> phi = 0.5 * cp.pi
    >>> nin = 1000
    >>> nout = 100000
    >>> frac_points = 0.9 # Fraction of points to select
    Randomly select a fraction of an array with timesteps:
    >>> r = cp.random.rand(nin)
    >>> x = cp.linspace(0.01, 10*cp.pi, nin)
    >>> x = x[r >= frac_points]
    Plot a sine wave for the selected times:
    >>> y = A * cp.sin(w*x+phi)
    Define the array of frequencies for which to compute the periodogram:
    >>> f = cp.linspace(0.01, 10, nout)
    Calculate Lomb-Scargle periodogram:
    >>> pgram = cusignal.lombscargle(x, y, f, normalize=True)
    Now make a plot of the input data:
    >>> plt.subplot(2, 1, 1)
    >>> plt.plot(cp.asnumpy(x), cp.asnumpy(y), 'b+')
    Then plot the normalized periodogram:
    >>> plt.subplot(2, 1, 2)
    >>> plt.plot(cp.asnumpy(f), cp.asnumpy(pgram))
    >>> plt.show()
    """

    x = cp.asarray(x, dtype=cp.float64)
    y = cp.asarray(y, dtype=cp.float64)
    freqs = cp.asarray(freqs, dtype=cp.float64)
    pgram = cp.empty(freqs.shape[0], dtype=cp.float64)

    assert x.ndim == 1
    assert y.ndim == 1
    assert freqs.ndim == 1

    # Check input sizes
    if x.shape[0] != y.shape[0]:
        raise ValueError("Input arrays do not have the same size.")

    y_dot = cp.zeros(1, dtype=cp.float64)
    if normalize:
        cp.dot(y, y, out=y_dot)

    if precenter:
        y_in = y - y.mean()
    else:
        y_in = y

    _lombscargle(x, y_in, freqs, pgram, y_dot)

    return pgram
```

### 2.2 CUDA kernel 调度函数 `_lombscargle`

**文件**: `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/_spectral_cuda.py:77-94`

```python
def _lombscargle(x, y, freqs, pgram, y_dot):

    threadsperblock, blockspergrid = _get_tpb_bpg()

    k_type = "lombscargle"

    _populate_kernel_cache(pgram.dtype, k_type)

    kernel = _get_backend_kernel(
        pgram.dtype,
        blockspergrid,
        threadsperblock,
        k_type,
    )

    kernel(x, y, freqs, pgram, y_dot)

    _print_atts(kernel)
```

### 2.3 Kernel wrapper 类 `_cupy_lombscargle_wrapper`

**文件**: `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/_spectral_cuda.py:21-52`

```python
class _cupy_lombscargle_wrapper(object):
    def __init__(self, grid, block, kernel):
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)

        self.grid = grid
        self.block = block
        self.kernel = kernel

    def __call__(
        self,
        x,
        y,
        freqs,
        pgram,
        y_dot,
    ):

        kernel_args = (
            x.shape[0],
            freqs.shape[0],
            x,
            y,
            freqs,
            pgram,
            y_dot,
        )

        self.kernel(self.grid, self.block, kernel_args)
```

### 2.4 辅助函数摘录

**文件**: `Learning/cusignal-23.08.00/python/cusignal/utils/helper_tools.py`

```python
def _get_tpb_bpg():

    numSM = _get_numSM()
    threadsperblock = 512
    blockspergrid = numSM * 20

    return threadsperblock, blockspergrid


def _get_function(fatbin, func):

    dir = os.path.dirname(Path(__file__).parent)

    module = cp.RawModule(
        path=dir + fatbin,
    )
    return module.get_function(func)
```

**文件**: `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/_spectral_cuda.py`

```python
def _populate_kernel_cache(np_type, k_type):

    if np_type not in _SUPPORTED_TYPES:
        raise ValueError("Datatype {} not found for '{}'".format(np_type, k_type))

    if (str(np_type), k_type) in _cupy_kernel_cache:
        return

    _cupy_kernel_cache[(str(np_type), k_type)] = _get_function(
        "/spectral_analysis/_spectral.fatbin",
        "_cupy_" + k_type + "_" + str(np_type),
    )


def _get_backend_kernel(dtype, grid, block, k_type):

    kernel = _cupy_kernel_cache[(str(dtype), k_type)]
    if kernel:
        return _cupy_lombscargle_wrapper(grid, block, kernel)
    else:
        raise ValueError("Kernel {} not found in _cupy_kernel_cache".format(k_type))
```

---

## 三、docstring 逐行翻译与解释

### 3.1 标题行

**原文**:
```python
"""
lombscargle(x, y, freqs)
Computes the Lomb-Scargle periodogram.
```

**翻译**:
- 函数签名:`lombscargle(x, y, freqs)`
- 功能:计算 Lomb-Scargle 周期图

**解释**:
- Lomb-Scargle 周期图是一种用于**非均匀采样数据**的频谱分析方法
- 与标准 FFT 不同,它不要求采样点等间隔分布

### 3.2 历史背景

**原文**:
```
The Lomb-Scargle periodogram was developed by Lomb [1]_ and further
extended by Scargle [2]_ to find, and test the significance of weak
periodic signals with uneven temporal sampling.
```

**翻译**:
Lomb-Scargle 周期图由 Lomb [1] 开发,并由 Scargle [2] 进一步扩展,用于检测和检验非均匀时间采样中微弱周期信号的显著性。

**解释**:
- Lomb (1976): 提出最小二乘拟合方法
- Scargle (1982): 完善统计理论,提出显著性检验方法

### 3.3 归一化说明

**原文**:
```
When *normalize* is False (default) the computed periodogram
is unnormalized, it takes the value ``(A**2) * N/4`` for a harmonic
signal with amplitude A for sufficiently large N.
When *normalize* is True the computed periodogram is normalized by
the residuals of the data around a constant reference model (at zero).
```

**翻译**:
- 当 `normalize=False`(默认)时,计算的周期图未归一化,对于幅度为 A 的谐波信号(当 N 足够大时),功率值约为 `(A**2) * N/4`
- 当 `normalize=True` 时,周期图通过数据相对于零参考模型的残差进行归一化

**解释**:
- **未归一化功率谱**:单位与 $y^2$ 相同(如 $V^2$),反映信号能量
- **归一化功率谱**:无量纲,峰值约为 $N/4$,便于不同数据集比较

**数学对应**:
- 未归一化: $P_{unnormalized}(\omega) \approx \frac{A^2 N}{4}$
- 归一化: $P_{normalized}(\omega) = \frac{P_{unnormalized}(\omega)}{\sigma^2}$

### 3.4 数据类型说明

**原文**:
```
Input arrays should be one-dimensional and will be cast to float64.
```

**翻译**:
输入数组应为一维,将被转换为 float64 类型。

**解释**:
- 强制 float64 确保数值精度
- 不支持多维数组(与 Scipy 不同)

### 3.5 参数说明

**原文**:
```
Parameters
----------
x : array_like
    Sample times.
y : array_like
    Measurement values.
freqs : array_like
    Angular frequencies for output periodogram.
precenter : bool, optional
    Pre-center amplitudes by subtracting the mean.
normalize : bool, optional
    Compute normalized periodogram.
```

**翻译**:
- `x`: 观测时间点数组
- `y`: 观测值数组
- `freqs`: 输出周期图的角频率数组
- `precenter`: 是否预中心化(减去均值)
- `normalize`: 是否计算归一化周期图

**关键点**:
- `freqs` 为**角频率**(单位 rad/s),需注意与普通频率 $f$ 的换算: $\omega = 2\pi f$
- `precenter` 默认为 `False`,当数据均值非零时建议设为 `True`

### 3.6 返回值与异常

**原文**:
```
Returns
-------
pgram : array_like
    Lomb-Scargle periodogram.

Raises
    ------
    ValueError
        If the input arrays `x` and `y` do not have the same shape.
```

**翻译**:
- 返回值: Lomb-Scargle 周期图数组
- 异常: 当输入数组 `x` 和 `y` 形状不一致时抛出 `ValueError`

**注意**:
- 没有检查 `x` 和 `freqs` 的重叠问题,可能导致 Nyquist 混叠

### 3.7 算法说明

**原文**:
```
Notes
-----
This subroutine calculates the periodogram using a slightly
modified algorithm due to Townsend [3]_ which allows the
periodogram to be calculated using only a single pass through
the input arrays for each frequency.
The algorithm running time scales roughly as O(x * freqs) or O(N^2)
for a large number of samples and frequencies.
```

**翻译**:
- 使用 Townsend [3] 提出的修改算法,允许对每个频率只遍历一次输入数组
- 算法时间复杂度约为 $O(N_x \cdot N_f)$,其中 $N_x$ 为时间点数,$N_f$ 为频率点数

**解释**:
- **Townsend (2010) 算法**: GPU 并行实现,每个频率点独立计算
- **复杂度**: 仍是 $O(N_x \cdot N_f)$,但通过 GPU 并行化实现约 30 倍加速
- **单遍历**: 避免重复计算,提高效率

### 3.8 参考文献

**原文**:
```
References
----------
.. [1] N.R. Lomb "Least-squares frequency analysis of unequally spaced
       data", Astrophysics and Space Science, vol 39, pp. 447-462, 1976
.. [2] J.D. Scargle "Studies in astronomical time series analysis. II -
       Statistical aspects of spectral analysis of unevenly spaced data",
       The Astrophysical Journal, vol 263, pp. 835-853, 1982
.. [3] R.H.D. Townsend, "Fast calculation of the Lomb-Scargle
       periodogram using graphics processing units.", The Astrophysical
       Journal Supplement Series, vol 191, pp. 247-253, 2010
```

**解释**:
- 文献 [1]: Lomb 原始论文,提出最小二乘拟合方法
- 文献 [2]: Scargle 扩展,完善统计理论
- 文献 [3]: Townsend GPU 加速算法,cuSignal 采用的实现

### 3.9 示例代码

**原文**:
```python
>>> import cusignal
>>> import cupy as cp
>>> import matplotlib.pyplot as plt
First define some input parameters for the signal:
>>> A = 2.
>>> w = 1.
>>> phi = 0.5 * cp.pi
>>> nin = 1000
>>> nout = 100000
>>> frac_points = 0.9 # Fraction of points to select
Randomly select a fraction of an array with timesteps:
>>> r = cp.random.rand(nin)
>>> x = cp.linspace(0.01, 10*cp.pi, nin)
>>> x = x[r >= frac_points]
Plot a sine wave for the selected times:
>>> y = A * cp.sin(w*x+phi)
Define the array of frequencies for which to compute the periodogram:
>>> f = cp.linspace(0.01, 10, nout)
Calculate Lomb-Scargle periodogram:
>>> pgram = cusignal.lombscargle(x, y, f, normalize=True)
Now make a plot of the input data:
>>> plt.subplot(2, 1, 1)
>>> plt.plot(cp.asnumpy(x), cp.asnumpy(y), 'b+')
Then plot the normalized periodogram:
>>> plt.subplot(2, 1, 2)
>>> plt.plot(cp.asnumpy(f), cp.asnumpy(pgram))
>>> plt.show()
```

**解释**:
- 构造非均匀采样数据:随机选择 90% 的时间点
- 信号: $y = A \sin(\omega t + \phi)$,幅度 $A=2$,频率 $\omega=1$ rad/s
- 频率数组:从 0.01 到 10 rad/s,共 100000 个频率点
- 归一化周期图,绘制输入信号与功率谱

---

## 四、源代码逐行解释

### 4.1 函数签名与参数

**第 24-30 行**:
```python
def lombscargle(
    x,
    y,
    freqs,
    precenter=False,
    normalize=False,
):
```

**解释**:
- 函数名: `lombscargle`(遵循 Scipy 命名惯例)
- 参数:
  - `x`, `y`, `freqs`:位置参数(无默认值)
  - `precenter`, `normalize`:关键字参数(有默认值)
- 无类型注解,接受任意 `array_like` 类型

### 4.2 数据转换与预分配(第 126-129 行)

**第 126 行**:
```python
    x = cp.asarray(x, dtype=cp.float64)
```

**语法说明**:
- `cp.asarray()`:将输入转换为 CuPy 数组
- `dtype=cp.float64`:强制转换为双精度浮点

**数学对应**:
- Lomb-Scargle 公式要求高精度数值计算,float64 避免舍入误差

**第 127-129 行**:
```python
    y = cp.asarray(y, dtype=cp.float64)
    freqs = cp.asarray(freqs, dtype=cp.float64)
    pgram = cp.empty(freqs.shape[0], dtype=cp.float64)
```

**解释**:
- 转换 `y` 和 `freqs` 为 float64
- `pgram`:预分配输出数组,大小为 `freqs.shape[0]`(频率点数)

**内存管理**:
- 输入数组转换后可能复制到 GPU(如果原本在 CPU)
- `pgram` 直接在 GPU 上分配,避免后续拷贝

### 4.3 维度检查(第 131-133 行)

**第 131-133 行**:
```python
    assert x.ndim == 1
    assert y.ndim == 1
    assert freqs.ndim == 1
```

**语法说明**:
- `assert` 语句:断言条件为真,否则抛出 `AssertionError`
- `.ndim`:NumPy/CuPy 数组属性,返回维度数

**工程约束**:
- cuSignal 的 `lombscargle` 只支持 1D 数组
- 与 Scipy 不同,不支持多维输入

**改进建议**:
- 应使用更友好的错误提示(如 `if x.ndim != 1: raise ValueError(...)`)

### 4.4 输入大小检查(第 135-137 行)

**第 135-137 行**:
```python
    # Check input sizes
    if x.shape[0] != y.shape[0]:
        raise ValueError("Input arrays do not have the same size.")
```

**语法说明**:
- `x.shape[0]`:时间点数 $N$
- `y.shape[0]`:观测值数 $N$
- 两者必须相等,否则无法配对

**边界条件**:
- 未检查 `x` 和 `freqs` 的范围(如频率是否合理)
- 未检查空数组(会导致 kernel 错误)

### 4.5 归一化准备(第 139-141 行)

**第 139 行**:
```python
    y_dot = cp.zeros(1, dtype=cp.float64)
```

**语法说明**:
- `cp.zeros(1)`:创建单元素零数组
- 用途:存储 $y \cdot y$ 的点积结果

**第 140-141 行**:
```python
    if normalize:
        cp.dot(y, y, out=y_dot)
```

**语法说明**:
- `cp.dot(y, y)`:计算向量内积 $\sum_{i=1}^{N} y_i^2$
- `out=y_dot`:输出写入预分配数组,避免临时分配

**数学对应**:
- Lomb-Scargle 归一化公式:
  $$P_{normalized}(\omega) = \frac{P_{unnormalized}(\omega)}{\sigma^2}$$
  其中 $\sigma^2 = \frac{1}{N-1} \sum_{i=1}^{N} (y_i - \bar{y})^2$

- 注意:cuSignal 直接用 $\sum y_i^2$ 而非方差 $\sigma^2$,这与 Scipy 的归一化方式略有不同

### 4.6 预中心化处理(第 143-146 行)

**第 143-146 行**:
```python
    if precenter:
        y_in = y - y.mean()
    else:
        y_in = y
```

**语法说明**:
- `y.mean()`:计算样本均值 $\bar{y} = \frac{1}{N} \sum_{i=1}^{N} y_i$
- `y - y.mean()`:广播减法,去均值

**数学对应**:
- Lomb-Scargle 原始公式假设数据已中心化($\bar{y} = 0$)
- 预中心化消除直流分量,避免低频泄漏

**工程优化**:
- 仅当 `precenter=True` 时才计算均值,节省计算

**注意**:
- 这里的预中心化是**静态的**,与 Zechmeister & Kürster (2009) 的**浮动均值**方法不同

### 4.7 CUDA kernel 调用(第 148 行)

**第 148 行**:
```python
    _lombscargle(x, y_in, freqs, pgram, y_dot)
```

**语法说明**:
- `_lombscargle`:导入的 CUDA 调度函数(见第 21 行导入)
- 参数:
  - `x`:时间数组(大小 $N$)
  - `y_in`:观测值数组(可能已中心化,大小 $N$)
  - `freqs`:频率数组(大小 $N_f$)
  - `pgram`:输出功率谱数组(大小 $N_f$)
  - `y_dot`:归一化因子(大小 1,可能为零)

**执行流程**:
1. `_lombscargle` 获取线程块配置
2. 加载或从缓存获取 CUDA kernel
3. 启动 kernel 计算
4. 结果写入 `pgram`

### 4.8 返回值(第 150 行)

**第 150 行**:
```python
    return pgram
```

**解释**:
- 返回 CuPy 数组 `pgram`,大小为 $N_f$
- 用户需用 `cp.asnumpy(pgram)` 转换回 NumPy 才能在 CPU 上使用

---

## 五、调用链与算法总结

### 5.1 完整调用链

```
用户调用
  └─> cusignal.lombscargle(x, y, freqs)
       ├─> 数据转换: cp.asarray(..., dtype=float64)
       ├─> 维度检查: assert x.ndim == 1
       ├─> 大小检查: if x.shape[0] != y.shape[0]: raise ValueError
       ├─> 归一化准备: if normalize: y_dot = cp.dot(y, y)
       ├─> 预中心化: if precenter: y_in = y - y.mean()
       └─> GPU 计算: _lombscargle(x, y_in, freqs, pgram, y_dot)
            ├─> 获取配置: threadsperblock, blockspergrid = _get_tpb_bpg()
            │    └─> 查询 GPU 属性: numSM, maxThreadsPerBlock
            ├─> 缓存 kernel: _populate_kernel_cache(dtype, "lombscargle")
            │    └─> 加载 fatbin: _get_function("/spectral_analysis/_spectral.fatbin", ...)
            ├─> 创建 wrapper: kernel = _get_backend_kernel(...)
            │    └─> 实例化: _cupy_lombscargle_wrapper(grid, block, kernel)
            └─> 启动 kernel: kernel(x, y, freqs, pgram, y_dot)
                 └─> CUDA kernel 执行(GPU 并行)
                      ├─> 计算时间参数 τ(每个频率)
                      ├─> 计算余弦和正弦项
                      ├─> 计算 Lomb-Scargle 功率
                      └─> 写入 pgram 数组
```

### 5.2 算法总结

**输入**:
- 非均匀采样时间序列 $\{(t_i, y_i)\}_{i=1}^{N}$
- 角频率数组 $\{\omega_j\}_{j=1}^{N_f}$

**输出**:
- 功率谱数组 $\{P(\omega_j)\}_{j=1}^{N_f}$

**核心步骤**:
1. **数据预处理**:
   - 转换为 float64
   - 可选:减去均值(`precenter=True`)
   - 可选:计算归一化因子(`normalize=True`)

2. **GPU kernel 执行**(Townsend 2010 算法):
   - 每个频率点 $\omega_j$ 在独立线程上并行计算
   - 单遍历数据,避免重复计算

3. **后处理**:
   - 返回功率谱数组(在 GPU 上)

**时间复杂度**:
- 理论: $O(N \cdot N_f)$(朴素算法)
- 实际: 通过 GPU 并行化,加速约 30 倍(Townsend 2010)

**空间复杂度**:
- 输入: $O(N + N_f)$
- 输出: $O(N_f)$
- 临时数组:无(无额外分配)

---

## 六、数学公式与代码的逐项映射

### 6.1 Lomb-Scargle 公式对应

**数学公式**(原始定义):
$$P_{LS}(\omega) = \frac{1}{2\sigma^2} \left[ \frac{\left( \sum_{i=1}^{N} y_i \cos(\omega(t_i - \tau)) \right)^2}{\sum_{i=1}^{N} \cos^2(\omega(t_i - \tau))} + \frac{\left( \sum_{i=1}^{N} y_i \sin(\omega(t_i - \tau)) \right)^2}{\sum_{i=1}^{N} \sin^2(\omega(t_i - \tau))} \right]$$

**代码对应**:
- **时间参数 $\tau$**:在 CUDA kernel 中计算
- **余弦项**: $\sum_{i} y_i \cos(\omega(t_i - \tau))$ → kernel 内部求和
- **正弦项**: $\sum_{i} y_i \sin(\omega(t_i - \tau))$ → kernel 内部求和
- **归一化**: `normalize=True` → kernel 内部除以 $y \cdot y$

**注意**:具体实现在 `.fatbin` 文件中,无法直接查看,但逻辑遵循 Townsend (2010) 算法。

### 6.2 预中心化对应

**数学公式**:
$$y'_i = y_i - \bar{y}$$

**代码对应**:
```python
if precenter:
    y_in = y - y.mean()
```

### 6.3 归一化对应

**数学公式**:
$$P_{normalized}(\omega) = \frac{P_{unnormalized}(\omega)}{\sum_{i=1}^{N} y_i^2}$$

**代码对应**:
```python
if normalize:
    cp.dot(y, y, out=y_dot)  # 计算分母
```
然后在 kernel 中除以 `y_dot`。

**注意**:cuSignal 使用 $\sum y_i^2$ 而非方差 $\sigma^2 = \frac{1}{N-1} \sum (y_i - \bar{y})^2$,这与 Scipy 的归一化方式略有不同。

---

## 七、边界处理、异常检查与数值稳定性

### 7.1 已实现的检查

**维度检查**:
```python
assert x.ndim == 1
assert y.ndim == 1
assert freqs.ndim == 1
```
- 确保输入为 1D 数组

**大小检查**:
```python
if x.shape[0] != y.shape[0]:
    raise ValueError("Input arrays do not have the same size.")
```
- 确保 `x` 和 `y` 长度一致

### 7.2 未实现的检查

**缺失的边界条件**:
1. **空数组检查**:
   - 未检查 `x.shape[0] == 0`,会导致 kernel 错误

2. **频率范围检查**:
   - 未检查 `freqs` 是否为负数或过大(可能超出 Nyquist 频率)
   - 未检查 `freqs` 是否单调递增

3. **NaN/Inf 检查**:
   - 未检查输入是否包含 NaN 或 Inf,会导致结果异常

4. **内存溢出检查**:
   - 未检查 $N \cdot N_f$ 是否超过 GPU 内存容量

**改进建议**:
```python
if x.shape[0] == 0:
    raise ValueError("Input arrays must not be empty.")
if cp.any(freqs < 0):
    raise ValueError("Angular frequencies must be non-negative.")
if cp.any(~cp.isfinite(x)) or cp.any(~cp.isfinite(y)):
    raise ValueError("Input arrays contain NaN or Inf.")
```

### 7.3 数值稳定性

**精度选择**:
- 强制 float64 确保数值精度
- 避免单精度(float32)导致的大数据舍入误差

**潜在数值问题**:
1. **大频率值**:当 $\omega t_i$ 很大时,$\sin$ 和 $\cos$ 计算可能损失精度
2. **极小方差**:当 `normalize=True` 且 $y$ 接近零向量时,除法可能溢出
3. **非均匀采样**:极端不规则采样可能导致 $\tau$ 计算不稳定

---

## 八、时间复杂度、空间复杂度与性能瓶颈

### 8.1 时间复杂度

**理论复杂度**:
- $O(N \cdot N_f)$,其中:
  - $N$:时间点数(输入数组长度)
  - $N_f$:频率点数(输出数组长度)

**实际加速**:
- GPU 并行化:每个频率点 $\omega_j$ 在独立线程上计算
- 加速比:约 30 倍(Townsend 2010,相比 CPU 单线程)

**性能瓶颈**:
1. **内存带宽**:大量访问 `x` 和 `y` 数组(每个频率点需遍历所有时间点)
2. **线程同步**:不同频率点的计算相互独立,无需同步
3. **寄存器压力**:kernel 需存储中间结果(余弦、正弦项)

### 8.2 空间复杂度

**输入空间**:
- `x`: $O(N)$
- `y`: $O(N)$
- `freqs`: $O(N_f)$

**输出空间**:
- `pgram`: $O(N_f)$

**临时空间**:
- `y_dot`: $O(1)$(单元素数组)
- 无其他临时数组(直接在 kernel 中计算)

**总空间**: $O(N + N_f)$

### 8.3 GPU 资源使用

**线程块配置**:
```python
threadsperblock = 512
blockspergrid = numSM * 20
```
- 每个线程块 512 个线程(典型值)
- 网格大小 = SM 数量 $\times$ 20(确保充分并行)

**kernel 参数**:
- 传递数组指针和数据长度
- 无动态共享内存

---

## 九、建议用户阅读源码的顺序

### 9.1 推荐阅读顺序

1. **先读主函数**(`spectral.py:24-150`):
   - 理解输入输出接口
   - 观察数据预处理流程
   - 注意参数含义和默认值

2. **再读 CUDA 调度函数**(`_spectral_cuda.py:77-94`):
   - 理解 GPU 配置如何获取
   - 观察 kernel 如何加载和启动
   - 注意缓存机制

3. **然后读 wrapper 类**(`_spectral_cuda.py:21-52`):
   - 理解参数如何打包传递给 kernel
   - 观察数组形状如何提取

4. **最后读辅助函数**(`helper_tools.py`):
   - 理解线程块配置如何计算
   - 观察 kernel 如何从 fatbin 加载

### 9.2 每段代码应观察的问题

**主函数观察点**:
- 数据类型转换是否强制为 float64?
- 维度检查是否严格(只支持 1D)?
- 归一化因子如何计算($y \cdot y$ 而非方差)?
- 预中心化是否影响数值精度?

**CUDA 调度函数观察点**:
- 线程块配置是否合理(512 线程/块)?
- kernel 缓存如何管理(避免重复加载)?
- 是否支持 float32(查看 `_SUPPORTED_TYPES`)?

**wrapper 类观察点**:
- 参数如何打包(形状和数据指针)?
- 是否有内存拷贝(全部在 GPU 上)?

**辅助函数观察点**:
- GPU 属性如何查询(SM 数量、最大线程数)?
- fatbin 文件路径如何构造?

---

## 十、与 Scipy 实现的差异

### 10.1 功能差异

| 特性 | cuSignal 23.08.00 | Scipy 1.6+ |
|-----|------------------|-----------|
| 多维数组支持 | ❌ 仅支持 1D | ✅ 支持 |
| 数据类型 | 强制 float64 | 自动推断,支持 float32/64 |
| 权重参数 | ❌ 不支持 | ✅ `weights` 参数 |
| 浮动均值 | ❌ 不支持 | ✅ `floating_mean` 参数 |
| 归一化方式 | 除以 $\sum y^2$ | 除以方差 $\sigma^2$ |
| 性能 | GPU 加速(约 30 倍) | CPU(单线程) |

### 10.2 实现差异

**cuSignal 采用**:
- Townsend (2010) GPU 单遍历算法
- 静态预中心化(`precenter` 参数)
- 简单归一化(直接除以 $\sum y^2$)

**Scipy 采用**:
- 标准 Lomb-Scargle 算法(CPU)
- 支持广义 Lomb-Scargle(Zechmeister & Kürster 2009):
  - 权重参数 `weights`
  - 浮动均值 `floating_mean`
  - 复数幅度模式 `normalize="amplitude"`

---

## 十一、自检问题

1. **代码定位**:
   - `lombscargle` 函数定义在哪个文件的哪一行?
   - CUDA kernel 从哪个 `.fatbin` 文件加载?

2. **参数语义**:
   - `freqs` 参数的单位是什么?如何与普通频率换算?
   - `precenter` 和 `normalize` 参数的作用分别是什么?

3. **数据流**:
   - 输入数组如何从 CPU 拷贝到 GPU?
   - 输出数组 `pgram` 何时分配?大小如何确定?

4. **调用链**:
   - `_lombscargle` 函数的作用是什么?
   - `_cupy_lombscargle_wrapper` 如何封装 kernel?

5. **数学映射**:
   - Lomb-Scargle 公式中的时间参数 $\tau$ 在哪里计算?
   - 归一化公式如何实现(分母是什么)?

6. **边界条件**:
   - 函数如何处理空数组?
   - 函数如何检查频率范围合理性?

7. **性能分析**:
   - 时间复杂度是多少?如何通过 GPU 加速?
   - 性能瓶颈在哪里(内存带宽还是计算)?

8. **与 Scipy 差异**:
   - cuSignal 为何不支持权重参数?
   - 归一化方式与 Scipy 有何不同?

---

**文档版本**: V1.0  
**创建日期**: 2026-08-08  
**源码版本**: cuSignal 23.08.00  
**只读基准**: `ZKX/cusignal-23.08.00`  
**学习副本**: `Learning/cusignal-23.08.00`(未插入学习注释)