# decimate 算子学习 — Python 源码算法

> 本文件对应 `Learning/operators/filtering/decimate/`，记录阶段二（cuSignal 23.08.00 Python 源码算法）。
> 阶段一原理见同目录 `数学物理原理.md`。本文所有源码摘录取自只读基准原文，行号同时给出「只读基准 / 学习副本」。
> 学习副本路径：`Learning/cusignal-23.08.00/python/cusignal/filtering/resample.py`
> 只读基准路径：`ZKX/cusignal-23.08.00/python/cusignal/filtering/resample.py`
> 学习副本尚未为 decimate 相关代码插入 `<学习注释：...>`，原始内容与只读基准逐行一致（已核对）。

---

## 一、代码定位与接口概览

### 1. 公开导入路径

`decimate` 通过以下两级导出暴露给用户：

| 层级 | 文件（共享根 `ZKX_dev/` 下相对路径） | 行号（只读基准 / 学习副本） |
| --- | --- | --- |
| 模块包 | `Learning/cusignal-23.08.00/python/cusignal/filtering/__init__.py` | 29 / 29 |
| 定义文件 | `Learning/cusignal-23.08.00/python/cusignal/filtering/resample.py` | 82 / 82 |

`__init__.py:29` 处 `    decimate,` 把 `resample.py` 中的 `decimate` 函数重新导出，用户可通过 `cusignal.decimate(...)` 或 `cusignal.filtering.decimate(...)` 调用。

### 2. 函数签名

```python
def decimate(x, q, n=None, axis=-1, zero_phase=True, gpupath=True):
```

| 参数 | 类型 | 默认值 | 语义 |
| --- | --- | --- | --- |
| `x` | `array_like` | 无（必填） | 输入信号，N 维数组。可为 CuPy/NumPy 数组或可转数组对象。 |
| `q` | `int` | 无（必填） | 降采样因子（正整数）。输出采样率为输入采样率的 $1/q$。 |
| `n` | `int` or `array_like` | `None` | FIR 滤波器阶数（长度减 1）或滤波器系数数组。`None` 时自动设计长度 $20q$ 的滤波器。 |
| `axis` | `int` | `-1` | 沿该轴降采样。负值表示从末尾计数（`-1` 为最后一根轴）。 |
| `zero_phase` | `bool` | `True` | `True` 时应用零相位滤波（输出居中对齐，无相位延迟）；`False` 时应用线性相位滤波。 |
| `gpupath` | `bool` | `True` | `True` 时滤波器设计与计算在 GPU 上执行；`False` 时在 CPU 上执行（适用于小滤波器）。 |

| 返回值 | 类型 | shape | 语义 |
| --- | --- | --- | --- |
| `y` | `cupy.ndarray` | 输入 shape 除 `axis` 维缩减为 $\lceil N_{\text{axis}}/q \rceil$ | 降采样后的信号。`zero_phase=True` 时样本居中；`zero_phase=False` 时样本从起始对齐。 |

### 3. 依赖的模块级导入

定义文件 `resample.py` 顶部导入了：
- `cupy as cp`（行 16）
- `numpy as np`（行 17）
- `from ..filter_design.fir_filter_design import firwin`（行 19）
- `from ._upfirdn_cuda import _output_len, _UpFIRDn`（行 21）

`decimate` 中：
- `cp`：`cp.asarray`、`cp.ndarray` 类型检查
- `np`：`np.ndarray` 类型检查
- `firwin`：设计 FIR 抗混叠滤波器
- `resample_poly`、`upfirdn`：同文件内定义的函数（行 304、448）

### 4. 直接依赖的本项目函数与 helper

| 名称 | 只读基准行号 | 学习副本行号 | 作用 |
| --- | --- | --- | --- |
| `firwin` | fir_filter_design.py:141 | 同左 | FIR 滤波器窗函数法设计。被 `decimate:134` 调用。 |
| `resample_poly` | resample.py:304-445 | 同左 | 零相位 polyphase 升采样/降采样。被 `decimate:139` 调用（`zero_phase=True`）。 |
| `upfirdn` | resample.py:448-539 | 同左 | 高效升采样-滤波-降采样。被 `decimate:144` 调用（`zero_phase=False`）。 |
| `_UpFIRDn` | _upfirdn_cuda.py | 同左 | CUDA polyphase 滤波器类。被 `upfirdn:537` 使用。 |
| `_output_len` | _upfirdn_cuda.py | 同左 | 计算 upfirdn 输出长度。被 `resample_poly:432` 使用。 |

`resample_poly` 与 `upfirdn` 虽在同一文件，但内部逻辑复杂，本文只摘录理解 `decimate` 调用所必需的部分（函数签名与核心流程），不展开其完整实现。

---

## 二、当前算子的完整相关源码

> 以下按源码原有顺序完整摘录（取自只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/resample.py` 原文，无省略、无伪代码）。导出语句见 `__init__.py:29`。

### 2.1 导出语句（`filtering/__init__.py`）

```python
from cusignal.filtering.resample import decimate, resample, resample_poly, upfirdn
```

（`decimate` 在该导入块第 29 行；为保持上下文完整摘录整个导入块，逐行解释只覆盖 `decimate` 一行。）

### 2.2 `decimate` 函数（只读基准 82-147）

```python
def decimate(x, q, n=None, axis=-1, zero_phase=True, gpupath=True):
    """
    Downsample the signal after applying an anti-aliasing filter.

    Parameters
    ----------
    x : array_like
        The signal to be downsampled, as an N-dimensional array.
    q : int
        The downsampling factor.
    n : int or array_like, optional
        The order of the filter (1 less than the length for FIR) to calculate,
        or the FIR filter coefficients to employ. Defaults to calculating the
        coefficients for 20 times the downsampling factor.
    axis : int, optional
        The axis along which to decimate.
    zero_phase : bool, optional
        Prevent shifting the outputs back by the filter's
        group delay when using an FIR filter. The default value of ``True`` is
        recommended, since a phase shift is generally not desired.
    gpupath : bool, Optional
        Optional path for filter design. gpupath == False may be desirable if
        filter sizes are small.

    Returns
    -------
    y : ndarray
        The down-sampled signal.

    See Also
    --------
    resample : Resample up or down using the FFT method.
    resample_poly : Resample using polyphase filtering and an FIR filter.

    Notes
    -----
    Only FIR filter types are currently supported in cuSignal.
    """

    x = cp.asarray(x)
    if gpupath:
        pp = cp
    else:
        pp = np

    if isinstance(n, (list, pp.ndarray)):
        b = pp.asarray(n)
    else:
        if n is None:
            half_len = 10 * q  # reasonable cutoff for our sinc-like function
            n = 2 * half_len

        b = firwin(n + 1, 1.0 / q, window="hamming", gpupath=gpupath)

    sl = [slice(None)] * x.ndim

    if zero_phase:
        y = resample_poly(x, 1, q, axis=axis, window=b, gpupath=gpupath)
    else:
        # upfirdn is generally faster than lfilter by a factor equal to the
        # downsampling factor, since it only calculates the needed outputs
        n_out = x.shape[axis] // q + bool(x.shape[axis] % q)
        y = upfirdn(b, x, 1, q, axis)
        sl[axis] = slice(None, n_out, None)

    return y[tuple(sl)]
```

### 2.3 直接依赖：`firwin` 函数签名（只读基准 fir_filter_design.py:141）

```python
def firwin(
    numtaps,
    cutoff,
    width=None,
    window="hamming",
    pass_zero=True,
    scale=True,
    nyq=None,
    fs=None,
    gpupath=True,
):
    """
    FIR filter design using the window method.
    ...
    """
```

（完整定义过长，本文只摘录签名与参数语义，不展开内部实现。）

### 2.4 直接依赖：`resample_poly` 函数签名（只读基准 304）

```python
def resample_poly(x, up, down, axis=0, window=("kaiser", 5.0), gpupath=True):
    """
    Resample `x` along the given axis using polyphase filtering.
    ...
    """
```

（完整定义见 resample.py:304-445，本文只摘录签名，调用时 `decimate:139` 传入 `up=1, down=q`。）

### 2.5 直接依赖：`upfirdn` 函数签名（只读基准 448）

```python
def upfirdn(
    h,
    x,
    up=1,
    down=1,
    axis=-1,
):
    """
    Upsample, FIR filter, and downsample.
    ...
    """
```

（完整定义见 resample.py:448-539，本文只摘录签名，调用时 `decimate:144` 传入 `up=1, down=q`。）

---

## 三、docstring 逐行翻译与解释

`decimate` 的 docstring 位于只读基准 83-119。逐行说明：

| 只读基准 | 学习副本 | 原文 | 翻译与解释 |
| --- | --- | --- | --- |
| 83 | 83 | `    """` | docstring 起始三引号。 |
| 84 | 84 | `    Downsample the signal after applying an anti-aliasing filter.` | 一句话摘要：应用抗混叠滤波后降低信号采样率。 |
| 85 | 85 | （空行） | 摘要与参数段分隔。 |
| 86-87 | 86-87 | `    Parameters` / `    ----------` | NumPy 风格参数段标题与下划线。 |
| 88-89 | 88-89 | `    x : array_like` / `        The signal to be downsampled, as an N-dimensional array.` | 参数 `x`，类型 `array_like`，语义：待降采样的 N 维数组信号。 |
| 90-91 | 90-91 | `    q : int` / `        The downsampling factor.` | 参数 `q`，类型 `int`，语义：降采样因子（正整数）。 |
| 92-95 | 92-95 | `    n : int or array_like, optional` + 说明 | 参数 `n`：FIR 滤波器阶数（长度减 1）或系数数组。`None` 时默认设计长度 $20q$ 的滤波器。 |
| 96-97 | 96-97 | `    axis : int, optional` / `        The axis along which to decimate.` | 参数 `axis`：可选整数，沿该轴降采样。 |
| 98-101 | 98-101 | `    zero_phase : bool, optional` + 说明 | 参数 `zero_phase`：`True` 时避免相位延迟，推荐默认值。 |
| 102-104 | 102-104 | `    gpupath : bool, Optional` + 说明 | 参数 `gpupath`：可选路径，`False` 时在 CPU 上设计滤波器（适用于小滤波器）。 |
| 105 | 105 | （空行） | 参数段与返回段分隔。 |
| 106-107 | 106-107 | `    Returns` / `    -------` | 返回段标题与下划线。 |
| 108-109 | 108-109 | `    y : ndarray` / `        The down-sampled signal.` | 返回 `y`，类型 `ndarray`，语义：降采样后的信号。 |
| 110 | 110 | （空行） | 返回段与 See Also 段分隔。 |
| 111-112 | 111-112 | `    See Also` / `    --------` | See Also 段标题与下划线。 |
| 113-114 | 113-114 | `    resample : ...` / `    resample_poly : ...` | 参考：`resample`（FFT 方法）、`resample_poly`（polyphase 方法）。 |
| 115 | 115 | （空行） | See Also 段与 Notes 段分隔。 |
| 116-118 | 116-118 | `    Notes` / `    -----` / `    Only FIR filter types ...` | Notes 段：说明 cuSignal 只支持 FIR 滤波器（不支持 IIR）。 |
| 119 | 119 | `    """` | docstring 结束三引号。 |

---

## 四、源代码逐行解释

### 4.1 函数体逐行解释（只读基准 121-147）

| 只读基准 | 学习副本 | 源码 | 解释 |
| --- | --- | --- | --- |
| 121 | 121 | `    x = cp.asarray(x)` | 把输入转为 CuPy 数组。若已在 GPU 上则原样返回；若为 NumPy 数组则搬运到 GPU。后续全部在 GPU 计算。 |
| 122 | 122 | `    if gpupath:` | 判断是否在 GPU 上执行滤波器设计与计算。 |
| 123 | 123 | `        pp = cp` | `gpupath=True` 时，`pp` 别名指向 `cupy`，后续用 `pp.ndarray` 类型检查、`pp.asarray` 转换。 |
| 124 | 124 | `    else:` | else 分支：`gpupath=False`。 |
| 125 | 125 | `        pp = np` | `pp` 别名指向 `numpy`，滤波器设计在 CPU 上执行。 |
| 126 | 126 | （空行） | 逻辑分段。 |
| 127 | 127 | `    if isinstance(n, (list, pp.ndarray)):` | 判断 `n` 是否为列表或数组（用户提供了滤波器系数）。 |
| 128 | 128 | `        b = pp.asarray(n)` | 若是，转为 CuPy/NumPy 数组（取决于 `pp`），得到滤波器系数 `b`。 |
| 129 | 129 | `    else:` | else 分支：`n` 不是系数数组，需要设计滤波器。 |
| 130 | 130 | `        if n is None:` | 判断 `n` 是否为 `None`（用户未指定滤波器阶数）。 |
| 131 | 131 | `            half_len = 10 * q  # reasonable cutoff for our sinc-like function` | `half_len = 10 * q`，注释说明：合理的 Sinc 类函数截断长度。原行内注释保留。 |
| 132 | 132 | `            n = 2 * half_len` | `n = 2 * half_len = 20 * q`，滤波器长度为 `n + 1 = 20q + 1`。 |
| 133 | 133 | （空行） | 逻辑分段。 |
| 134 | 134 | `        b = firwin(n + 1, 1.0 / q, window="hamming", gpupath=gpupath)` | 调用 `firwin` 设计 FIR 抗混叠滤波器。参数：长度 `n+1`、归一化截止频率 `1.0/q`（对应物理频率 $f_s/(2q)$）、窗函数 Hamming、`gpupath` 控制设计路径。返回滤波器系数数组 `b`。 |
| 135 | 135 | （空行） | 逻辑分段。 |
| 136 | 136 | `    sl = [slice(None)] * x.ndim` | 构造切片列表 `sl`，初始为 `[slice(None), slice(None), ...]`（长度等于数组维数），用于后续索引输出。 |
| 137 | 137 | （空行） | 逻辑分段。 |
| 138 | 138 | `    if zero_phase:` | 判断是否使用零相位滤波。 |
| 139 | 139 | `        y = resample_poly(x, 1, q, axis=axis, window=b, gpupath=gpupath)` | 调用 `resample_poly` 实现零相位降采样。参数：升采样因子 `1`、降采样因子 `q`、轴 `axis`、滤波器系数 `b`（通过 `window` 参数传递）、`gpupath`。`resample_poly` 内部实现 polyphase 分解与零相位对齐。 |
| 140 | 140 | `    else:` | else 分支：非零相位滤波。 |
| 141 | 141 | `        # upfirdn is generally faster than lfilter by a factor equal to the` | 原注释：`upfirdn` 比 `lfilter` 快约 `q` 倍，因为只计算需要的输出点。 |
| 142 | 142 | `        # downsampling factor, since it only calculates the needed outputs` | 原注释续。 |
| 143 | 143 | `        n_out = x.shape[axis] // q + bool(x.shape[axis] % q)` | 计算输出长度：`x.shape[axis] // q` 为整数除法部分，`bool(x.shape[axis] % q)` 为余数部分（若余数非零则加 1）。等效于 `ceil(N / q)`。 |
| 144 | 144 | `        y = upfirdn(b, x, 1, q, axis)` | 调用 `upfirdn` 实现升采样-滤波-降采样。参数：滤波器 `b`、输入 `x`、升采样 `1`、降采样 `q`、轴 `axis`。内部实现 polyphase 滤波。 |
| 145 | 145 | `        sl[axis] = slice(None, n_out, None)` | 更新切片列表：将目标轴的切片设为 `slice(None, n_out, None)`（取前 `n_out` 个样本）。 |
| 146 | 146 | （空行） | 逻辑分段。 |
| 147 | 147 | `    return y[tuple(sl)]` | 应用切片返回结果。`zero_phase=True` 时 `sl` 未修改（全选）；`zero_phase=False` 时截取前 `n_out` 个样本。 |

---

## 五、调用链与算法总结

### 调用链

```
cusignal.decimate(x, q, n, axis, zero_phase, gpupath)      # __init__.py:29 导出
  └─ resample.decimate(...)                                  # resample.py:82
       ├─ x = cp.asarray(x)                                   # 转 CuPy 数组
       ├─ pp = cp if gpupath else np                          # 选择路径
       ├─ if n is array:
       │     b = pp.asarray(n)                                # 使用用户滤波器
       ├─ else:
       │     ├─ n = 20 * q (default)                          # 默认滤波器长度
       │     └─ b = firwin(n+1, 1.0/q, window="hamming")      # 设计 FIR 抗混叠滤波器
       ├─ if zero_phase:
       │     └─ y = resample_poly(x, 1, q, axis, window=b)    # 零相位 polyphase 降采样
       └─ else:
             ├─ n_out = ceil(N / q)                           # 计算输出长度
             ├─ y = upfirdn(b, x, 1, q, axis)                 # 非零相位 polyphase 降采样
             └─ y = y[:n_out]                                 # 截取前 n_out 个样本
       └─ return y
```

### 算法总结

- **零相位分支（`zero_phase=True`）**：调用 `resample_poly`，内部通过 polyphase 分解实现零相位滤波 + 降采样。输出样本居中对齐，无相位延迟。
- **非零相位分支（`zero_phase=False`）**：调用 `upfirdn`，内部通过 polyphase 分解实现线性相位滤波 + 降采样。输出样本从起始对齐，有群延迟。
- **滤波器设计**：默认使用 Hamming 窗，长度 $20q+1$，归一化截止频率 $1/q$（对应物理截止频率 $f_s/(2q)$）。
- **GPU 加速**：所有数组运算、滤波器设计与滤波计算均在 GPU 上执行（`gpupath=True` 时）。

---

## 六、数学映射、边界、复杂度与阅读检查

### 1. 数学公式与代码逐项映射

| 数学（原理） | 代码落实 | 只读基准行 / 学习副本行 |
| --- | --- | --- |
| 抗混叠滤波器截止频率 $f_c = f_s/(2q)$（原理 2） | `firwin(n+1, 1.0/q, window="hamming")` | 134 / 134 |
| Hamming 窗设计 FIR（原理 6） | `window="hamming"` 参数 | 134 / 134 |
| 降采样因子 $q$ | `q` 参数 | 82 / 82 |
| 零相位滤波（原理 4） | `resample_poly(x, 1, q, axis, window=b)` | 139 / 139 |
| Polyphase 高效实现（原理 5） | `upfirdn(b, x, 1, q, axis)` | 144 / 144 |
| 输出长度 $\lceil N/q \rceil$ | `n_out = x.shape[axis] // q + bool(x.shape[axis] % q)` | 143 / 143 |

### 2. 底层机制

- **`firwin`**：窗函数法设计 FIR 滤波器，内部调用 `get_window` 生成窗函数，计算理想 Sinc 滤波器并加窗截断。
- **`resample_poly`**：零相位 polyphase 升采样-滤波-降采样。内部：
  - 调用 `_design_resample_poly` 设计原型滤波器（若 `window` 不是系数数组）
  - 零填充滤波器以居中对齐输出
  - 调用 `upfirdn` 执行 polyphase 滤波
  - 截取居中样本返回
- **`upfirdn`**：高效升采样-滤波-降采样。内部：
  - 创建 `_UpFIRDn` 对象（CUDA polyphase 滤波器）
  - 调用 `apply_filter` 执行 GPU kernel
- **`_UpFIRDn`**：CUDA 实现的 polyphase 滤波器，利用 Vaidyanathan 图 4.3-8d 的块结构，复杂度从 $O(N \cdot q)$ 降至 $O(N / q)$。

### 3. 边界处理、异常与数值稳定性

- **`q` 非法**：未显式检查 `q <= 0`，若传入负数或零会导致后续逻辑异常（如 `half_len = 10 * q` 为负）。
- **`n` 非法**：若 `n` 为负整数，`firwin` 会抛异常。
- **dtype 非浮点**：`firwin` 内部会检查并转为浮点；若输入 `x` 为整数，`cp.asarray` 后仍为整数，`upfirdn` 内部会转为浮点（卷积计算需要）。
- **负轴**：未显式处理负轴，但 `resample_poly` 和 `upfirdn` 内部支持负轴（`axis=-1` 为默认）。
- **数组维度不一致**：`sl = [slice(None)] * x.ndim` 构造切片列表，维度不一致会导致 `IndexError`。
- **数值稳定性**：Hamming 窗设计的 FIR 滤波器通带波纹约 0.0194、阻带衰减约 53 dB，数值稳定。`upfirdn` 使用 polyphase 分解，避免直接卷积的数值积累问题。

### 4. 时间/空间复杂度与性能瓶颈

- 设输入信号沿目标轴长度为 $N$，滤波器长度为 $M = 20q + 1$（默认）。
- **零相位分支**：`resample_poly` 内部调用 `upfirdn`，时间复杂度 $O(N \cdot M / q)$（polyphase 优化）。空间复杂度 $O(N + M)$。
- **非零相位分支**：`upfirdn` 直接执行，时间复杂度 $O(N \cdot M / q)$。空间复杂度 $O(N + M)$。
- **滤波器设计**：`firwin` 设计 Hamming 窗 FIR 滤波器，时间复杂度 $O(M)$（GPU 上并行计算）。
- **性能瓶颈**：
  - 滤波器长度 $M$ 随 $q$ 线性增长，计算量随 $q$ 增加而增加（虽然 polyphase 优化后每个输出样本的计算量仍为 $O(M/q)$，但总计算量 $O(N \cdot M / q) = O(N \cdot 20)$）。
  - GPU 内存搬运：若输入 `x` 在 CPU 上，`cp.asarray` 需搬运到 GPU，可能成为瓶颈。
  - 大数组的多维处理：`resample_poly` 和 `upfirdn` 内部对多维数组的轴处理涉及转置与 reshape，可能引入额外开销。

### 5. 建议阅读顺序与观察要点

1. **先读导出**：`__init__.py:29`，确认 `decimate` 的公开入口。
2. **读签名与 docstring**（`82-119`）：理解参数语义与零相位滤波的推荐。
3. **读滤波器设计分支**（`127-134`）：理解默认滤波器长度 $20q$ 的来源与 Hamming 窗选择。
4. **读零相位分支**（`138-139`）：理解为何默认使用 `resample_poly` 而非 `upfirdn`。
5. **读非零相位分支**（`140-145`）：理解 `upfirdn` 的性能优势（只计算需要的输出）与输出长度计算。
6. **读 `resample_poly` 签名**（`304-310`）：理解 polyphase 升采样-滤波-降采样的参数语义。
7. **读 `upfirdn` 签名**（`448-455`）：理解高效升采样-滤波-降采样的参数语义。
8. **观察问题**：
   - 为何默认滤波器长度为 $20q$ 而非固定值？
   - `resample_poly` 与 `upfirdn` 有何区别？为何 `decimate` 选择 `resample_poly` 作为默认？
   - `n_out` 计算中为何使用 `bool(x.shape[axis] % q)` 而非 `int` 转换？
   - 若用户传入自定义滤波器 `n`，如何保证其截止频率与长度合理？

---

## 七、阶段二自检

1. `decimate` 默认设计的 FIR 滤波器长度为多少？为何选择这个长度？
2. `firwin` 的 `cutoff` 参数为 `1.0 / q`，对应的物理频率是多少？
3. `zero_phase=True` 与 `zero_phase=False` 在实现上有何区别？输出样本位置如何不同？
4. 为何 `gpupath=False` 时 `pp = np`？在何种场景下使用 CPU 路径？
5. `n_out = x.shape[axis] // q + bool(x.shape[axis] % q)` 如何计算输出长度？`bool` 转换有何作用？
6. `resample_poly` 与 `upfirdn` 的区别是什么？为何 `decimate` 默认使用前者？
7. 若用户传入自定义滤波器 `n`（数组），如何确保其截止频率与降采样因子 `q` 匹配？
8. 为何 `decimate` 只支持 FIR 滤波器，不支持 IIR？这与 GPU 并行化有何关系？
9. 手算 `q=3`、输入信号长度 `N=10` 时的输出长度 `n_out`，并验证 `ceil(10/3) = 4`。
10. 若 `axis` 为负值（如 `-2`），`resample_poly` 与 `upfirdn` 如何处理？是否需要转为正值？

---

## 八、参考资料

| 序号 | 名称 | 作者/机构 | 链接 | 访问日期 |
| --- | --- | --- | --- | --- |
| 1 | cuSignal 官方文档：`cusignal.decimate` | NVIDIA | https://docs.rapids.ai/api/cusignal/stable/api.html#cusignal.decimate | 2026-08-06 |
| 2 | SciPy 官方文档：`scipy.signal.decimate` | SciPy 社区 | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html | 2026-08-06 |
| 3 | Multirate Systems and Filter Banks | P. P. Vaidyanathan, Prentice Hall | https://www.pearson.com/en-us/subject-catalog/p/multirate-systems-and-filter-banks/P200000003321/9780136057189 | 2026-08-06 |
| 4 | cuSignal 源码：`cusignal/filtering/resample.py` | NVIDIA | https://github.com/rapidsai/cusignal/blob/branch-23.08/python/cusignal/filtering/resample.py | 2026-08-06 |

> 说明：资料 1、2 为 cuSignal 与 SciPy 官方文档，是 `decimate` API 的直接参考；资料 3 为 polyphase 滤波与 upfirdn 算法的理论基础（Vaidyanathan 教材）；资料 4 为源码原始路径。