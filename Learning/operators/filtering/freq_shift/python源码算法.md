# freq_shift 算子学习 — Python 源码算法

> 本文件对应 `Learning/operators/filtering/freq_shift/`，记录阶段二（cuSignal 23.08.00 Python 源码算法）。
> 阶段一原理见同目录 `数学物理原理.md`。本文所有源码摘录取自只读基准原文，行号同时给出「只读基准 / 学习副本」。
> 学习副本路径：`Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py`
> 只读基准路径：`ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py`
> 学习副本已为 freq_shift 相关代码插入 14 行 `<学习注释：...>`，原始内容与只读基准逐行一致（已核对）。

---

## 一、代码定位与接口概览

### 1. 公开导入路径

`freq_shift` 通过以下两级导出暴露给用户：

| 层级 | 文件（共享根 `ZKX_dev/` 下相对路径） | 行号（只读基准 / 学习副本） |
| --- | --- | --- |
| 模块包 | `Learning/cusignal-23.08.00/python/cusignal/filtering/__init__.py` | 21 / 21 |
| 定义文件 | `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py` | 1109 / 1151 |

`__init__.py:21` 处 `    freq_shift,` 把 `filtering.py` 中的 `freq_shift` 函数重新导出，用户可通过 `cusignal.freq_shift(...)` 或 `cusignal.filtering.freq_shift(...)` 调用。

### 2. 函数签名

```python
def freq_shift(x, freq, fs):
```

| 参数 | 类型 | 默认值 | 语义 |
| --- | --- | --- | --- |
| `x` | `array_like` | 无（必填） | 输入信号。docstring 标注 complex valued；kernel 模板 `T x` 实际接受实数或复数任意 dtype 的一维或多维数组。 |
| `freq` | `float` | 无（必填） | 频移量（Hz）。cuSignal 负号约定下 `freq>0` 为下变频（频谱左移 `freq` Hz）。 |
| `fs` | `float` | 无（必填） | 采样率（Hz），用于把 `freq` 归一化为每样本相位增量 $\omega_0=2\pi\cdot\text{freq}/f_s$。 |

| 返回值 | 类型 | shape | 语义 |
| --- | --- | --- | --- |
| `ret` | `cupy.ndarray`（`complex128`） | 与输入 `x` 相同 | 频移后的复数信号 $y[n]=x[n]\cdot e^{-j2\pi\cdot\text{freq}/f_s\cdot n}$。无论输入实/复，输出恒为 `complex128`。 |

> 注意：docstring（见第三节）列出了 `domain : string` 参数，但函数签名中**不存在**该参数，函数体也始终走时域复乘。这是文档与实现的不一致（docstring 残留），当前实际生效的是时域实现。

### 3. 依赖的模块级导入

定义文件 `filtering.py` 顶部导入 `cupy as cp`（行 14）与 `numpy as np`（行 15）。`freq_shift` 中：
- `cp`：`cp.asarray`、`cp.ElementwiseKernel`（构造 `_freq_shift_kernel`）；
- `np`：`freq_shift` **未使用** numpy（同文件其他算子使用）。

### 4. 直接依赖的本项目 kernel 与 helper

| 名称 | 只读基准行号 | 学习副本行号 | 作用 |
| --- | --- | --- | --- |
| `_freq_shift_kernel` | 1096-1106 | 1128-1146 | CuPy `ElementwiseKernel`，在 GPU 上逐点计算 $y[n]=x[n]\cdot e^{-j2\pi\cdot\text{freq}/f_s\cdot n}$。 |

`_freq_shift_kernel` 与 `freq_shift` 定义在同一个 `filtering.py` 文件中，是 `freq_shift` 唯一直接调用的本项目符号；无其他 helper（`freq_shift` 不调用 `_prod` 等）。

---

## 二、当前算子的完整相关源码

> 以下按源码原有顺序完整摘录（取自只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py` 原文，无省略、无伪代码）。导出语句见 `__init__.py:14-28`。

### 2.1 导出语句（`filtering/__init__.py:14-28`）

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

（`freq_shift` 在该导入块第 21 行；为保持上下文完整摘录整个导入块，逐行解释只覆盖 `freq_shift` 一行。）

### 2.2 `_freq_shift_kernel`（只读基准 1096-1106）

```python
_freq_shift_kernel = cp.ElementwiseKernel(
    "T x, float64 freq, float64 fs",
    "complex128 out",
    """
    thrust::complex<double> temp(0, neg2pi * freq / fs * i);
    out = x * exp(temp);
    """,
    "_freq_shift_kernel",
    options=("-std=c++11",),
    loop_prep="const double neg2pi { -1 * 2 * M_PI };",
)
```

### 2.3 `freq_shift` 函数（只读基准 1109-1125）

```python
def freq_shift(x, freq, fs):
    """
    Frequency shift signal by freq at fs sample rate

    Parameters
    ----------
    x : array_like, complex valued
        The data to be shifted.
    freq : float
        Shift by this many (Hz)
    fs : float
        Sampling rate of the signal
    domain : string
        freq or time
    """
    x = cp.asarray(x)
    return _freq_shift_kernel(x, freq, fs)
```

---

## 三、docstring 逐行翻译与解释

`freq_shift` 的 docstring 位于只读基准 1110-1123 / 学习副本 1152-1165。逐行说明：

| 只读基准 | 学习副本 | 原文 | 翻译与解释 |
| --- | --- | --- | --- |
| 1110 | 1152 | `    """` | docstring 起始三引号。 |
| 1111 | 1153 | `    Frequency shift signal by freq at fs sample rate` | 一句话摘要：在采样率 `fs` 下把信号平移 `freq` Hz。 |
| 1112 | 1154 | （空行） | 摘要与参数段分隔。 |
| 1113 | 1155 | `    Parameters` | NumPy 风格参数段标题。 |
| 1114 | 1156 | `    ----------` | 参数段下划线。 |
| 1115 | 1157 | `    x : array_like, complex valued` | 参数 `x`，类型 `array_like`，标注为复值（实际 kernel 模板 `T` 也接受实数）。 |
| 1116 | 1158 | `        The data to be shifted.` | `x` 说明：待频移的数据。 |
| 1117 | 1159 | `    freq : float` | 参数 `freq`，类型 `float`。 |
| 1118 | 1160 | `        Shift by this many (Hz)` | `freq` 说明：平移量（Hz）。 |
| 1119 | 1161 | `    fs : float` | 参数 `fs`，类型 `float`。 |
| 1120 | 1162 | `        Sampling rate of the signal` | `fs` 说明：信号采样率。 |
| 1121 | 1163 | `    domain : string` | **文档残留参数** `domain`，类型 `string`；函数签名中不存在此参数，函数体也未使用，当前只实现时域分支。 |
| 1122 | 1164 | `        freq or time` | `domain` 说明：可选 "freq" 或 "time"（未实现）。 |
| 1123 | 1165 | `    """` | docstring 结束三引号。 |

> 说明：该 docstring 没有 `Returns` 段与 `Examples` 段（与 `detrend` 不同），信息较简略；`domain` 参数是文档与实现不一致的残留。

---

## 四、源代码逐行解释

### 4.1 `_freq_shift_kernel`（只读基准 1096-1106 / 学习副本 1128-1146）

| 只读基准 | 学习副本 | 源码 | 解释 |
| --- | --- | --- | --- |
| 1096 | 1128 | `_freq_shift_kernel = cp.ElementwiseKernel(` | 把一个 CuPy `ElementwiseKernel` 赋给模块级变量 `_freq_shift_kernel`。`ElementwiseKernel` 是 CuPy 的逐元素 GPU kernel 封装：调用 `_freq_shift_kernel(x, freq, fs)` 会启动与 `x` 元素数相同的线程，每个线程执行一次给定的 C++/CUDA 主体。 |
| 1097 | 1130 | `    "T x, float64 freq, float64 fs",` | 第 1 参数（输入参数列表）字符串：声明三个输入。`T` 是 CuPy 模板类型占位符，使 `x` 可匹配实数（float32/64）或复数（complex64/128）任意 dtype；`freq`、`fs` 声明为 `float64` 标量，调用时 Python 端的 float 会被转换。 |
| 1098 | 1132 | `    "complex128 out",` | 第 2 参数（输出参数列表）：声明一个 `complex128` 输出 `out`；每个线程写出一个复数，最终得到与 `x` 形状相同的 `complex128` 数组。无论 `x` 是实数还是复数，输出恒为 `complex128`（复混频必然产生复数）。 |
| 1099 | 1136 | `    """` | 第 3 参数（kernel 主体）开始的 C++/CUDA 三引号字符串。 |
| 1100 | 1137 | `    thrust::complex<double> temp(0, neg2pi * freq / fs * i);` | C++ 语句：构造 Thrust 双精度复数 `temp`。构造函数 `thrust::complex<double>(real, imag)` 此处实部为 0、虚部为 `neg2pi * freq / fs * i`。`i` 是 CuPy 注入的线程线性索引（即样本索引 $n$，从 0 起）；`neg2pi=-2π`（见 `loop_prep`）。故 `temp = 0 + j\cdot(-2\pi\cdot\text{freq}/f_s\cdot n) = j\cdot(-2\pi\cdot\text{freq}/f_s\cdot n)`。语句末尾 `;` 是 C++ 语句终止符。 |
| 1101 | 1138 | `    out = x * exp(temp);` | C++ 语句：`exp(temp)` 对复数取指数，由欧拉公式 $e^{j\theta}=\cos\theta+j\sin\theta$ 得 $e^{-j2\pi\cdot\text{freq}/f_s\cdot n}$（负号复本振）；`x * exp(temp)` 逐点相乘，写出 $y[n]=x[n]\cdot e^{-j2\pi\cdot\text{freq}/f_s\cdot n}$。`out` 是当前线程对应的输出元素。 |
| 1102 | 1139 | `    """,` | kernel 主体字符串结束；逗号表示 `ElementwiseKernel(...)` 还有后续参数。 |
| 1103 | 1141 | `    "_freq_shift_kernel",` | 第 4 参数：kernel 名称字符串，仅用于 NVCC 编译与调试标识（如 profiler 中显示的 kernel 名）。 |
| 1104 | 1143 | `    options=("-std=c++11",),` | 关键字参数 `options`：传给 NVCC 的编译选项元组，启用 C++11（因为主体用了 `thrust::complex` 与 C++11 列表初始化 `{ }`）。 |
| 1105 | 1145 | `    loop_prep="const double neg2pi { -1 * 2 * M_PI };",` | 关键字参数 `loop_prep`：在循环开始前执行一次的 C++ 语句。`M_PI` 是 CUDA math 头中的 $\pi$ 常量；`-1 * 2 * M_PI = -2π`，用 C++11 列表初始化赋给 `const double neg2pi`。该常数是后续每个线程相位计算的公共因子，预先算一次避免线程内重复计算。 |
| 1106 | 1146 | `)` | `ElementwiseKernel(...)` 调用闭合括号。 |

**kernel 语义总结**：调用 `_freq_shift_kernel(x, freq, fs)` 对 `x` 的每个元素 $x[n]$ 执行 $y[n]=x[n]\cdot e^{-j2\pi\cdot\text{freq}/f_s\cdot n}$，其中索引 $i=n$ 从 0 起算（相位参考点 $n=0$，本振初始相位为 0）。输出为与 `x` 同形状的 `complex128` 数组。对应 `数学物理原理.md` 原理 1（频移算子）+ 原理 2（频移定理负指数分支，$Y(f)=X(f+\text{freq})$）+ 原理 3（复混频）。

### 4.2 `freq_shift` 函数体（只读基准 1109-1125 / 学习副本 1151-1169）

| 只读基准 | 学习副本 | 源码 | 解释 |
| --- | --- | --- | --- |
| 1109 | 1151 | `def freq_shift(x, freq, fs):` | 函数定义与签名。三个位置参数 `x, freq, fs`，无默认值、无 `type`/`bp`/`domain` 等模式分支。 |
| 1110 | 1152 | `    """` | docstring 开始（见第三节逐行翻译）。 |
| 1111-1122 | 1153-1164 | （docstring 内容） | 见第三节，此处不重复。docstring 无 `Returns`、无 `Examples`；`domain` 行为文档残留。 |
| 1123 | 1165 | `    """` | docstring 结束。 |
| 1124 | 1167 | `    x = cp.asarray(x)` | 把输入转为 CuPy 数组：已在 GPU 上则原样返回；CPU 数组（NumPy 等）则搬运到 GPU。后续 kernel 调用要求 `x` 为 CuPy 数组。未做 dtype 检查或转换（由 kernel 模板 `T` 自适应）。 |
| 1125 | 1169 | `    return _freq_shift_kernel(x, freq, fs)` | 调用 GPU kernel 逐点计算并直接返回结果。`ElementwiseKernel` 的 `__call__` 会启动 grid、把 `x` 作为输入、`freq`/`fs` 作为标量广播、分配 `complex128` 输出数组并返回。函数无其他处理（无 FFT、无滤波、无形状调整）。 |

---

## 五、调用链与算法总结

### 调用链

```
cusignal.freq_shift(x, freq, fs)                          # __init__.py:21 导出
  └─ filtering.freq_shift(x, freq, fs)                    # filtering.py:1109
       ├─ x = cp.asarray(x)                               # 转 CuPy 数组（GPU）
       └─ return _freq_shift_kernel(x, freq, fs)          # filtering.py:1125
              └─ [GPU] 每线程 n: y[n] = x[n]·exp(-j2π·freq/fs·n)   # 原理 1+2+3
```

### 算法总结

- **单一时域复混频实现**：$y[n]=x[n]\cdot e^{-j2\pi\cdot\text{freq}/f_s\cdot n}$，对应 `数学物理原理.md` 原理 1（频移算子）+ 原理 2（频移定理负指数分支）+ 原理 3（复混频/本振）。
- **负号约定**：`freq>0` → 频谱左移（下变频），$Y(f)=X(f+\text{freq})$。
- **GPU 并行**：`ElementwiseKernel` 把逐点复乘映射为每元素一个线程，无 Python 端循环；`loop_prep` 预计算 `-2π` 常数。
- **无 FFT、无滤波、无 reshape**：纯逐点运算，输入输出形状一致；输出强制 `complex128`。
- 与 SciPy 关系：`scipy.signal` 无直接对应的 `freq_shift` 函数；等价的手动实现是 `x * np.exp(-2j*np.pi*freq/fs*np.arange(N))`，cuSignal 把它 GPU 化并封装为算子。

---

## 六、数学映射、边界、复杂度与阅读检查

### 1. 数学公式与代码逐项映射

| 数学（原理） | 代码落实 | 只读基准行 / 学习副本行 |
| --- | --- | --- |
| 归一化角频率 $\omega_0=2\pi f_0/f_s$（原理 4） | `freq / fs` 在 kernel 内参与相位计算 | 1100 / 1137 |
| 负号本振 $e^{-j2\pi\cdot\text{freq}/f_s\cdot n}$（原理 2 负指数） | `temp = thrust::complex<double>(0, neg2pi*freq/fs*i)` + `exp(temp)` | 1100-1101 / 1137-1138 |
| 频移算子 $y[n]=x[n]\cdot e^{-j\omega_0 n}$（原理 1） | `out = x * exp(temp)` | 1101 / 1138 |
| 常数 $-2\pi$ 预计算（工程实现） | `loop_prep="const double neg2pi { -1 * 2 * M_PI };"` | 1105 / 1145 |
| $Y(f)=X(f+\text{freq})$（原理 2，下变频方向） | 整体 kernel 效果（负号约定） | 1096-1106 / 1128-1146 |
| 输入数组搬运到 GPU | `x = cp.asarray(x)` | 1124 / 1167 |

### 2. 底层机制

- **`cp.ElementwiseKernel`**：CuPy 的 GPU 逐元素 kernel 封装。输入参数用类型声明串（`T x, float64 freq, float64 fs`），`T` 为模板占位符；输出声明为 `complex128 out`；主体为 C++/CUDA 字符串；`loop_prep` 在循环前执行一次；`options` 传 NVCC 编译选项。
- **`thrust::complex<double>`**：Thrust 库的双精度复数类型，提供 `exp`、乘法等复数运算，由 CUDA math 支持。
- **`cp.asarray`**：CuPy 数组转换，确保输入在 GPU 上。
- 无 FFT、无插值、无卷积、无线性代数；仅逐点复乘。

### 3. 边界处理、异常与数值稳定性

- **无显式参数校验**：函数体不检查 `fs==0`、`freq` 是否有限、`x` 是否为空。`fs==0` 会在 kernel 内触发除零（`freq/fs` → inf/nan）；`x` 为空数组时 kernel 启动 0 个线程，返回空 `complex128` 数组。
- **dtype 自适应**：`x` 经 `cp.asarray` 后保持原 dtype，kernel 模板 `T` 适配；输出恒为 `complex128`。实数输入经复乘后自然变为复数（虚部可能非零）。
- **数值稳定性**：`exp(jθ)` 由 `cos θ + j sin θ` 计算，模长恒为 1，不引入幅度放大或衰减（原理 5）；相位 $\theta=-2\pi\cdot\text{freq}/f_s\cdot n$ 随 $n$ 线性增长，大 $n$ 下 $\theta$ 数值变大但 `cos/sin` 周期函数仍稳定（CUDA math 内部做参数归约）。
- **混叠/卷绕（原理 4）**：`freq` 按 $f_s$ 取模生效——`freq=fs` 时 $e^{-j2\pi n}=1$（恒等）；`freq=fs/2` 时 $e^{-j\pi n}=(-1)^n$（奈奎斯特翻转）；`|freq|>fs/2` 会卷绕回 $[-f_s/2,f_s/2)$。函数不检查是否越界，调用方需自行保证信号带限。
- **多维数组**：kernel 按线性索引 `i` 处理，`i` 是展平后的元素序号（row-major/C 序）；对多维 `x`，等价于沿展平顺序按 $n=0,1,2,\dots$ 施加相位。这与“沿时间轴频移”的语义一致仅当 `x` 是一维时序；多维输入时需调用方确保展平顺序即时间顺序（通常 `x` 为一维）。

### 4. 时间/空间复杂度与性能瓶颈

- 设 `x` 元素数为 $N$。
- **时间**：$O(N)$，每个元素一次复数乘法 + 一次 `exp`（内部 cos/sin）；`loop_prep` 为 $O(1)$。无 $O(N\log N)$ 的 FFT。
- **空间**：输出 `complex128` 数组占 $16N$ 字节（每复数 16 字节）；无额外大缓冲。
- **瓶颈**：`exp`（cos/sin）是每元素主要开销；对超大 $N$ 受 GPU 显存与带宽限制。相比 FFT 循环频移（$O(N\log N)$），时域复乘在单次频移下更轻量；但若需同时搬移多个频点，FFT 一次变换后批量搬移可能更优（cuSignal 未采用）。

### 5. 建议阅读顺序与观察要点

1. **先读导出**：`__init__.py:21`，确认 `freq_shift` 的公开入口。
2. **读签名与 docstring**（`1109-1123`）：理解三个参数；注意 `domain` 是文档残留、未实现。
3. **读函数体**（`1124-1125`）：仅 `cp.asarray` + kernel 调用，极简包装。
4. **读 `_freq_shift_kernel`**（`1096-1106`）：核心。重点理解：
   - `T x` 模板使 `x` 可实可复；
   - `complex128 out` 强制复数输出；
   - `temp = (0, neg2pi*freq/fs*i)` 如何对应 $j\cdot(-2\pi\cdot\text{freq}/f_s\cdot n)$；
   - `exp(temp)` 如何对应负号本振 $e^{-j2\pi\cdot\text{freq}/f_s\cdot n}$；
   - `loop_prep` 预计算 `neg2pi=-2π` 的作用；
   - `i` 为线程线性索引即样本索引 $n$，相位参考点 $n=0$。
5. **对照原理**：把 kernel 主体与 `数学物理原理.md` 原理 1、2（负指数）、3、4、5 逐一对应。
6. **观察问题**：为什么输出恒为 `complex128`？`freq>0` 为何是下变频？`freq=fs` 与 `freq=fs/2` 各有什么效果？`domain` 参数为何是文档残留？

---

## 七、阶段二自检

1. `_freq_shift_kernel` 的输入声明 `T x, float64 freq, float64 fs` 中 `T` 的作用是什么？为什么 `x` 可以是实数也可以是复数？
2. `thrust::complex<double> temp(0, neg2pi * freq / fs * i)` 构造的复数 `temp` 等于什么？为什么它对应 $j\cdot(-2\pi\cdot\text{freq}/f_s\cdot n)$？
3. `exp(temp)` 等于什么？为什么 `out = x * exp(temp)` 实现的是**下变频**（`freq>0` 时频谱左移）？结合原理 2 的正负号约定说明。
4. `loop_prep="const double neg2pi { -1 * 2 * M_PI };"` 中 `neg2pi` 等于多少？为什么要在循环前算一次而不是每个线程算？
5. `options=("-std=c++11",)` 的作用是什么？如果去掉会怎样？
6. 为什么输出 dtype 恒为 `complex128`？若输入是 `float64` 实数组，输出的虚部一定为 0 吗？
7. docstring 中的 `domain : string` 参数在签名中不存在，这说明了什么？当前实际生效的是哪种实现？
8. 若 `freq = fs`，输出是什么？若 `freq = fs/2` 呢？用原理 4 解释。
9. `i`（线程线性索引）对多维输入 `x` 意味着什么？这与“沿时间轴频移”的语义在什么条件下一致？
10. 该实现与 `x * np.exp(-2j*np.pi*freq/fs*np.arange(N))` 在数学上是否等价？为什么 cuSignal 用 GPU kernel 而不是 CuPy 的向量表达式？
