# gausspulse Python 源码算法

## 1. 公开导入路径、定义文件与符号名

| 项目 | 内容 |
|------|------|
| 公开导入路径 | `import cusignal; cusignal.gausspulse` |
| 导出入口 | `Learning/cusignal-23.08.00/python/cusignal/waveforms/__init__.py:14-20`（`from cusignal.waveforms.waveforms import (chirp, gausspulse, sawtooth, square, unit_impulse)`） |
| 顶层再导出 | `Learning/cusignal-23.08.00/python/cusignal/__init__.py`（将 `waveforms` 模块纳入 `cusignal` 命名空间） |
| 函数定义文件 | `Learning/cusignal-23.08.00/python/cusignal/waveforms/waveforms.py` |
| 函数定义行号 | 学习副本第 248 行（只读基准 `ZKX/cusignal-23.08.00` 第 221 行） |
| 符号名 | `gausspulse` |
| kernel 定义行号 | 学习副本第 185-244 行（只读基准第 165-217 行），共 4 个 `cp.ElementwiseKernel` |

> 以下引用均以学习副本行号为准；括号内标注只读基准原始行号，便于对照。

## 2. 函数签名与参数语义

```python
def gausspulse(t, fc=1000, bw=0.5, bwr=-6, tpr=-60, retquad=False, retenv=False):
```

| 参数 | 类型 | 默认值 | 语义 |
|------|------|--------|------|
| `t` | `ndarray` 或字符串 `'cutoff'` | 必填 | 时间采样点数组（秒），或字符串 `'cutoff'` 触发截止时间计算 |
| `fc` | `float` | `1000` | 中心频率（Hz），必须 $\geq 0$ |
| `bw` | `float` | `0.5` | 分数带宽（无量纲），必须 $> 0$；绝对带宽 $\Delta f = f_c \times b_w$ |
| `bwr` | `float` | `-6` | 带宽参考电平（dB），必须 $< 0$；对应线性幅度 $r = 10^{b_{wr}/20}$ |
| `tpr` | `float` | `-60` | 截止时间参考电平（dB），仅 `t='cutoff'` 时使用，必须 $< 0$ |
| `retquad` | `bool` | `False` | 是否返回正交分量 $y_Q$ |
| `retenv` | `bool` | `False` | 是否返回包络 $y_{\text{env}}$ |

**返回值**（`t` 为数组时）：

| 返回 | dtype | shape | 条件 |
|------|-------|-------|------|
| `yI` | 与输入 `t` 的 dtype 一致（经 `cp.asarray` 转换后由 kernel 模板 `T` 推导） | 与 `t` 相同 | 始终返回 |
| `yQ` | 同上 | 同上 | 仅 `retquad=True` 时返回 |
| `yenv` | 同上 | 同上 | 仅 `retenv=True` 时返回 |

**返回值**（`t='cutoff'` 时）：返回 Python `float` 标量 $t_c$。

> **dtype 注意**：`a` 的计算使用 `np.pi`、`np.log` 等 NumPy 标量运算，结果为 Python `float`（float64）。传入 kernel 后，CuPy 的模板类型 `T` 由 `t` 数组的 dtype 决定。若 `t` 为 `float32`，则 `a` 和 `fc` 会被隐式转为 `float32` 参与计算；若 `t` 为 `float64`，则全链路 float64。

## 3. 从公开 API 到最终计算的调用链

```
cusignal.gausspulse(t, fc, bw, bwr, tpr, retquad, retenv)
  → waveforms.waveforms.gausspulse(...)          # 函数定义
    → 参数校验 (fc, bw, bwr)                      # 第 309-316 行
    → ref = pow(10.0, bwr/20.0)                   # 第 322 行
    → a = -((np.pi*fc*bw)**2) / (4.0*np.log(ref)) # 第 328 行
    → if t == 'cutoff': return np.sqrt(...)       # 第 331-342 行（截止时间分支）
    → t = cp.asarray(t)                           # 第 345 行（转 GPU 数组）
    → 根据 (retquad, retenv) 选择 kernel:          # 第 348-355 行
        (F, F) → _gausspulse_kernel_F_F(t, a, fc)
        (F, T) → _gausspulse_kernel_F_T(t, a, fc)
        (T, F) → _gausspulse_kernel_T_F(t, a, fc)
        (T, T) → _gausspulse_kernel_T_T(t, a, fc)
      → cp.ElementwiseKernel 在 GPU 上逐元素执行
```

**调用链不含**：不使用 FFT、插值、卷积或线性代数库；核心计算全部在自定义 CUDA kernel 中完成。

## 4. 按源码执行顺序拆解

### 4.1 四个 kernel 定义（第 185-244 行 / 基准第 165-217 行）

在模块加载时，4 个 `cp.ElementwiseKernel` 被创建为模块级变量。命名规则：`_gausspulse_kernel_{retquad}{retenv}`，`F`=False, `T`=True。

#### `_gausspulse_kernel_F_F`（第 185-194 行 / 基准第 165-174 行）

```python
_gausspulse_kernel_F_F = cp.ElementwiseKernel(
    "T t, T a, T fc",
    "T yI",
    """
    T yenv = exp(-a * t * t);
    yI = yenv * cos( 2 * M_PI * fc * t);
    """,
    "_gausspulse_kernel",
    options=("-std=c++11",),
)
```

- **输入**：`t`（时间）、`a`（衰减系数）、`fc`（中心频率），均为模板类型 `T`
- **输出**：`yI`（同相分量）
- **kernel 逻辑**：先算包络 `yenv = exp(-a*t*t)`，再算 `yI = yenv * cos(2*PI*fc*t)`
- **用途**：默认模式，`retquad=False, retenv=False`

#### `_gausspulse_kernel_F_T`（第 198-207 行 / 基准第 176-185 行）

- 比 `F_F` 多输出 `yenv`
- `yenv` 在 kernel 内赋值后直接作为输出传出
- **用途**：`retquad=False, retenv=True`

#### `_gausspulse_kernel_T_F`（第 212-226 行 / 基准第 187-201 行）

```cpp
T yenv { exp(-a * t * t) };
T l_yI {};
T l_yQ {};
sincos(2 * M_PI * fc * t, &l_yQ, &l_yI);
yI = yenv * l_yI;
yQ = yenv * l_yQ;
```

- 使用 CUDA 内置 `sincos(angle, &sin_ptr, &cos_ptr)` **同时**计算 sin 和 cos
- `l_yQ` 存正弦值，`l_yI` 存余弦值
- C++ 语法说明：
  - `T yenv { ... }`：花括号初始化（brace initialization）
  - `T l_yI {}`：值初始化为零
  - `&l_yQ`：取地址传给 sincos 的指针参数
- **用途**：`retquad=True, retenv=False`

#### `_gausspulse_kernel_T_T`（第 230-244 行 / 基准第 203-217 行）

- `T_F` 的扩展版，额外输出 `yenv`
- **用途**：`retquad=True, retenv=True`

### 4.2 函数体（第 248-355 行 / 基准第 221-317 行）

#### 步骤 1：参数校验（第 309-316 行 / 基准第 280-287 行）

```python
if fc < 0:
    raise ValueError("Center frequency (fc=%.2f) must be >=0." % fc)
if bw <= 0:
    raise ValueError("Fractional bandwidth (bw=%.2f) must be > 0." % bw)
if bwr >= 0:
    raise ValueError("Reference level for bandwidth (bwr=%.2f) must be < 0 dB" % bwr)
```

- `fc < 0` → 中心频率不能为负
- `bw <= 0` → 分数带宽必须为正
- `bwr >= 0` → 带宽参考电平必须为负 dB（否则 `ref >= 1`，`ln(ref) >= 0`，$a$ 为非正，无意义）
- 注意：`tpr` 的校验仅在 `t='cutoff'` 分支内进行（第 336-337 行）

#### 步骤 2：计算参考电平和衰减系数（第 319-328 行 / 基准第 289-295 行）

```python
ref = pow(10.0, bwr / 20.0)
a = -((np.pi * fc * bw) ** 2) / (4.0 * np.log(ref))
```

- `ref`：将 `bwr`（dB）转为线性幅度，$r = 10^{b_{wr}/20}$
- `a`：高斯衰减系数，$a = -\dfrac{(\pi f_c b_w)^2}{4 \ln r}$
- 原代码注释 `# exp(-a t^2) <-> sqrt(pi/a) exp(-pi^2/a * f^2) = g(f)` 说明了使用的傅里叶变换对
- `np.pi` 和 `np.log` 是 NumPy 标量运算，在 CPU 上执行（标量计算无需上 GPU）

#### 步骤 3：cutoff 分支（第 331-342 行 / 基准第 297-306 行）

```python
if isinstance(t, str):
    if t == "cutoff":
        if tpr >= 0:
            raise ValueError("Reference level for time cutoff must be < 0 dB")
        tref = pow(10.0, tpr / 20.0)
        return np.sqrt(-np.log(tref) / a)
    else:
        raise ValueError("If `t` is a string, it must be 'cutoff'")
```

- `isinstance(t, str)` 检查 `t` 是否为字符串
- `t == "cutoff"` 进入截止时间计算
- `tref = 10^{tpr/20}`：时域参考幅度
- `return np.sqrt(-np.log(tref) / a)`：$t_c = \sqrt{-\ln(t_{\text{ref}}) / a}$
- 此分支返回 Python `float` 标量，不涉及 GPU

#### 步骤 4：数组模式（第 345-355 行 / 基准第 308-317 行）

```python
t = cp.asarray(t)

if not retquad and not retenv:
    return _gausspulse_kernel_F_F(t, a, fc)
if not retquad and retenv:
    return _gausspulse_kernel_F_T(t, a, fc)
if retquad and not retenv:
    return _gausspulse_kernel_T_F(t, a, fc)
if retquad and retenv:
    return _gausspulse_kernel_T_T(t, a, fc)
```

- `cp.asarray(t)`：将输入转为 CuPy 数组。如果 `t` 已是 CuPy 数组则无操作；如果是 NumPy 数组或 Python 列表则搬运到 GPU
- 四个 `if` 分支根据 `(retquad, retenv)` 的布尔组合选择对应 kernel
- 每个 kernel 调用返回 1-3 个 CuPy 数组，直接 `return`

## 5. 关键语句的 Python/CuPy 语法说明

| 语句 | 语法说明 |
|------|----------|
| `cp.ElementwiseKernel("T t, T a, T fc", "T yI", "...", name, options=...)` | CuPy 的 elementwise kernel 定义：第一参数是输入参数列表（逗号分隔，`T` 是模板类型占位符），第二参数是输出参数列表，第三参数是 C++ kernel 代码字符串，第四参数是 kernel 名称，`options` 传给 NVCC 编译选项 |
| `T yenv = exp(-a * t * t);` | C++ 声明变量 `yenv`，类型为模板 `T`（由输入 dtype 决定），调用 CUDA `exp` 函数 |
| `sincos(angle, &sin_ptr, &cos_ptr)` | CUDA math API，一次调用同时计算 sin 和 cos，比分别调用 `sin` + `cos` 更高效 |
| `pow(10.0, bwr / 20.0)` | Python 内置 `pow`，等效于 `10.0 ** (bwr / 20.0)`，返回 float |
| `np.log(ref)` | NumPy 自然对数，`ref` 是 Python float 标量 |
| `cp.asarray(t)` | CuPy 数组转换函数，等价于 NumPy 的 `np.asarray` 但结果在 GPU 内存 |
| `isinstance(t, str)` | Python 类型检查，判断 `t` 是否为字符串 |

## 6. 数学公式与代码语句的逐项映射

| 数学公式 | 代码语句 | 行号（学习副本 / 基准） |
|----------|----------|------------------------|
| $r = 10^{b_{wr}/20}$ | `ref = pow(10.0, bwr / 20.0)` | 322 / 291 |
| $a = -\dfrac{(\pi f_c b_w)^2}{4 \ln r}$ | `a = -((np.pi * fc * bw) ** 2) / (4.0 * np.log(ref))` | 328 / 295 |
| $t_{\text{ref}} = 10^{t_{pr}/20}$ | `tref = pow(10.0, tpr / 20.0)` | 338 / 303 |
| $t_c = \sqrt{-\ln(t_{\text{ref}}) / a}$ | `return np.sqrt(-np.log(tref) / a)` | 340 / 304 |
| $y_{\text{env}} = e^{-a t^2}$ | `yenv = exp(-a * t * t)` | 189, 202, 216, 234 / 169, 180, 191, 207 |
| $y_I = e^{-a t^2} \cos(2\pi f_c t)$ | `yI = yenv * cos(2 * M_PI * fc * t)` | 190, 203 / 170, 181 |
| $y_Q = e^{-a t^2} \sin(2\pi f_c t)$ | `sincos(2*M_PI*fc*t, &l_yQ, &l_yI); yQ = yenv * l_yQ` | 220, 222, 238, 240 / 195, 197, 211, 213 |

## 7. 底层机制

- **CuPy ElementwiseKernel**：cuSignal gausspulse 不使用 FFT、插值、卷积或线性代数库。全部核心计算由 4 个 `cp.ElementwiseKernel` 完成，每个 kernel 在 GPU 上对输入数组的每个元素独立执行相同的 C++ 代码。
- **kernel 编译**：CuPy 在首次调用时将 kernel 代码字符串用 NVCC 编译为 PTX，后续调用直接复用缓存。`options=("-std=c++11",)` 指定编译标准。
- **模板类型 `T`**：CuPy 根据输入数组的 dtype 自动实例化模板。例如 `t` 为 `float64` 数组时 `T = double`，为 `float32` 时 `T = float`。
- **`sincos` 优化**：`_gausspulse_kernel_T_F` 和 `_T_T` 使用 CUDA 的 `sincos` 内置函数，在需要正弦和余弦时只需一次三角函数调用，比分别调用 `sin` 和 `cos` 减少约 50% 的三角函数开销。
- **标量在 CPU 计算**：`ref` 和 `a` 是标量，用 NumPy（`np.pi`、`np.log`）在 CPU 上计算。标量计算开销极小，不值得搬运到 GPU。

## 8. 边界处理、异常检查、数值稳定性与 dtype 转换

### 边界处理与异常检查

| 检查 | 条件 | 异常 | 行号 |
|------|------|------|------|
| `fc < 0` | 中心频率为负 | `ValueError` | 309 / 280 |
| `bw <= 0` | 分数带宽非正 | `ValueError` | 311 / 282 |
| `bwr >= 0` | 带宽参考非负 dB | `ValueError` | 313 / 284 |
| `tpr >= 0`（仅 cutoff） | 截止参考非负 dB | `ValueError` | 336 / 301 |
| `t` 是字符串但非 `'cutoff'` | 无法识别 | `ValueError` | 342 / 306 |

### 数值稳定性

- 当 $a \cdot t^2$ 很大时（远离中心的尾部），`exp(-a * t * t)` 下溢到 0，这是数学上的正确行为，不会产生 NaN 或 Inf。
- 当 `bwr` 接近 0（如 -0.001 dB）时，`ref` 接近 1，`ln(ref)` 接近 0，$a$ 趋向无穷大，脉冲变成极窄的尖峰。代码不做额外保护，但参数校验 `bwr < 0` 防止了 `bwr >= 0` 导致 `a` 为负或除零。
- `fc = 0` 时合法，输出退化为纯高斯函数 `exp(-a*t^2)`（无载波振荡），因为 `cos(0) = 1`。

### dtype 转换

- `a` 通过 `np.pi`（float64）和 `np.log`（float64）计算，结果为 Python `float`（float64 精度）。
- `cp.asarray(t)` 将 `t` 转为 CuPy 数组，dtype 由原始输入决定。
- kernel 执行时，`a`（float64）和 `fc`（Python int 或 float）会根据 `t` 的 dtype 被 CuPy 隐式转换：若 `t` 为 `float32`，则 `a` 和 `fc` 被降为 `float32` 参与计算，可能导致精度损失。
- 截止时间分支 `np.sqrt(-np.log(tref) / a)` 全部在 NumPy 上执行，返回 float64 标量。

## 9. 时间复杂度、空间复杂度与性能瓶颈

| 维度 | 复杂度 |
|------|--------|
| 时间复杂度 | $O(N)$，$N$ 为 `t` 数组长度。每个元素执行常数次 exp、cos/sin 和乘法 |
| 空间复杂度 | $O(N)$，输出 1-3 个与 `t` 同 shape 的数组 |
| kernel 启动开销 | 4 个 kernel 在模块加载时预编译，调用时只需一次 kernel launch |
| 数据搬运 | `cp.asarray(t)` 可能触发 CPU→GPU 拷贝（若 `t` 为 NumPy 数组）；返回值在 GPU 上，需 `cp.asnumpy()` 取回 |

**性能瓶颈**：
1. **三角函数计算**：`exp`、`cos`/`sin`/`sincos` 是 kernel 中的主要计算开销。`sincos` 版本（T_F、T_T）比分别调用 `sin`+`cos` 更高效。
2. **数据搬运**：若 `t` 输入为 NumPy 数组，`cp.asarray` 会触发 PCIe 传输，对短数组可能成为瓶颈。
3. **kernel 选择策略**：四个独立 kernel 避免了 kernel 内部的条件分支（`if retquad`），但增加了代码体积。对 GPU 而言，消除分支有利于 warp 效率。

## 10. 建议阅读顺序与观察问题

### 推荐阅读顺序

1. **导出入口** `waveforms/__init__.py:14-20` — 确认 `gausspulse` 从何处导入
2. **函数签名和 docstring** `waveforms.py:248-307`（基准 221-279）— 理解参数和返回值
3. **参数校验** `waveforms.py:309-316`（基准 280-287）— 注意哪些参数在何时被校验
4. **衰减系数计算** `waveforms.py:319-328`（基准 289-295）— 对照数学物理原理文档中 $a$ 的推导
5. **cutoff 分支** `waveforms.py:331-342`（基准 297-306）— 理解截止时间的解析解
6. **kernel 选择逻辑** `waveforms.py:345-355`（基准 308-317）— 理解四种组合如何映射到四个 kernel
7. **四个 kernel 定义** `waveforms.py:185-244`（基准 165-217）— 从 `F_F`（最简单）到 `T_T`（最完整）依次阅读
8. **对比 F_F 和 T_F** — 观察 `sincos` 的使用方式和 C++ 花括号初始化语法

### 每段代码应观察的问题

| 代码段 | 观察问题 |
|--------|----------|
| 参数校验 | 为什么 `tpr` 的校验放在 cutoff 分支内部而不是函数开头？ |
| `ref = pow(10.0, bwr/20.0)` | 为什么用 `/20` 而不是 `/10`？（提示：幅度 dB vs 功率 dB） |
| `a = -((np.pi*fc*bw)**2) / (4.0*np.log(ref))` | `np.log(ref)` 为什么是负数？负负得正使 `a` 为正？ |
| cutoff 分支 | 为什么 `return` 在 `if` 内部？这之后还有代码会执行吗？ |
| `t = cp.asarray(t)` | 如果 `t` 已经是 CuPy 数组，这行有什么开销？ |
| kernel 选择四个 `if` | 为什么用四个独立 `if` 而不是 `if-elif-else`？有没有可能多个分支同时满足？ |
| `F_F` kernel | 为什么 `yenv` 声明为 `T yenv = ...` 而不是直接 `yI = exp(-a*t*t) * cos(...)`？（提示：可读性 vs 优化） |
| `T_F` kernel 的 `sincos` | `&l_yQ` 在前 `&l_yI` 在后，这个顺序对应 sin 还是 cos？为什么 `yQ` 对应 sin？ |
| `options=("-std=c++11",)` | 为什么需要 C++11？`T yenv { ... }` 花括号初始化是 C++11 特性 |

## 完整代码索引

| 代码元素 | 学习副本行号 | 只读基准行号 |
|----------|-------------|-------------|
| `_gausspulse_kernel_F_F` 定义 | 185-194 | 165-174 |
| `_gausspulse_kernel_F_T` 定义 | 198-207 | 176-185 |
| `_gausspulse_kernel_T_F` 定义 | 212-226 | 187-201 |
| `_gausspulse_kernel_T_T` 定义 | 230-244 | 203-217 |
| `gausspulse` 函数定义 | 248 | 221 |
| 参数校验 | 309-316 | 280-287 |
| `ref` 计算 | 322 | 291 |
| `a` 计算 | 328 | 295 |
| cutoff 分支 | 331-342 | 297-306 |
| `cp.asarray(t)` | 345 | 308 |
| kernel 选择 | 348-355 | 310-317 |

所有代码引用路径：`Learning/cusignal-23.08.00/python/cusignal/waveforms/waveforms.py`
