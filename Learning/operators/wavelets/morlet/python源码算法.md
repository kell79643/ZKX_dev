# morlet 算子学习 — Python 源码算法

> 本文件对应 `Learning/operators/wavelets/morlet/` 目录,记录阶段二(Python 源码算法)。
> 源码以学习副本 `Learning/cusignal-23.08.00` 为准,只读基准为 `ZKX/cusignal-23.08.00`。
> 本阶段深入到代码语法和执行逻辑,逐行解释当前算子的完整源码。

---

## 一、代码定位与接口概览

### 1.1 公开导入路径

`morlet` 算子通过以下路径公开导出:

**顶层 API 入口**(`Learning/cusignal-23.08.00/python/cusignal/__init__.py:85`):
```python
from cusignal.wavelets.wavelets import cwt, morlet, morlet2, qmf, ricker
```

**模块级 API 入口**(`Learning/cusignal-23.08.00/python/cusignal/wavelets/__init__.py:14`):
```python
from cusignal.wavelets.wavelets import cwt, morlet, morlet2, qmf, ricker
```

**函数定义位置**:
- 文件:`Learning/cusignal-23.08.00/python/cusignal/wavelets/wavelets.py`
- 函数签名:67-118 行(只读基准同)
- GPU Kernel:44-64 行(只读基准同)

### 1.2 函数签名

```python
def morlet(M, w=5.0, s=1.0, complete=True):
```

**参数语义**:
- `M` (int):输出序列长度(样本数),对应时间轴上的采样点数;
- `w` (float,默认 5.0):中心角频率 $\omega_0$,无量纲,控制载波振荡速度;
- `s` (float,默认 1.0):缩放因子,控制时间窗口宽度,窗口区间为 $[-s\cdot2\pi, +s\cdot2\pi]$;
- `complete` (bool,默认 True):是否使用完整版(含容许性修正项),False 为标准版。

**返回值语义**:
- 类型:`cupy.ndarray` of `complex128`;
- 形状:`(M,)`,长度为 `M` 的一维复数数组;
- 数值:Morlet 小波在 $M$ 个采样点上的复数取值。

**dtype 强制性**:无论输入参数为何,输出恒为 `complex128`,由 kernel 输出参数声明(46 行)决定。

---

## 二、当前算子的完整相关源码

以下按源码原有顺序完整摘录 `morlet` 函数定义及其直接依赖的 `_morlet_kernel`,包括签名、完整 docstring、docstring 中的全部示例、函数体,以及 kernel 的完整定义。

### 2.1 GPU Kernel 定义(44-64 行)

```python
_morlet_kernel = cp.ElementwiseKernel(
    "float64 w, float64 s, bool complete",
    "complex128 output",
    """
    const double x { start + delta * i };

    thrust::complex<double> temp { exp(
        thrust::complex<double>( 0, w * x ) ) };

    if ( complete ) {
        temp -= exp( -0.5 * ( w * w ) );
    }

    output = temp * exp( -0.5 * ( x * x ) ) * pow( M_PI, -0.25 )
    """,
    "_morlet_kernel",
    options=("-std=c++11",),
    loop_prep="const double end { s * 2.0 * M_PI }; \
               const double start { -s * 2.0 * M_PI }; \
               const double delta { ( end - start ) / ( _ind.size() - 1 ) };",
)
```

### 2.2 morlet 函数定义(67-118 行)

```python
def morlet(M, w=5.0, s=1.0, complete=True):
    """
    Complex Morlet wavelet.

    Parameters
    ----------
    M : int
        Length of the wavelet.
    w : float, optional
        Omega0. Default is 5
    s : float, optional
        Scaling factor, windowed from ``-s*2*pi`` to ``+s*2*pi``. Default is 1.
    complete : bool, optional
        Whether to use the complete or the standard version.

    Returns
    -------
    morlet : (M,) ndarray

    See Also
    --------
    cusignal.gausspulse

    Notes
    -----
    The standard version::

        pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))

    This commonly used wavelet is often referred to simply as the
    Morlet wavelet.  Note that this simplified version can cause
    admissibility problems at low values of `w`.

    The complete version::

        pi**-0.25 * (exp(1j*w*x) - exp(-0.5*(w**2))) * exp(-0.5*(x**2))

    This version has a correction
    term to improve admissibility. For `w` greater than 5, the
    correction term is negligible.

    Note that the energy of the return wavelet is not normalised
    according to `s`.

    The fundamental frequency of this wavelet in Hz is given
    by ``f = 2*s*w*r / M`` where `r` is the sampling rate.

    Note: This function was created before `cwt` and is not compatible
    with it.

    """
    return _morlet_kernel(w, s, complete, size=M)
```

---

## 三、docstring 逐行翻译与解释

以下按 docstring 原有顺序,逐行翻译并解释每一部分的内容。

### 3.1 函数概览(69 行)

**原文**:
```
Complex Morlet wavelet.
```

**翻译**:复数 Morlet 小波。

**解释**:这是函数的功能概览,说明该函数生成的是复数形式的 Morlet 小波(包含实部与虚部),而非仅取实部的简化版本。

### 3.2 参数说明(71-80 行)

#### 3.2.1 参数 M(72-73 行)

**原文**:
```
M : int
    Length of the wavelet.
```

**翻译**:
```
M : int
    小波的长度。
```

**解释**:参数 `M` 指定输出序列的样本点数,对应时间轴上均匀采样的点数。例如 `M=100` 时,输出为一个长度为 100 的复数数组,包含小波在 100 个时间位置上的取值。

#### 3.2.2 参数 w(74-76 行)

**原文**:
```
w : float, optional
    Omega0. Default is 5
```

**翻译**:
```
w : float, 可选
    Omega0(中心角频率)。默认为 5。
```

**解释**:参数 `w` 对应数学公式中的 $\omega_0$,是载波的角频率(单位 rad,无量纲)。取值通常 $\geq 5$ 以满足容许性条件。默认值 5.0 是工程上的平衡选择:既保证足够多的振荡周期(频域集中),又避免过多的时间支撑(时域展宽)。

#### 3.2.3 参数 s(77-79 行)

**原文**:
```
s : float, optional
    Scaling factor, windowed from ``-s*2*pi`` to ``+s*2*pi``. Default is 1.
```

**翻译**:
```
s : float, 可选
    缩放因子,时间窗口从 ``-s*2*pi`` 到 ``+s*2*pi``。默认为 1。
```

**解释**:参数 `s` 控制时间窗口的宽度。实际采样区间为 $x \in [-s\cdot2\pi, +s\cdot2\pi]$,即从 $-2\pi s$ 到 $+2\pi s$。例如 `s=1.0` 时,窗口宽度为 $4\pi \approx 12.566$,覆盖高斯包络的主要部分。增大 `s` 会展宽时间窗口(更多周期),减小 `s` 会压缩窗口(更紧凑)。

#### 3.2.4 参数 complete(80 行)

**原文**:
```
complete : bool, optional
    Whether to use the complete or the standard version.
```

**翻译**:
```
complete : bool, 可选
    是否使用完整版或标准版。
```

**解释**:参数 `complete` 控制是否应用容许性修正项。`complete=True`(默认)时,使用完整版公式(减去 $e^{-\omega_0^2/2}$),严格满足小波容许性条件;`complete=False` 时,使用标准版公式(无修正项),在低 $\omega_0$ 时可能不满足容许性。

### 3.3 返回值说明(82-84 行)

**原文**:
```
Returns
-------
morlet : (M,) ndarray
```

**翻译**:
```
返回值
------
morlet : (M,) ndarray
```

**解释**:返回值是一个形状为 `(M,)` 的一维数组,包含 `M` 个复数样本点,对应 Morlet 小波在时间轴上的采样值。

### 3.4 参见(86-88 行)

**原文**:
```
See Also
--------
cusignal.gausspulse
```

**翻译**:
```
参见
----
cusignal.gausspulse
```

**解释**:提供相关函数的交叉引用。`gausspulse` 是高斯脉冲生成函数,同样涉及高斯包络与载波调制,但参数化方式不同(用中心频率 `fc` 而非角频率 $\omega_0$)。

### 3.5 注释部分(90-117 行)

#### 3.5.1 标准版公式(91-95 行)

**原文**:
```
The standard version::

    pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))
```

**翻译**:
```
标准版::

    pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))
```

**解释**:给出标准 Morlet 小波的数学公式,对应原理 1 中的定义:
- `pi**-0.25`:归一化常数 $\pi^{-0.25}$,确保能量归一化;
- `exp(1j*w*x)`:复指数载波 $e^{j\omega_0 x}$,提供角频率为 $\omega_0$ 的振荡;
- `exp(-0.5*(x**2))`:高斯包络 $e^{-x^2/2}$,控制时间局部化。

这是最常用的 Morlet 小波形式,但严格不满足容许性条件(直流分量非零)。

#### 3.5.2 容许性问题说明(96-98 行)

**原文**:
```
This commonly used wavelet is often referred to simply as the
Morlet wavelet.  Note that this simplified version can cause
admissibility problems at low values of `w`.
```

**翻译**:
```
这个常用小波通常简称为 Morlet 小波。注意,这个简化版本在低 `w` 值时
可能导致容许性问题。
```

**解释**:说明标准版在低 $\omega_0$ 时的容许性风险。当 $\omega_0 < 5$ 时,$e^{-\omega_0^2/2}$ 不够小,小波均值偏离零较明显,逆变换可能不稳定。这是使用标准版时需注意的限制。

#### 3.5.3 完整版公式(100-102 行)

**原文**:
```
The complete version::

    pi**-0.25 * (exp(1j*w*x) - exp(-0.5*(w**2))) * exp(-0.5*(x**2))
```

**翻译**:
```
完整版::

    pi**-0.25 * (exp(1j*w*x) - exp(-0.5*(w**2))) * exp(-0.5*(x**2))
```

**解释**:给出完整 Morlet 小波的数学公式,对应原理 2 中的修正形式:
- 括号内的减法 `exp(1j*w*x) - exp(-0.5*(w**2))` 是修正项;
- `exp(-0.5*(w**2))` 消除直流分量,严格满足容许性条件;
- 数学推导见阶段一文档原理 2。

#### 3.5.4 修正项作用说明(104-107 行)

**原文**:
```
This version has a correction
term to improve admissibility. For `w` greater than 5, the
correction term is negligible.
```

**翻译**:
```
此版本包含一个修正项以改善容许性。对于 `w` 大于 5 的情况,
修正项可以忽略。
```

**解释**:说明修正项 $e^{-\omega_0^2/2}$ 的数值大小。当 $\omega_0 > 5$ 时,$e^{-\omega_0^2/2} < e^{-12.5} \approx 3.7 \times 10^{-6}$,已极小,标准版与完整版数值上几乎等价。这给出了参数选择的指导:高 $\omega_0$ 时可任选版本,低 $\omega_0$ 时必须用完整版。

#### 3.5.5 能量归一化说明(109-110 行)

**原文**:
```
Note that the energy of the return wavelet is not normalised
according to `s`.
```

**翻译**:
```
注意,返回小波的能量未按 `s` 归一化。
```

**解释**:说明能量归一化策略。虽然公式中包含 $\pi^{-0.25}$ 因子使连续积分意义下的能量为 1,但离散序列的能量会因截断而略有偏差,且不随 `s` 缩放调整。若需按尺度归一化,应使用 `morlet2` 函数(专门为 CWT 设计)。

#### 3.5.6 基频公式(112-113 行)

**原文**:
```
The fundamental frequency of this wavelet in Hz is given
by ``f = 2*s*w*r / M`` where `r` is the sampling rate.
```

**翻译**:
```
此小波在 Hz 下的基频由 ``f = 2*s*w*r / M`` 给出,其中 `r` 为采样率。
```

**解释**:给出基频 $f_{\text{fund}}$ 与参数的关系。推导见阶段一原理 5:
- 时间窗口总长 $T = 4\pi s / r$;
- 载波在窗口内完成 $N_{\text{cycles}} = 2s\omega_0$ 个周期;
- 基频 $f = N_{\text{cycles}} / T \cdot r = 2s\omega_0 r / M$。

这个公式帮助用户根据目标频率选择参数 `s` 与 `w`。

#### 3.5.7 与 CWT 不兼容说明(115-117 行)

**原文**:
```
Note: This function was created before `cwt` and is not compatible
with it.
```

**翻译**:
```
注意:此函数在 `cwt` 之前创建,不兼容 `cwt`。
```

**解释**:说明 `morlet` 与 `morlet2` 的关系。`morlet` 是独立的小波生成函数,不按尺度归一化,不能直接作为 `cwt` 的母小波参数。若需进行连续小波变换,应使用 `morlet2`(199-256 行),它按 $\sqrt{1/s}$ 归一化,兼容 CWT 的尺度约定。

---

## 四、源代码逐行解释

以下按源码原有顺序,逐行解释 `_morlet_kernel` 定义与 `morlet` 函数体的每一行非空内容。

### 4.1 GPU Kernel 定义(44-64 行)

#### 第 44 行:Kernel 对象赋值语句

**源码**:
```python
_morlet_kernel = cp.ElementwiseKernel(
```

**行号**:Learning 副本 44 行,只读基准 44 行。

**语法说明**:
- `_morlet_kernel`:模块级私有变量名,以下划线开头表示仅供本模块内部使用;
- `=`:赋值运算符,将右边的 `ElementwiseKernel` 对象绑定到左边变量;
- `cp.ElementwiseKernel(...)`:CuPy 库的元素级 kernel 构造函数,用于在 GPU 上并行计算逐点操作。

**作用**:创建一个 GPU kernel 对象,专门用于生成 Morlet 小波序列。

#### 第 45-46 行:输入输出参数声明

**源码**:
```python
    "float64 w, float64 s, bool complete",
    "complex128 output",
```

**行号**:Learning 副本 45-46 行,只读基准 45-46 行。

**语法说明**:
- 第一个字符串:输入参数列表,逗号分隔,格式为 `"dtype param1, dtype param2, ..."`;
  - `float64 w`:第一个输入参数,类型为 64 位浮点数,对应中心角频率 $\omega_0$;
  - `float64 s`:第二个输入参数,类型为 64 位浮点数,对应缩放因子;
  - `bool complete`:第三个输入参数,类型为布尔值,控制是否应用修正项。
- 第二个字符串:输出参数声明;
  - `complex128 output`:输出参数,类型为 128 位复数(64 位实部 + 64 位虚部),对应小波样本值。

**作用**:声明 kernel 的输入输出接口,指定参数类型与名称,这些参数在 kernel 体中可直接访问。

#### 第 47-57 行:Kernel 体(C++/CUDA 代码)

**源码**:
```python
    """
    const double x { start + delta * i };

    thrust::complex<double> temp { exp(
        thrust::complex<double>( 0, w * x ) ) };

    if ( complete ) {
        temp -= exp( -0.5 * ( w * w ) );
    }

    output = temp * exp( -0.5 * ( x * x ) ) * pow( M_PI, -0.25 )
    """,
```

**行号**:Learning 副本 47-57 行,只读基准 47-57 行。

**逐行解释**:

**(48 行)时间位置计算**:
```cpp
const double x { start + delta * i };
```
- `const double x`:声明一个双精度浮点常量 `x`,表示当前样本的时间位置;
- `start`:循环准备阶段(`loop_prep`)预计算的起始位置 $-s \cdot 2\pi$;
- `delta`:预计算的采样间隔 $(4\pi s)/(M-1)$;
- `i`:CuPy 自动提供的线程索引变量,范围 $[0, M-1]$,对应样本序号;
- `start + delta * i`:等距采样公式,第 `i` 个样本的时间位置 $x_i = -2\pi s + i \cdot \frac{4\pi s}{M-1}$;
- 大括号 `{ ... }`:C++11 初始化语法,等价于 `const double x = start + delta * i;`。

**(49-51 行)复指数载波计算**:
```cpp
thrust::complex<double> temp { exp(
    thrust::complex<double>( 0, w * x ) ) };
```
- `thrust::complex<double> temp`:声明一个 Thrust 库的复数变量 `temp`,用于存储载波值;
- `exp(...)`:指数函数,计算 $e^{\text{复数}}$;
- `thrust::complex<double>( 0, w * x )`:构造一个复数,实部为 0,虚部为 $w \cdot x$;
- 等价于计算 $e^{j \omega_0 x}$,其中 $j$ 为虚数单位;
- 数学映射:`temp = exp(1j * w * x)`,对应公式中的 $e^{j\omega_0 x}$。

**(53-55 行)容许性修正项(条件执行)**:
```cpp
if ( complete ) {
    temp -= exp( -0.5 * ( w * w ) );
}
```
- `if ( complete )`:条件语句,检查布尔输入参数 `complete`;
- `temp -= exp( -0.5 * ( w * w ) )`:复合减法赋值运算符,等价于 `temp = temp - exp(-0.5 * w * w)`;
- `exp( -0.5 * ( w * w ) )`:计算修正项 $e^{-\omega_0^2/2}$;
- 数学映射:`temp = exp(1j * w * x) - exp(-0.5 * w**2)`,对应完整版公式中的减法;
- 只有 `complete=True` 时执行,否则跳过修正项(标准版)。

**(57 行)输出赋值(高斯包络与归一化)**:
```cpp
output = temp * exp( -0.5 * ( x * x ) ) * pow( M_PI, -0.25 )
```
- `output`:kernel 输出参数,将计算结果写入 GPU 内存;
- `temp`:前面计算的复指数载波(或已修正);
- `exp( -0.5 * ( x * x ) )`:高斯包络 $e^{-x^2/2}$;
- `pow( M_PI, -0.25 )`:归一化常数 $\pi^{-0.25}$,其中 `M_PI` 是 `<cmath>` 定义的 $\pi$ 常量;
- `*`:乘法运算符,按从左到右顺序连乘;
- 数学映射:`output = temp * exp(-0.5 * x**2) * π^{-0.25}`,对应完整公式 $\psi(t) = \pi^{-0.25} \cdot \text{temp} \cdot e^{-t^2/2}$。

**(47, 58 行)三引号字符串**:
- Python 三引号字符串 `"""..."""`,用于包裹多行 C++/CUDA 代码;
- CuPy 会将此字符串传递给 NVRTC(NVIDIA Runtime Compiler),在运行时编译为 GPU 可执行代码。

#### 第 59 行:Kernel 名称

**源码**:
```python
    "_morlet_kernel",
```

**行号**:Learning 副本 59 行,只读基准 59 行。

**语法说明**:
- `"_morlet_kernel"`:kernel 的标识符字符串,用于调试与性能分析;
- 作为 `ElementwiseKernel` 的第 4 个位置参数传递。

**作用**:为 kernel 提供一个名称,便于在 profiling 工具(如 `nvprof`)中识别。

#### 第 60 行:编译选项

**源码**:
```python
    options=("-std=c++11",),
```

**行号**:Learning 副本 60 行,只读基准 60 行。

**语法说明**:
- `options=(...)`:关键字参数,传递编译选项给 NVRTC;
- `"-std=c++11"`:指定使用 C++11 标准(支持大括号初始化、`auto` 类型推导等特性);
- 单元素元组语法:`("item",)` 注意逗号不能省略。

**作用**:告诉 NVRTC 编译器使用 C++11 标准,确保 kernel 体中的现代 C++ 语法合法。

#### 第 61-63 行:循环准备代码

**源码**:
```python
    loop_prep="const double end { s * 2.0 * M_PI }; \
               const double start { -s * 2.0 * M_PI }; \
               const double delta { ( end - start ) / ( _ind.size() - 1 ) };",
```

**行号**:Learning 副本 61-63 行,只读基准 61-63 行。

**语法说明**:
- `loop_prep="..."`:关键字参数,传递循环准备阶段的 C++ 代码字符串;
- 反斜杠 `\`:Python 行续符,将一个长字符串分成多行书写;
- `_ind.size()`:CuPy 提供的内置变量,返回输出数组的长度(即 `M`),等价于 Python 中的 `len(output)`。

**逐行解释**:

**(61 行)结束位置预计算**:
```cpp
const double end { s * 2.0 * M_PI };
```
- 计算时间轴结束位置 `end = s * 2π`,对应 $+s \cdot 2\pi$。

**(62 行)起始位置预计算**:
```cpp
const double start { -s * 2.0 * M_PI };
```
- 计算时间轴起始位置 `start = -s * 2π`,对应 $-s \cdot 2\pi$。

**(63 行)采样间隔预计算**:
```cpp
const double delta { ( end - start ) / ( _ind.size() - 1 ) };
```
- 计算采样间隔 `delta = (end - start) / (M - 1) = 4πs / (M - 1)`;
- 除以 `_ind.size() - 1` 而非 `_ind.size()` 是因为 $M$ 个点之间有 $M-1$ 个间隔。

**作用**:在 kernel 启动前预计算这些常量,避免在每个线程中重复计算,提高性能。

#### 第 64 行:Kernel 定义结束

**源码**:
```python
)
```

**行号**:Learning 副本 64 行,只读基准 64 行。

**语法说明**:
- `)`:闭合 `ElementwiseKernel(...)` 构造函数调用;
- 此时 `_morlet_kernel` 对象已完成创建,可在后续函数中调用。

### 4.2 morlet 函数体(67-118 行)

#### 第 67 行:函数签名

**源码**:
```python
def morlet(M, w=5.0, s=1.0, complete=True):
```

**行号**:Learning 副本 67 行,只读基准 67 行。

**语法说明**:
- `def`:Python 函数定义关键字;
- `morlet`:函数名,与导出名称一致;
- `(...)`:参数列表,括号内为形式参数;
- `M`:位置参数,无默认值,必须由调用方提供;
- `w=5.0`:关键字参数,默认值为 5.0;
- `s=1.0`:关键字参数,默认值为 1.0;
- `complete=True`:关键字参数,默认值为 True;
- `:`:函数体开始标记,后续为缩进的代码块。

**参数语义**:
- `M`:输出序列长度,控制采样点数;
- `w`:中心角频率 $\omega_0$,控制载波振荡速度;
- `s`:缩放因子,控制时间窗口宽度;
- `complete`:是否应用容许性修正项。

#### 第 118 行:Kernel 调用与返回

**源码**:
```python
    return _morlet_kernel(w, s, complete, size=M)
```

**行号**:Learning 副本 118 行,只读基准 118 行。

**语法说明**:
- `return`:Python 返回语句,将结果返回给调用方;
- `_morlet_kernel(w, s, complete, size=M)`:调用 GPU kernel 对象;
  - `_morlet_kernel`:前面定义的 `ElementwiseKernel` 对象;
  - `w, s, complete`:按顺序传递给 kernel 的输入参数,对应 kernel 定义中的 `"float64 w, float64 s, bool complete"`;
  - `size=M`:关键字参数,指定输出数组的长度,对应 kernel 的线程数;
- kernel 执行流程:
  1. CuPy 在 GPU 上启动 `M` 个并行线程;
  2. 每个线程获得一个唯一的索引 `i`(从 0 到 `M-1`);
  3. 每个线程执行 kernel 体中的 C++/CUDA 代码,计算一个样本值;
  4. 所有样本值写入输出数组 `output`;
  5. kernel 返回一个形状为 `(M,)` 的 CuPy 数组;
  6. Python 函数返回该数组。

**作用**:这是函数的唯一可执行语句,将参数传递给 GPU kernel 并返回计算结果。整个 Morlet 小波生成过程在 GPU 上并行完成。

---

## 五、调用链与算法总结

### 5.1 调用链

```
用户代码
  ↓
cusignal.morlet(M, w, s, complete)  # Python 函数入口
  ↓
_morlet_kernel(w, s, complete, size=M)  # GPU kernel 调用
  ↓
CuPy ElementwiseKernel 执行引擎
  ↓
NVRTC 编译 C++/CUDA 代码为 GPU 可执行代码
  ↓
GPU 并行执行:
  - 每个线程计算一个样本:
    1. 计算 x = -2πs + i * 4πs/(M-1)
    2. 计算 temp = exp(1j * w * x)
    3. 若 complete=True: temp -= exp(-w²/2)
    4. 计算 output = temp * exp(-x²/2) * π^{-0.25}
  ↓
返回 CuPy 数组 (M,) complex128
```

### 5.2 算法总结

**输入**:
- `M`:序列长度;
- `w`:中心角频率 $\omega_0$;
- `s`:缩放因子;
- `complete`:是否应用修正项。

**步骤**:
1. **时间轴采样**(loop_prep 阶段):
   - 预计算 `start = -s * 2π`, `end = s * 2π`, `delta = 4πs / (M-1)`;
   - 等距采样 $M$ 个时间点 $x_0, x_1, ..., x_{M-1}$。

2. **并行计算**(kernel 体):
   - 对每个样本索引 `i`(并行执行):
     - 计算时间位置 $x_i = \text{start} + \text{delta} \cdot i$;
     - 计算复指数载波 $\text{temp} = e^{j \omega_0 x_i}$;
     - 若 `complete=True`,应用修正 $\text{temp} \leftarrow \text{temp} - e^{-\omega_0^2/2}$;
     - 应用高斯包络与归一化 $\psi_i = \text{temp} \cdot e^{-x_i^2/2} \cdot \pi^{-0.25}$;
     - 写入输出数组。

3. **返回结果**:
   - 返回形状为 `(M,)` 的 `complex128` 数组。

**输出**:
- 形状 `(M,)` 的 CuPy 数组,包含 Morlet 小波在 $M$ 个采样点上的复数取值。

---

## 六、数学映射、边界、复杂度和阅读检查

### 6.1 数学公式与代码语句对应表

| 数学公式 | 代码语句 | 行号 | 说明 |
| --- | --- | --- | --- |
| $x_i = -2\pi s + i \cdot \frac{4\pi s}{M-1}$ | `const double x { start + delta * i };` | 48 | 时间轴等距采样 |
| $e^{j\omega_0 x_i}$ | `exp(thrust::complex<double>( 0, w * x ))` | 50-51 | 复指数载波 |
| $e^{-\omega_0^2/2}$ | `exp( -0.5 * ( w * w ) )` | 54 | 容许性修正项 |
| $\psi_i = \pi^{-0.25} e^{j\omega_0 x_i} e^{-x_i^2/2}$ | `output = temp * exp( -0.5 * ( x * x ) ) * pow( M_PI, -0.25 )` (标准版) | 57 | 标准 Morlet 小波 |
| $\psi_i = \pi^{-0.25} (e^{j\omega_0 x_i} - e^{-\omega_0^2/2}) e^{-x_i^2/2}$ | 同上(完整版,temp 已修正) | 57 | 完整 Morlet 小波 |
| $\pi^{-0.25}$ | `pow( M_PI, -0.25 )` | 57 | 归一化常数 |

### 6.2 边界处理与数值稳定性

**边界处理**:
- 时间窗口固定为 $[-s\cdot2\pi, +s\cdot2\pi]$,高斯包络在边界处衰减至 $e^{-(2\pi s)^2/2}$;
- 例如 $s=1$ 时,边界值衰减至 $e^{-2\pi^2} \approx 2.7 \times 10^{-9}$,已接近零;
- 无显式锥化边界处理,依赖高斯包络自然衰减。

**数值稳定性**:
- 复指数 `exp(1j * w * x)`:使用 Thrust 库的复数指数函数,数值稳定;
- 高斯指数 `exp(-x²/2)`:自变量 $-x²/2 \leq 0$,不会出现指数爆炸;
- 修正项 `exp(-w²/2)`:当 $w \geq 5$ 时趋近于零,数值安全;
- 归一化常数 $\pi^{-0.25} \approx 0.752$:范围合理,无溢出风险。

**dtype 强制性**:
- 输出恒为 `complex128`,即使修正项为零(标准版在高 $w$ 时);
- 实数输入参数 `w`, `s` 在 kernel 内部被提升为 `double`(64 位浮点)。

### 6.3 时间复杂度与空间复杂度

**时间复杂度**:
- CPU 端:函数调用为 $O(1)$(参数传递与 kernel 启动);
- GPU 端:每个线程计算一个样本,并行度为 $M$,单样本计算复杂度 $O(1)$(有限次浮点运算);
- 总时间复杂度:$O(M)$,但实际执行时间由 GPU 并行度决定,接近 $O(1)$(假设 GPU 有足够线程)。

**空间复杂度**:
- 输入空间:常数个标量参数(`w`, `s`, `complete`),$O(1)$;
- 输出空间:形状为 `(M,)` 的 `complex128` 数组,$O(M)$;
- GPU 临时空间:CuPy 内部管理的 kernel 参数与寄存器,$O(1)$ per thread。

**性能瓶颈**:
- GPU kernel 启动开销(微秒级);
- 内存分配:输出数组 `complex128` 需要 $16M$ 字节(16 字节/样本 × M 样本);
- 计算:每个样本约 10 次浮点运算(复指数、高斯、乘法),GPU 可高效并行;
- 数据传输:若结果需传回 CPU,则存在 PCIe 带宽瓶颈,但 `morlet` 本身不涉及此步骤。

### 6.4 建议阅读顺序与观察要点

**推荐阅读顺序**:
1. **先看函数签名**(67 行):理解参数语义与默认值;
2. **再看 docstring**(69-117 行):掌握数学公式与使用注意事项;
3. **最后看 kernel 定义**(44-64 行):理解 GPU 并行计算逻辑。

**观察要点**:
1. **参数验证缺失**:函数未检查 `M > 0`、`w > 0`、`s > 0` 等合法性,用户需自行保证;
2. **版本选择**:`complete=True` 为默认,建议保持,除非明确需要标准版;
3. **时间窗口截断**:高斯包络理论上无限延伸,但被截断至 $[-2\pi s, +2\pi s]$,观察边界衰减是否足够;
4. **与 morlet2 的区别**:对比 `morlet2`(199-256 行),观察归一化方式差异($\sqrt{1/s}$ 因子);
5. **基频公式验证**:手动计算一个例子,验证 $f = 2swr/M$ 的正确性;
6. **GPU 性能**:若关注性能,可用 `nvprof` 观察 kernel 执行时间与内存带宽。

---

## 七、自检问题

1. 写出 `morlet(M, w=5.0, s=1.0, complete=True)` 的调用示例,并说明输出数组的 dtype 与 shape。
2. 解释 `_morlet_kernel` 的输入输出参数声明,为何输出固定为 `complex128` 而非根据 `complete` 变化?
3. 逐行解释 kernel 体中的第 48 行 `const double x { start + delta * i };`,说明 `start`、`delta`、`i` 的来源与含义。
4. 为什么 `complete=True` 时需要执行 `temp -= exp( -0.5 * ( w * w ) )`?这个修正项的数学含义是什么?
5. 解释 `loop_prep` 中 `delta = (end - start) / (_ind.size() - 1)` 为何除以 `_ind.size() - 1` 而非 `_ind.size()`?
6. 函数体只有一行 `return _morlet_kernel(w, s, complete, size=M)`,整个计算在何处完成?CPU 端还是 GPU 端?
7. 计算 $M=100$, $w=5.0$, $s=1.0$, $r=1000$ Hz 时的基频 $f = 2swr/M$,并说明如何选择 `s` 使基频接近 10 Hz。
8. docstring 为何说"Note: This function was created before `cwt` and is not compatible with it"?`morlet` 与 `morlet2` 的主要区别是什么?
9. 若需生成一个时间窗口更宽(覆盖更多周期)的 Morlet 小波,应调整哪个参数?增大还是减小?
10. 观察边界样本(如 $i=0$ 或 $i=M-1$)的数值大小,验证高斯包络在 $x = \pm 2\pi s$ 处的衰减是否足够(以 $s=1$ 为例)。

---

## 八、参考资料

| 序号 | 名称 | 作者/机构 | 链接 | 访问日期 |
| --- | --- | --- | --- | --- |
| 1 | CuPy ElementwiseKernel 文档 | CuPy 官方 | https://docs.cupy.dev/en/stable/reference/generated/cupy.ElementwiseKernel.html | 2026-08-08 |
| 2 | Thrust 库文档(复数运算) | NVIDIA | https://thrust.github.io/ | 2026-08-08 |
| 3 | NVRTC(NVIDIA Runtime Compiler)文档 | NVIDIA | https://docs.nvidia.com/cuda/nvrtc/ | 2026-08-08 |
| 4 | SciPy signal.morlet 文档(对比参考) | SciPy 官方 | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.morlet.html | 2026-08-08 |

> 说明:资料 1 提供CuPy ElementwiseKernel 的 API 文档;资料 2 提供 Thrust 复数库的使用说明;资料 3 提供 NVRTC 编译器背景;资料 4 提供 SciPy 的 `morlet` 实现(纯 Python)对比。