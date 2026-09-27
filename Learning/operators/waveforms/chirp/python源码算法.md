# chirp：cuSignal 23.08.00 Python 源码算法

> 当前阶段：阶段二（Python 源码算法）
> 算子：`waveforms/chirp`
> 基准：`ZKX/cusignal-23.08.00` 只读原文
> 学习副本：`Learning/cusignal-23.08.00`，只新增合法 `<学习注释：...>`
> 核对日期：2026-08-06

## 1. 代码定位与接口概览

### 1.1 公开导入路径

| 层级          | 学习副本位置                                                                  | 只读基准位置                                                             | 作用                                                          |
| ------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------- |
| 顶层 API      | `Learning/cusignal-23.08.00/python/cusignal/__init__.py:80-86`              | `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:78-84`              | 允许`from cusignal import chirp` 和 `cusignal.chirp(...)` |
| waveforms 包  | `Learning/cusignal-23.08.00/python/cusignal/waveforms/__init__.py:14-21`    | `ZKX/cusignal-23.08.00/python/cusignal/waveforms/__init__.py:14-20`    | 允许`cusignal.waveforms.chirp`                              |
| kernel 与函数 | `Learning/cusignal-23.08.00/python/cusignal/waveforms/waveforms.py:368-666` | `ZKX/cusignal-23.08.00/python/cusignal/waveforms/waveforms.py:321-519` | 五个逐元素 kernel、公开函数、校验与分发                       |

公开签名：

```python
chirp(t, f0, t1, f1, method="linear", phi=0,
      vertex_zero=True, type="real")
```

参数语义：

| 参数            | 默认值       | 语义                                                                |
| --------------- | ------------ | ------------------------------------------------------------------- |
| `t`           | 无           | 求值位置；转为 CuPy 数组，输出 shape 与它一致                       |
| `f0`          | 无           | `t=0` 时的瞬时频率，单位是 cycles / `t` 的单位                  |
| `t1`          | 无           | 规定`f(t1)=f1` 的参考位置，不负责裁剪 `t`                       |
| `f1`          | 无           | `t=t1` 时的瞬时频率                                               |
| `method`      | `"linear"` | `linear`、`quadratic`、`logarithmic`、`hyperbolic` 及短别名 |
| `phi`         | `0`        | degree 初相位，函数体转成 rad                                       |
| `vertex_zero` | `True`     | 仅 quadratic 使用；选择频率抛物线顶点位置                           |
| `type`        | `"real"`   | 仅 linear 使用；选择实值或复值输出                                  |

## 2. 当前算子的完整相关源码

以下代码按只读基准原样摘录；没有省略 `chirp` 的 docstring、示例、函数体或直接 kernel。

### 2.1 公开导出语句

`ZKX/cusignal-23.08.00/python/cusignal/waveforms/__init__.py:14-20`：

```python
from cusignal.waveforms.waveforms import (
    chirp,
    gausspulse,
    sawtooth,
    square,
    unit_impulse,
)
```

`ZKX/cusignal-23.08.00/python/cusignal/__init__.py:78-84`：

```python
from cusignal.waveforms.waveforms import (
    chirp,
    gausspulse,
    sawtooth,
    square,
    unit_impulse,
)
```

同一导入组还列出其他 waveforms 算子；它们只是原语句不可拆分的上下文，不在本文展开。

逐行解释如下。两个文件的原始导入块文字相同，先分别给出定位，再共同解释原文；不能只用行号代替源码。

学习副本 `cusignal/waveforms/__init__.py:14-21`，基准 `cusignal/waveforms/__init__.py:14-20`：

```python
from cusignal.waveforms.waveforms import (
    chirp,
    gausspulse,
    sawtooth,
    square,
    unit_impulse,
)
```

学习副本 `cusignal/__init__.py:80-86`，基准 `cusignal/__init__.py:78-84`：

```python
from cusignal.waveforms.waveforms import (
    chirp,
    gausspulse,
    sawtooth,
    square,
    unit_impulse,
)
```

第一行的 `from ... import (` 从实现模块导入名字；左括号允许导入列表跨行。`chirp,` 把算子绑定到当前模块，尾逗号分隔下一个名字：在顶层文件中形成 `cusignal.chirp`，在 waveforms 包中形成 `cusignal.waveforms.chirp`。`gausspulse` 至 `unit_impulse` 是同一不可拆分原始导入块中的其他公开算子，不参与 chirp 计算。最后的 `)` 闭合多行导入语句。

### 2.2 五个直接 kernel

`ZKX/cusignal-23.08.00/python/cusignal/waveforms/waveforms.py:321-401`：

```python
_chirp_phase_lin_kernel_real = cp.ElementwiseKernel(
    "T t, T f0, T t1, T f1, T phi",
    "T phase",
    """
    const T beta { (f1 - f0) / t1 };
    const T temp { 2 * M_PI * (f0 * t + 0.5 * beta * t * t) };
    // Convert  phi to radians.
    phase = cos(temp + phi);
    """,
    "_chirp_phase_lin_kernel",
    options=("-std=c++11",),
)

_chirp_phase_lin_kernel_cplx = cp.ElementwiseKernel(
    "T t, T f0, T t1, T f1, T phi",
    "Y phase",
    """
    const T beta { (f1 - f0) / t1 };
    const T temp { 2 * M_PI * (f0 * t + 0.5 * beta * t * t) };
    // Convert  phi to radians.
    phase = Y(cos(temp + phi), cos(temp + phi + M_PI/2) * -1);
    """,
    "_chirp_phase_lin_kernel",
    options=("-std=c++11",),
)

_chirp_phase_quad_kernel = cp.ElementwiseKernel(
    "T t, T f0, T t1, T f1, T phi, bool vertex_zero",
    "T phase",
    """
    T temp {};
    const T beta { (f1 - f0) / (t1 * t1) };
    if ( vertex_zero ) {
        temp = 2 * M_PI * (f0 * t + beta * (t * t * t) / 3);
    } else {
        temp = 2 * M_PI *
            ( f1 * t + beta *
            ( ( (t1 - t) * (t1 - t) * (t1 - t) ) - (t1 * t1 * t1)) / 3);
    }
    // Convert  phi to radians.
    phase = cos(temp + phi);
    """,
    "_chirp_phase_quad_kernel",
    options=("-std=c++11",),
)

_chirp_phase_log_kernel = cp.ElementwiseKernel(
    "T t, T f0, T t1, T f1, T phi",
    "T phase",
    """
    T temp {};
    if ( f0 == f1 ) {
        temp = 2 * M_PI * f0 * t;
    } else {
        T beta { t1 / log(f1 / f0) };
        temp = 2 * M_PI * beta * f0 * ( pow(f1 / f0, t / t1) - 1.0 );
    }
    // Convert  phi to radians.
    phase = cos(temp + phi);
    """,
    "_chirp_phase_log_kernel",
    options=("-std=c++11",),
)

_chirp_phase_hyp_kernel = cp.ElementwiseKernel(
    "T t, T f0, T t1, T f1, T phi",
    "T phase",
    """
    T temp {};
    if ( f0 == f1 ) {
        temp = 2 * M_PI * f0 * t;
    } else {
        T sing { -f1 * t1 / (f0 - f1) };
        temp = 2 * M_PI * ( -sing * f0 ) * log( abs( 1 - t / sing ) );
    }
    // Convert  phi to radians.
    phase = cos(temp + phi);
    """,
    "_chirp_phase_hyp_kernel",
    options=("-std=c++11",),
)
```

### 2.3 完整 `chirp` 定义、docstring、示例与函数体

`ZKX/cusignal-23.08.00/python/cusignal/waveforms/waveforms.py:404-519`：

```python
def chirp(t, f0, t1, f1, method="linear", phi=0, vertex_zero=True, type="real"):
    """Frequency-swept cosine generator.

    In the following, 'Hz' should be interpreted as 'cycles per unit';
    there is no requirement here that the unit is one second.  The
    important distinction is that the units of rotation are cycles, not
    radians. Likewise, `t` could be a measurement of space instead of time.

    Parameters
    ----------
    t : array_like
        Times at which to evaluate the waveform.
    f0 : float
        Frequency (e.g. Hz) at time t=0.
    t1 : float
        Time at which `f1` is specified.
    f1 : float
        Frequency (e.g. Hz) of the waveform at time `t1`.
    method : {'linear', 'quadratic', 'logarithmic', 'hyperbolic'}, optional
        Kind of frequency sweep.  If not given, `linear` is assumed.  See
        Notes below for more details.
    phi : float, optional
        Phase offset, in degrees. Default is 0.
    vertex_zero : bool, optional
        This parameter is only used when `method` is 'quadratic'.
        It determines whether the vertex of the parabola that is the graph
        of the frequency is at t=0 or t=t1.
    type : {'real', 'complex'}, optional
        Specify output chirp type, only applicable when `method` is 'linear'.

    Returns
    -------
    y : ndarray
        A numpy array containing the signal evaluated at `t` with the
        requested time-varying frequency.  More precisely, the function
        returns ``cos(phase + (pi/180)*phi)`` where `phase` is the integral
        (from 0 to `t`) of ``2*pi*f(t)``. ``f(t)`` is defined below.

    Examples
    --------
    The following will be used in the examples:

    >>> from cusignal import chirp, spectrogram
    >>> import matplotlib.pyplot as plt
    >>> import cupy as cp

    For the first example, we'll plot the waveform for a linear chirp
    from 6 Hz to 1 Hz over 10 seconds:

    >>> t = cp.linspace(0, 10, 5001)
    >>> w = chirp(t, f0=6, f1=1, t1=10, method='linear')
    >>> plt.plot(cp.asnumpy(t), cp.asnumpy(w))
    >>> plt.title("Linear Chirp, f(0)=6, f(10)=1")
    >>> plt.xlabel('t (sec)')
    >>> plt.show()

    For the remaining examples, we'll use higher frequency ranges,
    and demonstrate the result using `cusignal.spectrogram`.
    We'll use a 10 second interval sampled at 8000 Hz.

    >>> fs = 8000
    >>> T = 10
    >>> t = cp.linspace(0, T, T*fs, endpoint=False)

    Quadratic chirp from 1500 Hz to 250 Hz over 10 seconds
    (vertex of the parabolic curve of the frequency is at t=0):

    >>> w = chirp(t, f0=1500, f1=250, t1=10, method='quadratic')
    >>> ff, tt, Sxx = spectrogram(w, fs=fs, noverlap=256, nperseg=512,
    ...                           nfft=2048)
    >>> plt.pcolormesh(cp.asnumpy(tt), cp.asnumpy(ff[:513]),
                       cp.asnumpy(Sxx[:513]), cmap='gray_r')
    >>> plt.title('Quadratic Chirp, f(0)=1500, f(10)=250')
    >>> plt.xlabel('t (sec)')
    >>> plt.ylabel('Frequency (Hz)')
    >>> plt.grid()
    >>> plt.show()
    """

    t = cp.asarray(t)

    phi *= np.pi / 180

    if method in ["linear", "lin", "li"]:
        if type == "real":
            return _chirp_phase_lin_kernel_real(t, f0, t1, f1, phi)
        elif type == "complex":
            phase = cp.empty(t.shape, dtype=cp.complex64)
            if np.issubclass_(t.dtype, (np.float64)):
                phase = cp.empty(t.shape, dtype=cp.complex128)
            _chirp_phase_lin_kernel_cplx(t, f0, t1, f1, phi, phase)
            return phase
        else:
            raise NotImplementedError("No kernel for type {}".format(type))

    elif method in ["quadratic", "quad", "q"]:
        return _chirp_phase_quad_kernel(t, f0, t1, f1, phi, vertex_zero)

    elif method in ["logarithmic", "log", "lo"]:
        if f0 * f1 <= 0.0:
            raise ValueError(
                "For a logarithmic chirp, f0 and f1 must be "
                "nonzero and have the same sign."
            )
        return _chirp_phase_log_kernel(t, f0, t1, f1, phi)

    elif method in ["hyperbolic", "hyp"]:
        if f0 == 0 or f1 == 0:
            raise ValueError("For a hyperbolic chirp, f0 and f1 must be " "nonzero.")
        return _chirp_phase_hyp_kernel(t, f0, t1, f1, phi)

    else:
        raise ValueError(
            "method must be 'linear', 'quadratic', 'logarithmic',"
            " or 'hyperbolic', but a value of %r was given." % method
        )
```

## 3. docstring 逐行翻译与解释

本节按 docstring 的自然语义分块。每个块原样贴出一段完整原文，并按源码顺序覆盖其中每一行非空内容。

### 3.1 函数签名与概要

学习副本第 502-528 行，基准第 404-405 行：

```python
def chirp(t, f0, t1, f1, method="linear", phi=0, vertex_zero=True, type="real"):
    """Frequency-swept cosine generator.
```

第一行用 `def` 定义公开函数：`t、f0、t1、f1` 必填，后四项有默认值，冒号开启缩进函数体；参数名 `type` 会遮蔽 Python 内置 `type`。第二行以三双引号开始 docstring，并把函数概括为扫频余弦生成器。

### 3.2 cycles、radians 与自变量单位

学习副本第 530-533 行，基准第 407-410 行：

```python
    In the following, 'Hz' should be interpreted as 'cycles per unit';
    there is no requirement here that the unit is one second.  The
    important distinction is that the units of rotation are cycles, not
    radians. Likewise, `t` could be a measurement of space instead of time.
```

这四行属于同一自然段：频率单位应理解为 cycles per unit，而非限定为 Hz/s；相位旋转用 cycle 表示，进入三角函数前才乘 $2\pi$ 转成 rad。反引号中的 `t` 是文档代码标记；`t` 可以表示时间，也可以表示空间等其他自变量。

### 3.3 参数章节与 `t`

学习副本第 535-538 行，基准第 412-415 行：

```python
    Parameters
    ----------
    t : array_like
        Times at which to evaluate the waveform.
```

`Parameters` 和下划线是 NumPy 风格 docstring 的章节标记，不是执行语句。`t : array_like` 声明文档类型，下一缩进行说明函数在这些位置逐元素求波形。

### 3.4 `f0` 参数

学习副本第 539-540 行，基准第 416-417 行：

```python
    f0 : float
        Frequency (e.g. Hz) at time t=0.
```

`f0 : float` 声明起始频率参数；说明行给出边界条件 $f(0)=f_0$。

### 3.5 `t1` 参数

学习副本第 541-542 行，基准第 418-419 行：

```python
    t1 : float
        Time at which `f1` is specified.
```

`t1` 是指定 `f1` 所在位置的浮点标量，不是自动裁剪输出的终止开关；反引号仅用于文档渲染。

### 3.6 `f1` 参数

学习副本第 543-544 行，基准第 420-421 行：

```python
    f1 : float
        Frequency (e.g. Hz) of the waveform at time `t1`.
```

`f1` 是 $t=t_1$ 时的目标瞬时频率，与 `f0、t1` 一起确定扫频曲线。

### 3.7 `method` 参数

学习副本第 545-547 行，基准第 422-424 行：

```python
    method : {'linear', 'quadratic', 'logarithmic', 'hyperbolic'}, optional
        Kind of frequency sweep.  If not given, `linear` is assumed.  See
        Notes below for more details.
```

花括号列出四个正式 method，`optional` 表示可省略；省略时采用 linear。最后一行提到下方 Notes，但本版本 docstring 实际没有 Notes 章节，属于移植残留。

### 3.8 `phi` 参数

学习副本第 548-549 行，基准第 425-426 行：

```python
    phi : float, optional
        Phase offset, in degrees. Default is 0.
```

`phi` 是可选浮点初相位，API 单位为 degree，默认 0；函数体稍后负责换算为 rad。

### 3.9 `vertex_zero` 参数

学习副本第 550-553 行，基准第 427-430 行：

```python
    vertex_zero : bool, optional
        This parameter is only used when `method` is 'quadratic'.
        It determines whether the vertex of the parabola that is the graph
        of the frequency is at t=0 or t=t1.
```

这组四行共同定义一个布尔控制项：它只在 quadratic method 使用，决定频率—时间抛物线的顶点按契约位于 $t=0$ 还是 $t=t_1$。

### 3.10 `type` 参数

学习副本第 554-555 行，基准第 431-432 行：

```python
    type : {'real', 'complex'}, optional
        Specify output chirp type, only applicable when `method` is 'linear'.
```

`type` 在文档中只允许 real/complex，并且只适用于 linear；它表示输出形式而不是任意 NumPy dtype。

### 3.11 返回值与相位公式

学习副本第 557-563 行，基准第 434-440 行：

```python
    Returns
    -------
    y : ndarray
        A numpy array containing the signal evaluated at `t` with the
        requested time-varying frequency.  More precisely, the function
        returns ``cos(phase + (pi/180)*phi)`` where `phase` is the integral
        (from 0 to `t`) of ``2*pi*f(t)``. ``f(t)`` is defined below.
```

`Returns` 及下划线开启返回值章节，`y : ndarray` 给出抽象返回类型。连续说明行必须合起来读：在每个 `t` 上求时变频率信号，实值输出为 `cos(phase+(pi/180)*phi)`，其中 phase 是 $2\pi f(t)$ 从 0 到 `t` 的积分。实现实际返回 CuPy 数组，原文的 numpy array 是移植遗留；文中还说下方定义 $f(t)$，但本 docstring 并未列出公式。

### 3.12 示例公共导入

学习副本第 565-571 行，基准第 442-448 行：

```python
    Examples
    --------
    The following will be used in the examples:

    >>> from cusignal import chirp, spectrogram
    >>> import matplotlib.pyplot as plt
    >>> import cupy as cp
```

`Examples` 和下划线开启 doctest 示例；说明行指出三条导入供后文共用。`>>>` 是 doctest 主提示符：依次从 cusignal 导入 `chirp/spectrogram`、导入 Matplotlib 为 `plt`、导入 CuPy 为 `cp`。

### 3.13 linear chirp 示例

学习副本第 573-581 行，基准第 450-458 行：

```python
    For the first example, we'll plot the waveform for a linear chirp
    from 6 Hz to 1 Hz over 10 seconds:

    >>> t = cp.linspace(0, 10, 5001)
    >>> w = chirp(t, f0=6, f1=1, t1=10, method='linear')
    >>> plt.plot(cp.asnumpy(t), cp.asnumpy(w))
    >>> plt.title("Linear Chirp, f(0)=6, f(10)=1")
    >>> plt.xlabel('t (sec)')
    >>> plt.show()
```

说明段给出 10 秒内从 6 Hz 下扫到 1 Hz。`cp.linspace` 在闭区间生成 5001 点，间隔 0.002 s；`chirp` 使用关键字参数选择 linear 和默认实值输出；`cp.asnumpy` 把 GPU 数组复制到 CPU 供 Matplotlib 使用。后续三行依次设置标题、横轴并显示图形。

### 3.14 高采样率示例准备

学习副本第 583-589 行，基准第 460-466 行：

```python
    For the remaining examples, we'll use higher frequency ranges,
    and demonstrate the result using `cusignal.spectrogram`.
    We'll use a 10 second interval sampled at 8000 Hz.

    >>> fs = 8000
    >>> T = 10
    >>> t = cp.linspace(0, T, T*fs, endpoint=False)
```

前三行说明余下示例使用更高频率和 spectrogram，并规定 10 秒、8000 Hz。`fs` 与 `T` 是 Python 整数；`T*fs` 得 80000，`endpoint=False` 生成 $[0,T)$，使采样间隔恰为 $1/f_s$。

### 3.15 quadratic chirp 与 spectrogram 示例

学习副本第 591-603 行，基准第 468-480 行：

```python
    Quadratic chirp from 1500 Hz to 250 Hz over 10 seconds
    (vertex of the parabolic curve of the frequency is at t=0):

    >>> w = chirp(t, f0=1500, f1=250, t1=10, method='quadratic')
    >>> ff, tt, Sxx = spectrogram(w, fs=fs, noverlap=256, nperseg=512,
    ...                           nfft=2048)
    >>> plt.pcolormesh(cp.asnumpy(tt), cp.asnumpy(ff[:513]),
                       cp.asnumpy(Sxx[:513]), cmap='gray_r')
    >>> plt.title('Quadratic Chirp, f(0)=1500, f(10)=250')
    >>> plt.xlabel('t (sec)')
    >>> plt.ylabel('Frequency (Hz)')
    >>> plt.grid()
    >>> plt.show()
```

开头两行说明从 1500 Hz 到 250 Hz 的 quadratic chirp，默认顶点在零点。生成 `w` 后，跨两行的 `spectrogram` 调用通过开放括号续接；doctest 的 `...` 是续行提示符。`pcolormesh` 也跨行调用，`[:513]` 选择索引 0～512，`gray_r` 是反转灰度色图。最后五行设置标题、时间轴、频率轴、网格并显示。

### 3.16 关闭 docstring

学习副本第 604 行，基准第 481 行：

```python
    """
```

三双引号关闭 docstring；从下一条非空语句开始才是函数运行时执行的函数体。

## 4. kernel 源码逐行解释

每个 `ElementwiseKernel` 定义是一个完整语义块，不把输入声明、operation 续行和闭合括号拆成孤立小节。

### 4.1 linear real kernel

学习副本第 369-388 行，基准第 321-332 行：

```python
_chirp_phase_lin_kernel_real = cp.ElementwiseKernel(
    "T t, T f0, T t1, T f1, T phi",
    "T phase",
    """
    const T beta { (f1 - f0) / t1 };
    const T temp { 2 * M_PI * (f0 * t + 0.5 * beta * t * t) };
    // Convert  phi to radians.
    phase = cos(temp + phi);
    """,
    "_chirp_phase_lin_kernel",
    options=("-std=c++11",),
)
```

整个块在模块导入时创建一个可复用的 `cp.ElementwiseKernel`。输入字符串声明同一模板类型 T 的 `t、f0、t1、f1、phi`，输出字符串声明 `T phase`，但该变量实际保存余弦样本。三引号内部是 C++ operation：先计算 $\beta=(f_1-f_0)/t_1$，再计算 $2\pi(f_0t+\beta t^2/2)$，最后执行 `cos(temp+phi)`。原注释称此处转换 phi，但转换实际在 Python 函数体完成。末尾依次关闭 operation 字符串、指定 kernel 名称、传入单元素 C++11 选项 tuple，并关闭构造调用。

### 4.2 linear complex kernel

学习副本第 391-409 行，基准第 334-345 行：

```python
_chirp_phase_lin_kernel_cplx = cp.ElementwiseKernel(
    "T t, T f0, T t1, T f1, T phi",
    "Y phase",
    """
    const T beta { (f1 - f0) / t1 };
    const T temp { 2 * M_PI * (f0 * t + 0.5 * beta * t * t) };
    // Convert  phi to radians.
    phase = Y(cos(temp + phi), cos(temp + phi + M_PI/2) * -1);
    """,
    "_chirp_phase_lin_kernel",
    options=("-std=c++11",),
)
```

该块与 real kernel 共享输入和相位公式，但输出模板改为独立的 Y，以便由预分配数组选择 complex64 或 complex128。operation 用 `Y(real,imag)` 构造复数；虚部 `-cos(psi+pi/2)` 等于 `sin(psi)`，所以输出是 $e^{j\psi}$。最后三行分别结束字符串、给出与 real 版本相同的内部名称、指定 C++11 并闭合构造。

### 4.3 quadratic real kernel

学习副本第 412-441 行，基准第 347-365 行：

```python
_chirp_phase_quad_kernel = cp.ElementwiseKernel(
    "T t, T f0, T t1, T f1, T phi, bool vertex_zero",
    "T phase",
    """
    T temp {};
    const T beta { (f1 - f0) / (t1 * t1) };
    if ( vertex_zero ) {
        temp = 2 * M_PI * (f0 * t + beta * (t * t * t) / 3);
    } else {
        temp = 2 * M_PI *
            ( f1 * t + beta *
            ( ( (t1 - t) * (t1 - t) * (t1 - t) ) - (t1 * t1 * t1)) / 3);
    }
    // Convert  phi to radians.
    phase = cos(temp + phi);
    """,
    "_chirp_phase_quad_kernel",
    options=("-std=c++11",),
)
```

输入比 linear 多一个 `bool vertex_zero`，输出仍是实值 T。`temp {}` 值初始化为 0，$\beta=(f_1-f_0)/t_1^2$。真分支计算 $2\pi(f_0t+\beta t^3/3)$；假分支跨三行构造 $2\pi[f_1t+\beta((t_1-t)^3-t_1^3)/3]$，其求导为 $f_1-\beta(t_1-t)^2$，与文档期望的加号不一致。分支闭合后取 `cos(temp+phi)`，再结束字符串、名称、编译选项和构造。

### 4.4 logarithmic real kernel

学习副本第 444-470 行，基准第 367-383 行：

```python
_chirp_phase_log_kernel = cp.ElementwiseKernel(
    "T t, T f0, T t1, T f1, T phi",
    "T phase",
    """
    T temp {};
    if ( f0 == f1 ) {
        temp = 2 * M_PI * f0 * t;
    } else {
        T beta { t1 / log(f1 / f0) };
        temp = 2 * M_PI * beta * f0 * ( pow(f1 / f0, t / t1) - 1.0 );
    }
    // Convert  phi to radians.
    phase = cos(temp + phi);
    """,
    "_chirp_phase_log_kernel",
    options=("-std=c++11",),
)
```

该 kernel 声明五个 T 输入和一个 T 输出。`temp` 先置零；`f0==f1` 时直接使用定频相位，避免 $\log(f_1/f_0)=0$ 引发除零。一般分支令 `beta=t1/log(f1/f0)`，再计算标准几何扫频积分，其中 `pow` 表示幂、减 1 保证零时刻累计相位为零。随后加初相位取余弦，并完成 kernel 名称、C++11 选项和构造闭合。

### 4.5 hyperbolic real kernel

学习副本第 473-499 行，基准第 385-401 行：

```python
_chirp_phase_hyp_kernel = cp.ElementwiseKernel(
    "T t, T f0, T t1, T f1, T phi",
    "T phase",
    """
    T temp {};
    if ( f0 == f1 ) {
        temp = 2 * M_PI * f0 * t;
    } else {
        T sing { -f1 * t1 / (f0 - f1) };
        temp = 2 * M_PI * ( -sing * f0 ) * log( abs( 1 - t / sing ) );
    }
    // Convert  phi to radians.
    phase = cos(temp + phi);
    """,
    "_chirp_phase_hyp_kernel",
    options=("-std=c++11",),
)
```

结构与 log kernel 相同。相等端点退化为定频并避开 `f0-f1=0`；一般分支先求奇点 `sing=-f1*t1/(f0-f1)`，再计算 $2\pi(-t_sf_0)\log|1-t/t_s|$。`abs` 允许对数使用绝对值语义，但 `t=sing` 仍发散。最后取余弦并完成字符串、名称、编译选项与构造闭合。

## 5. 函数体源码逐行解释

函数体按一次完整动作或控制流分支组织；每块原文和解释按实际执行顺序排列。

### 5.1 把输入转换为 CuPy 数组

学习副本第 607 行，基准第 483 行：

```python
    t = cp.asarray(t)
```

`cp.asarray` 把列表、NumPy 数组或 CuPy 数组统一为 CuPy 数组；兼容的现有 CuPy 数组通常不复制。重新赋值后，局部变量 `t` 统一具有设备数组的 shape 和 dtype。

### 5.2 初相位从 degree 转为 rad

学习副本第 610 行，基准第 485 行：

```python
    phi *= np.pi / 180
```

`*=` 先计算右侧乘积再重新绑定 `phi`；乘 $\pi/180$ 把 API 的 degree 转为三角函数所需的 rad，`np.pi` 是主机侧标量常量。

### 5.3 linear real/complex 分支

学习副本第 613-631 行，基准第 487-497 行：

```python
    if method in ["linear", "lin", "li"]:
        if type == "real":
            return _chirp_phase_lin_kernel_real(t, f0, t1, f1, phi)
        elif type == "complex":
            phase = cp.empty(t.shape, dtype=cp.complex64)
            if np.issubclass_(t.dtype, (np.float64)):
                phase = cp.empty(t.shape, dtype=cp.complex128)
            _chirp_phase_lin_kernel_cplx(t, f0, t1, f1, phi, phase)
            return phase
        else:
            raise NotImplementedError("No kernel for type {}".format(type))
```

外层成员测试接受 linear 的全名和两个别名。real 时直接调用自动分配输出的实值 kernel 并返回。complex 时先按 `t.shape` 分配 complex64；`np.issubclass_` 检查 `t.dtype`，其中 `(np.float64)` 没有逗号，并不是 tuple。float64 输入会重新分配 complex128，使前一个 complex64 临时分配失去引用；随后 kernel 显式写入 `phase` 并返回。其他 type 进入 `else`，用 `str.format` 生成消息并抛 `NotImplementedError`。

### 5.4 quadratic 分支

学习副本第 634-636 行，基准第 499-500 行：

```python
    elif method in ["quadratic", "quad", "q"]:
        return _chirp_phase_quad_kernel(t, f0, t1, f1, phi, vertex_zero)
```

`elif` 只在 linear 未匹配时检查 quadratic 全名和两个别名；调用把 `vertex_zero` 作为标量传给 kernel 并立即返回。此分支完全不读取 `type`，所以 `type="complex"` 也仍返回实值。

### 5.5 logarithmic 校验与执行

学习副本第 639-649 行，基准第 502-508 行：

```python
    elif method in ["logarithmic", "log", "lo"]:
        if f0 * f1 <= 0.0:
            raise ValueError(
                "For a logarithmic chirp, f0 and f1 must be "
                "nonzero and have the same sign."
            )
        return _chirp_phase_log_kernel(t, f0, t1, f1, phi)
```

该分支接受全名和两个别名。`f0*f1<=0` 同时覆盖零端点和异号端点，违反实数对数 chirp 定义域时抛 `ValueError`；括号中的两段相邻字符串在编译期自动拼接。合法时调用 logarithmic kernel 并返回。

### 5.6 hyperbolic 校验与执行

学习副本第 652-658 行，基准第 510-513 行：

```python
    elif method in ["hyperbolic", "hyp"]:
        if f0 == 0 or f1 == 0:
            raise ValueError("For a hyperbolic chirp, f0 and f1 must be " "nonzero.")
        return _chirp_phase_hyp_kernel(t, f0, t1, f1, phi)
```

该分支接受全名和 `hyp`。`or` 短路判断任一端点为零并抛 `ValueError`；同一行相邻字符串也会自动拼接。源码没有检查 `t` 是否命中双曲奇点。参数通过后调用 hyperbolic kernel。

### 5.7 非法 method 异常

学习副本第 660-666 行，基准第 515-519 行：

```python
    else:
        raise ValueError(
            "method must be 'linear', 'quadratic', 'logarithmic',"
            " or 'hyperbolic', but a value of %r was given." % method
        )
```

所有合法 method 都未匹配时进入最终 `else`。`raise ValueError(` 跨行构造异常，相邻字符串先拼接，再用旧式 `%r` 把 `repr(method)` 插入消息；最后的右括号闭合异常构造。

## 6. 调用链与算法总结

```text
cusignal.chirp
  └─ chirp(...)
      ├─ cp.asarray(t)
      ├─ phi: degree → rad
      └─ method 分发
          ├─ linear + real    → _chirp_phase_lin_kernel_real
          ├─ linear + complex → 分配 complex 输出 → _chirp_phase_lin_kernel_cplx
          ├─ quadratic        → _chirp_phase_quad_kernel
          ├─ logarithmic      → 定义域检查 → _chirp_phase_log_kernel
          └─ hyperbolic       → 非零检查 → _chirp_phase_hyp_kernel
```

`cp.ElementwiseKernel` 负责广播、确定逐元素迭代规模、生成/缓存 GPU kernel 和启动执行。operation 字符串内没有显式线程索引，因为 CuPy 为每个广播后的元素自动执行一次。这里没有 FFT、插值、卷积或线性代数库调用。

## 7. 数学公式—代码映射

令 $\psi(t)=\theta(t)+\phi$。

| 分支                    | 瞬时频率                                 | 累计相位                                          | 代码位置               |
| ----------------------- | ---------------------------------------- | ------------------------------------------------- | ---------------------- |
| linear                  | $f_0+\beta t$，$\beta=(f_1-f_0)/t_1$ | $2\pi(f_0t+\beta t^2/2)$                        | 基准 325-328、338-341  |
| quadratic true          | $f_0+\beta t^2$                        | $2\pi(f_0t+\beta t^3/3)$                        | 基准 352-354           |
| quadratic false（声明） | $f_1+\beta(t_1-t)^2$                   | $2\pi[f_1t+\beta(t_1^3-(t_1-t)^3)/3]$           | 源码没有此正确符号形式 |
| quadratic false（实际） | 求导得$f_1-\beta(t_1-t)^2$             | $2\pi[f_1t+\beta((t_1-t)^3-t_1^3)/3]$           | 基准 355-358           |
| logarithmic             | $f_0(f_1/f_0)^{t/t_1}$                 | $2\pi t_1f_0[(f_1/f_0)^{t/t_1}-1]/\ln(f_1/f_0)$ | 基准 372-379           |
| hyperbolic              | $f_0f_1t_1/[(f_0-f_1)t+f_1t_1]$        | $2\pi(-t_sf_0)\ln|1-t/t_s|$                     | 基准 390-397           |

linear real 输出 $\cos\psi$；linear complex 输出

$$
\cos\psi+j[-\cos(\psi+\pi/2)]
=\cos\psi+j\sin\psi=e^{j\psi}.
$$

## 8. dtype、shape、广播与内存

- `t` 的 shape 决定普通标量参数下的输出 shape；若其他 kernel 输入可广播，遵守 CuPy 广播规则。
- 实值 kernel 的输入和输出都声明为同一 `T`。实际 dtype 由 CuPy 类型解析决定，不能仅凭 docstring 的 `float` 断言固定为 float64。
- complex 分支明确预分配：`t.dtype == float64` 时用 complex128，其余输入默认 complex64。
- complex float64 路径先分配一次 complex64 再替换为 complex128，产生短暂额外显存分配。
- `cp.empty` 内容未初始化，但 kernel 随后应覆盖每个元素；若 kernel 启动失败，函数会抛错而不是返回可靠内容。
- 返回值是 CuPy 设备数组；docstring 中的 “numpy array” 不符合实际实现。

## 9. 边界、异常与数值稳定性

1. 源码没有检查 `t1==0`：linear/quadratic 会除零，logarithmic 公式也失去端点定义。
2. logarithmic 明确要求 `f0*f1>0`，即非零且同号。
3. hyperbolic 只检查端点非零，没有检查采样位置是否等于奇点 `sing`。
4. `f0==f1` 的 log/hyp 分支退化为定频，避开 $0/0$。
5. 浮点精确比较 `f0==f1` 对计算产生的近似值可能不触发；非常接近时一般公式可能出现消减或大比例因子。
6. 大时间或大频率导致相位很大，三角函数参数约简在 float32 中可能损失精度。
7. quadratic `vertex_zero=False` 存在可由符号求导直接证明的实现不一致；本文没有声称执行测试。
8. `type` 在非 linear 分支被静默忽略；传 `type="complex"` 仍返回实值。
9. 未知 `method` 抛 `ValueError`；linear 下未知 `type` 抛 `NotImplementedError`。

## 10. 复杂度与性能

设广播后的输出元素总数为 $N$：

- 时间复杂度：$O(N)$；每个元素做常数次算术和一个三角函数，对数/双曲分支还含 `log`/`pow`。
- 额外空间：输出为 $O(N)$；除 complex float64 的短暂重复分配外，kernel 内每线程只用常数个局部标量。
- 并行性：元素之间无依赖，适合 GPU；没有归约和同步。
- 性能瓶颈：kernel 启动、小数组启动开销、`pow/log/cos` 特殊函数吞吐、输出显存带宽，以及首次调用的即时编译/缓存成本。
- `cp.asnumpy` 只出现在示例绘图，不在生成函数内；实际算法不会自动把结果复制回主机。

## 11. 建议阅读顺序与自检问题答案

### 11.1 建议阅读顺序

1. 先读公开导出，确认 `cusignal.chirp` 如何出现。
2. 读函数签名和 docstring，先建立 API 契约。
3. 读函数体的数组转换、角度转换和 method/type 分发。
4. 按 linear real → linear complex → quadratic → logarithmic → hyperbolic 阅读 kernel。
5. 每读一个相位表达式都对 $t$ 求导，检查是否还原声明的瞬时频率。
6. 特别手算 quadratic `vertex_zero=False`，不要因为函数名和注释就假定正确。

### 11.2 自检问题

1. 为什么 `phi *= np.pi / 180` 必须发生在 kernel 调用之前？
2. `cp.ElementwiseKernel` 为什么不需要源码显式写线程索引？
3. complex kernel 虚部为什么是正弦，而不是负正弦？
4. `np.issubclass_(t.dtype, (np.float64))` 中的括号是不是 tuple？
5. 为什么 log chirp 用 `f0*f1<=0` 一次覆盖零值和异号？
6. `f0==f1` 时为何必须使用特殊分支？
7. `vertex_zero=False` 实际相位求导后得到哪条频率律？
8. `type="complex"` 配合 quadratic 会发生什么？
9. 输出究竟是 NumPy 还是 CuPy 数组？证据在哪里？
10. 哪些性能开销属于算法，哪些只出现在绘图示例？

### 11.3 参考答案

1. kernel 中只执行 `temp + phi`，没有角度转换；因此 Python 基准第 485 行必须先把 API 接收的 degree 乘 $\pi/180$ 转成 rad，否则 `cos` 会把度数误当弧度。
2. `cp.ElementwiseKernel` 根据广播后的输出规模自动生成逐元素索引和 launch；operation 字符串描述“单个元素做什么”，所以源码无需显式写 `blockIdx/threadIdx`。
3. 令 $\psi=temp+phi$。基准第 341 行写 `-cos(psi+pi/2)`；因 $\cos(\psi+\pi/2)=-\sin\psi$，外层负号使虚部成为 $+\sin\psi$。
4. 不是。`(np.float64)` 没有尾逗号，只是括号表达式，值仍是 `np.float64`。单元素 tuple 必须写成 `(np.float64,)`。
5. `f0*f1<=0` 覆盖三类非法情况：任一端点为零，或两端异号。此时 $f_1/f_0$ 不满足当前实数对数 chirp 的定义域。
6. 当 $f_0=f_1$ 时，log 分支的 $\ln(f_1/f_0)=0$，hyp 分支的 $f_0-f_1=0$；直接套一般式会产生除零。特殊分支返回定频相位 $2\pi f_0t$。
7. 对基准第 356-358 行的相位求导得到
   $$
   f_1-\beta(t_1-t)^2,
   $$

   而非文档声明的加号形式。
8. quadratic 分支不会读取 `type`，直接调用 `_chirp_phase_quad_kernel`，因此即使传 `type="complex"` 仍返回实值数组，也不会报 type 错误。
9. 输出是 CuPy 设备数组。证据是函数先用 `cp.asarray`，real kernel 由 CuPy 自动分配输出，complex 用 `cp.empty` 分配；docstring 的 “numpy array” 是移植遗留。
10. 每元素相位、`cos/pow/log`、GPU kernel 启动、首次即时编译和输出显存访问属于生成算法/实现开销；`cp.asnumpy`、Matplotlib 和 `spectrogram` 只出现在示例，不在 `chirp` 函数执行路径中。
