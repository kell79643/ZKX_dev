# vectorstrength Python 源码算法

> 当前阶段：阶段二（cuSignal 23.08.00 Python 源码算法）  
> 编写日期：2026-08-09  
> 原始摘录：`ZKX/cusignal-23.08.00` 只读基准  
> 学习注释：`Learning/cusignal-23.08.00`  
> 阶段边界：本文不进入 `cusignal_cpp` CPU/GPU 复现

## 1. 代码定位与接口概览

### 1.1 公开调用路径

```text
cusignal.vectorstrength
└─ cusignal.spectral_analysis.spectral.vectorstrength
   ├─ cp.asarray / cp.atleast_2d
   ├─ cp.dot / cp.exp
   └─ cp.mean / cp.abs / cp.angle
```

`vectorstrength` 没有调用 cuSignal 项目内 kernel 或 helper。其数学主体完全写在函数中，底层数组、复指数和归约由 CuPy 执行。

| 层次 | 只读基准位置 | 学习副本位置 | 作用 |
| --- | --- | --- | --- |
| 顶层公开导出 | `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:60–70` | `Learning/cusignal-23.08.00/python/cusignal/__init__.py:73–84` | 提供 `cusignal.vectorstrength` |
| 模块公开导出 | `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/__init__.py:14–24` | `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/__init__.py:14–25` | 提供 `cusignal.spectral_analysis.vectorstrength` |
| 函数定义 | `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1476–1551` | `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1489–1580` | 参数验证、相位向量、复均值及输出 |

函数签名：

```python
vectorstrength(events, period)
```

### 1.2 参数、dtype、shape 与返回值

设事件数为 $N$，候选周期数为 $P$。

| 参数或结果 | 输入形式 | 内部 shape | dtype 语义 | 说明 |
| --- | --- | --- | --- | --- |
| `events` | 标量或一维 array-like；文档契约写一维 | 标量变为 `(1,1)`；一维变为 `(1,N)` | `cp.asarray` 保留/推导输入 dtype；没有显式强制转换 | 事件发生时刻 |
| `period` | 正标量或正的一维 array-like | 标量变为 `(1,1)`；数组变为 `(1,P)` | 同样由 `cp.asarray` 推导 | 必须与 `events` 使用同一时间单位 |
| `vectors` | 内部结果 | `(P,N)` | CuPy 按 `2j * pi / period * events` 做类型提升，结果为复浮点 | 每个周期、每个事件的单位复向量 |
| `vectormean` | 内部结果 | `(P,)` | 复浮点 | 沿事件轴的一阶圆矩 |
| `strength` | 返回值 1 | 标量周期时为 CuPy 标量/零维结果；周期数组时 `(P,)` | `vectormean` 对应的实浮点类型 | 同步强度，理论范围 $[0,1]$ |
| `phase` | 返回值 2 | 同上 | 实浮点 | 首选相位，`cp.angle` 给出弧度主值 |

函数没有 `axis` 参数：所有事件只能放在唯一的一维事件序列中。它也没有权重、显著性水平或高阶谐波参数。

## 2. 当前算子的完整相关源码

以下内容严格摘录只读基准，未包含学习副本中后来插入的注释，也没有使用省略号。当前函数没有 docstring 示例，也没有项目内直接 helper。

### 2.1 顶层公开导出

基准：`ZKX/cusignal-23.08.00/python/cusignal/__init__.py:60`；学习副本对应起点：`Learning/cusignal-23.08.00/python/cusignal/__init__.py:73`。

```python
from cusignal.spectral_analysis.spectral import (
    coherence,
    csd,
    istft,
    lombscargle,
    periodogram,
    spectrogram,
    stft,
    vectorstrength,
    welch,
)
```

### 2.2 `spectral_analysis` 模块公开导出

基准：`ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/__init__.py:14`；学习副本对应起点：`Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/__init__.py:14`。

```python
from cusignal.spectral_analysis.spectral import (
    coherence,
    csd,
    istft,
    lombscargle,
    periodogram,
    spectrogram,
    stft,
    vectorstrength,
    welch,
)
```

### 2.3 `vectorstrength` 完整定义

基准：`ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1476`；学习副本定义：`Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1490`。

```python
def vectorstrength(events, period):
    """
    Determine the vector strength of the events corresponding to the given
    period.

    The vector strength is a measure of phase synchrony, how well the
    timing of the events is synchronized to a single period of a periodic
    signal.

    If multiple periods are used, calculate the vector strength of each.
    This is called the "resonating vector strength".

    Parameters
    ----------
    events : 1D array_like
        An array of time points containing the timing of the events.
    period : float or array_like
        The period of the signal that the events should synchronize to.
        The period is in the same units as `events`.  It can also be an array
        of periods, in which case the outputs are arrays of the same length.

    Returns
    -------
    strength : float or 1D array
        The strength of the synchronization.  1.0 is perfect synchronization
        and 0.0 is no synchronization.  If `period` is an array, this is also
        an array with each element containing the vector strength at the
        corresponding period.
    phase : float or array
        The phase that the events are most strongly synchronized to in radians.
        If `period` is an array, this is also an array with each element
        containing the phase for the corresponding period.

    References
    ----------
    .. [1] van Hemmen, JL, Longtin, A, and Vollmayr, AN. Testing resonating
           vector strength: Auditory system, electric fish, and noise.
           Chaos 21, 047508 (2011);
           :doi:`10.1063/1.3670512`.
    .. [2] van Hemmen, JL. Vector strength after Goldberg, Brown, and
           von Mises: biological and mathematical perspectives.  Biol Cybern.
           2013 Aug;107(4):385-96. :doi:`10.1007/s00422-013-0561-7`.
    .. [3] van Hemmen, JL and Vollmayr, AN.  Resonating vector strength:
           what happens when we vary the "probing" frequency while keeping
           the spike times fixed.  Biol Cybern. 2013 Aug;107(4):491-94.
           :doi:`10.1007/s00422-013-0560-8`.
    """
    events = cp.asarray(events)
    period = cp.asarray(period)
    if events.ndim > 1:
        raise ValueError("events cannot have dimensions more than 1")
    if period.ndim > 1:
        raise ValueError("period cannot have dimensions more than 1")

    # we need to know later if period was originally a scalar
    scalarperiod = not period.ndim

    events = cp.atleast_2d(events)
    period = cp.atleast_2d(period)
    if (period <= 0).any():
        raise ValueError("periods must be positive")

    # this converts the times to vectors
    vectors = cp.exp(cp.dot(2j * cp.pi / period.T, events))

    # the vector strength is just the magnitude of the mean of the vectors
    # the vector phase is the angle of the mean of the vectors
    vectormean = cp.mean(vectors, axis=1)
    strength = cp.abs(vectormean)
    phase = cp.angle(vectormean)

    # if the original period was a scalar, return scalars
    if scalarperiod:
        strength = strength[0]
        phase = phase[0]
    return strength, phase
```

## 3. docstring 逐行翻译与解释

本节覆盖完整 docstring 的每一行非空内容。空行只负责分段，不单列。

#### 函数用途概述

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1491` 至 `:1493`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1477` 至 `:1479`。

```python
    """
    Determine the vector strength of the events corresponding to the given
    period.
```

逐行对应说明：

- 第 1 行：开始函数 docstring；Python 会把紧随函数定义的字符串保存为函数说明。
- 第 2 行：“确定这些事件相对于给定……”；该句在下一行继续，说明计算目标。
- 第 3 行：“…周期的向量强度。”；`period` 是被探测的候选周期，不是由函数自动估计的周期。

#### phase synchrony 定义

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1495` 至 `:1497`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1481` 至 `:1483`。

```python
    The vector strength is a measure of phase synchrony, how well the
    timing of the events is synchronized to a single period of a periodic
    signal.
```

逐行对应说明：

- 第 1 行：“向量强度是相位同步程度的度量”；强调它测时间相位的一致性。
- 第 2 行：“即事件发生时刻与周期信号的某一个周期对齐得有多好”；本行继续定义 phase synchrony。
- 第 3 行：完成上句：“周期信号”。它不要求输入完整连续波形，只需要事件时刻和周期。

#### 多周期与 resonating vector strength

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1499` 至 `:1500`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1485` 至 `:1486`。

```python
    If multiple periods are used, calculate the vector strength of each.
    This is called the "resonating vector strength".
```

逐行对应说明：

- 第 1 行：若传入多个候选周期，就为每个周期分别计算一个向量强度。
- 第 2 行：多周期扫描称为 resonating vector strength；不是把多个周期的结果平均。

#### 输入参数说明

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1502` 至 `:1509`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1488` 至 `:1495`。

```python
    Parameters
    ----------
    events : 1D array_like
        An array of time points containing the timing of the events.
    period : float or array_like
        The period of the signal that the events should synchronize to.
        The period is in the same units as `events`.  It can also be an array
        of periods, in which case the outputs are arrays of the same length.
```

逐行对应说明：

- 第 1 行：NumPy 风格 docstring 的“参数”章节标题。
- 第 2 行：reStructuredText/NumPy docstring 标题分隔线，不参与执行。
- 第 3 行：`events` 应是一维类数组对象；源码实际也接受零维标量。
- 第 4 行：数组每个元素是一个事件发生时刻，而不是均匀采样信号的幅值。
- 第 5 行：`period` 可以是单个浮点周期，也可以是候选周期数组。
- 第 6 行：它表示事件预计与之同步的信号周期。
- 第 7 行：周期与事件时刻必须同单位；同时重申可传数组。
- 第 8 行：若有 $P$ 个周期，两个输出都包含 $P$ 项。

#### strength 与 phase 返回说明

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1511` 至 `:1519`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1497` 至 `:1505`。

```python
    Returns
    -------
    strength : float or 1D array
        The strength of the synchronization.  1.0 is perfect synchronization
        and 0.0 is no synchronization.  If `period` is an array, this is also
        an array with each element containing the vector strength at the
        corresponding period.
    phase : float or array
        The phase that the events are most strongly synchronized to in radians.
```

逐行对应说明：

- 第 1 行：NumPy 风格 docstring 的“返回值”章节标题。
- 第 2 行：返回值标题分隔线，不参与执行。
- 第 3 行：同步强度：标量周期对应标量结果，周期数组对应一维结果。
- 第 4 行：`1.0` 表示所有事件映射后的单位向量完全同向。
- 第 5 行：文档把 `0.0` 概括为无同步；更精确地说是一阶圆矩抵消。周期数组会返回数组。
- 第 6 行：输出数组的每一项保存一个候选周期的强度。
- 第 7 行：完成上句：数组位置与输入周期位置一一对应。
- 第 8 行：第二个返回值是首选相位；shape 与 `strength` 对应。
- 第 9 行：phase 是平均同步向量的方向，单位为弧度。

#### period 数组对应的 phase shape

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1520` 至 `:1521`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1506` 至 `:1507`。

```python
        If `period` is an array, this is also an array with each element
        containing the phase for the corresponding period.
```

逐行对应说明：

- 第 1 行：多周期输入时，每个候选周期都有自己的相位。
- 第 2 行：phase 数组与 period 数组逐项对应。

#### 三项理论参考资料

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1523` 至 `:1534`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1509` 至 `:1520`。

```python
    References
    ----------
    .. [1] van Hemmen, JL, Longtin, A, and Vollmayr, AN. Testing resonating
           vector strength: Auditory system, electric fish, and noise.
           Chaos 21, 047508 (2011);
           :doi:`10.1063/1.3670512`.
    .. [2] van Hemmen, JL. Vector strength after Goldberg, Brown, and
           von Mises: biological and mathematical perspectives.  Biol Cybern.
           2013 Aug;107(4):385-96. :doi:`10.1007/s00422-013-0561-7`.
    .. [3] van Hemmen, JL and Vollmayr, AN.  Resonating vector strength:
           what happens when we vary the "probing" frequency while keeping
           the spike times fixed.  Biol Cybern. 2013 Aug;107(4):491-94.
```

逐行对应说明：

- 第 1 行：参考资料章节标题。
- 第 2 行：参考资料标题分隔线。
- 第 3 行：Sphinx 引文 `[1]` 的首行，列出 resonating vector strength 论文作者和题名前半。
- 第 4 行：完成论文标题，说明研究背景包括听觉系统、电鱼和噪声。
- 第 5 行：给出期刊、卷/文章号与年份。
- 第 6 行：Sphinx DOI 角色，指向第一篇论文的永久标识符。
- 第 7 行：引文 `[2]` 首行，讨论 vector strength 的生物学与数学来源。
- 第 8 行：完成题名并给出期刊缩写 `Biol Cybern.`。
- 第 9 行：给出发表时间、卷期页码和 DOI。
- 第 10 行：引文 `[3]` 首行，主题仍是 resonating vector strength。
- 第 11 行：说明研究操作：固定 spike 时刻，改变探测频率。
- 第 12 行：完成题名并给出期刊、日期和页码。

#### 第三项 DOI 与 docstring 结束符

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1535` 至 `:1536`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1521` 至 `:1522`。

```python
           :doi:`10.1007/s00422-013-0560-8`.
    """
```

逐行对应说明：

- 第 1 行：第三篇参考资料的 DOI。
- 第 2 行：结束 docstring；从下一行起才是可执行函数体。


docstring 没有 `Examples` 章节，因此不存在被省略的示例代码。

## 4. 源代码逐行解释

### 4.1 顶层公开导入逐行解释

#### 公开导入或依赖导入

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/__init__.py:73` 至 `:84`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:60` 至 `:70`。

```python
from cusignal.spectral_analysis.spectral import (
    coherence,
    csd,
    istft,
    lombscargle,
    periodogram,
    spectrogram,
    stft,
    vectorstrength,
    welch,
)
```

逐行对应说明：

- 第 1 行：从 `spectral` 模块导入括号内多个名称；左括号允许跨行书写。
- 第 2 行：导入同模块的 `coherence`；尾逗号表示列表尚未结束。
- 第 3 行：导入 `csd`；它不是 `vectorstrength` 的运行依赖。
- 第 4 行：导入 `istft`；仅是同一公开导入组的兄弟 API。
- 第 5 行：导入 `lombscargle`。
- 第 6 行：导入 `periodogram`。
- 第 7 行：导入 `spectrogram`。
- 第 8 行：导入 `stft`。
- 第 9 行：把目标函数绑定到 `cusignal` 顶层命名空间，从而可调用 `cusignal.vectorstrength`。学习行 81 是新增合法学习注释。
- 第 10 行：导入 `welch`，仍是兄弟 API。
- 第 11 行：关闭多行 import 语句。


### 4.2 `spectral_analysis` 模块导入逐行解释

#### 公开导入或依赖导入

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/__init__.py:14` 至 `:25`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/__init__.py:14` 至 `:24`。

```python
from cusignal.spectral_analysis.spectral import (
    coherence,
    csd,
    istft,
    lombscargle,
    periodogram,
    spectrogram,
    stft,
    vectorstrength,
    welch,
)
```

逐行对应说明：

- 第 1 行：在包初始化文件中从实现模块导入公开名称。
- 第 2 行：导入兄弟 API `coherence`。
- 第 3 行：导入兄弟 API `csd`。
- 第 4 行：导入兄弟 API `istft`。
- 第 5 行：导入兄弟 API `lombscargle`。
- 第 6 行：导入兄弟 API `periodogram`。
- 第 7 行：导入兄弟 API `spectrogram`。
- 第 8 行：导入兄弟 API `stft`。
- 第 9 行：将目标函数暴露为 `cusignal.spectral_analysis.vectorstrength`；学习行 22 是新增注释。
- 第 10 行：导入兄弟 API `welch`。
- 第 11 行：结束该多行 import。


### 4.3 函数签名与函数体逐行解释

docstring 已在上一节逐行覆盖，本表继续覆盖签名、原源码注释和每一行可执行语句。

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1490` 至 `:1490`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1476` 至 `:1476`。

```python
def vectorstrength(events, period):
```

逐行对应说明：

- 第 1 行：`def` 定义函数；两个位置参数都没有默认值；冒号开启缩进函数体。学习行 1489 是函数总览注释。

#### 将输入统一为 CuPy 数组

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1538` 至 `:1540`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1523` 至 `:1524`。

```python
    events = cp.asarray(events)
    period = cp.asarray(period)
```

逐行对应说明：

- 第 1 行：调用 `cupy.asarray` 将输入转成 GPU 数组并重新绑定变量；没有指定 dtype。
- 第 2 行：将周期转换成 CuPy 数组；标量输入成为零维数组。

#### 检查 events 的维数

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1542` 至 `:1543`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1525` 至 `:1526`。

```python
    if events.ndim > 1:
        raise ValueError("events cannot have dimensions more than 1")
```

逐行对应说明：

- 第 1 行：读取数组维数；若超过一维，进入异常分支。零维标量和一维数组都会通过。
- 第 2 行：`raise` 立即终止调用并抛出带说明文字的 `ValueError`。

#### 检查 period 的维数

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1545` 至 `:1546`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1527` 至 `:1528`。

```python
    if period.ndim > 1:
        raise ValueError("period cannot have dimensions more than 1")
```

逐行对应说明：

- 第 1 行：周期只允许零维或一维；不接受二维周期网格。
- 第 2 行：对非法 period 维数抛出 `ValueError`。

#### 保存 period 的原始标量形态

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1548` 至 `:1550`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1530` 至 `:1531`。

```python
    # we need to know later if period was originally a scalar
    scalarperiod = not period.ndim
```

逐行对应说明：

- 第 1 行：原源码注释：后面需要记住 period 最初是否为标量。注释不执行。
- 第 2 行：零维数组的 `ndim` 为 0，`not 0` 得 `True`；一维时得到 `False`。必须在升维前记录。

#### 将 events 和 period 统一为二维行向量

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1553` 至 `:1555`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1533` 至 `:1534`。

```python
    events = cp.atleast_2d(events)
    period = cp.atleast_2d(period)
```

逐行对应说明：

- 第 1 行：至少升为二维：一维事件 `(N,)` 变成行向量 `(1,N)`。
- 第 2 行：标量或一维周期统一成行向量 `(1,P)`，便于下一步转置。

#### 拒绝非正周期

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1557` 至 `:1558`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1535` 至 `:1536`。

```python
    if (period <= 0).any():
        raise ValueError("periods must be positive")
```

逐行对应说明：

- 第 1 行：逐元素比较是否非正，再用 `any()` 判断是否至少存在一个非法周期。括号先完成数组比较。
- 第 2 行：任一周期小于等于零就拒绝整个调用。

#### 把全部“周期—事件”组合映射为单位复向量

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1560` 至 `:1563`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1538` 至 `:1539`。

```python
    # this converts the times to vectors
    vectors = cp.exp(cp.dot(2j * cp.pi / period.T, events))
```

逐行对应说明：

- 第 1 行：原源码注释：下一行把时间转换为复平面单位向量。
- 第 2 行：`period.T` 为 `(P,1)`；`2j*pi/period.T` 与 `(1,N)` 的 `events` 点积得到 `(P,N)` 的 $j2\pi t_i/T_p$；`cp.exp` 逐元素得到单位复向量。

#### 求复均值并分解 strength 与 phase

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1565` 至 `:1572`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1541` 至 `:1545`。

```python
    # the vector strength is just the magnitude of the mean of the vectors
    # the vector phase is the angle of the mean of the vectors
    vectormean = cp.mean(vectors, axis=1)
    strength = cp.abs(vectormean)
    phase = cp.angle(vectormean)
```

逐行对应说明：

- 第 1 行：原注释：向量强度是平均复向量的模。
- 第 2 行：原注释：向量相位是平均复向量的辐角。
- 第 3 行：沿每行的事件维归约；`(P,N)` 变成 `(P,)`，即每个周期一个复均值。
- 第 4 行：逐项取复数模，得到 $R_p=|\bar z_p|$。
- 第 5 行：逐项取复数辐角，得到 $\phi_p=\arg(\bar z_p)$，单位为弧度。

#### 标注标量 period 的返回契约

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1574` 至 `:1574`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1547` 至 `:1547`。

```python
    # if the original period was a scalar, return scalars
```

逐行对应说明：

- 第 1 行：原注释：若最初是标量周期，应恢复标量式返回。

#### 将长度 1 结果拆回标量

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1576` 至 `:1578`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1548` 至 `:1550`。

```python
    if scalarperiod:
        strength = strength[0]
        phase = phase[0]
```

逐行对应说明：

- 第 1 行：根据升维前保存的布尔值选择返回 shape。
- 第 2 行：从 `(1,)` 数组取第 0 项作为标量式 strength。
- 第 3 行：同样取第 0 项作为标量式 phase。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1580` 至 `:1580`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/spectral_analysis/spectral.py:1551` 至 `:1551`。

```python
    return strength, phase
```

逐行对应说明：

- 第 1 行：用逗号构造二元组并结束函数；返回顺序固定为强度在前、相位在后。


## 5. 调用链与执行顺序总结

一次调用严格按以下顺序执行：

1. Python 导入层把同一函数对象暴露到 `cusignal.spectral_analysis` 和 `cusignal` 顶层。
2. `cp.asarray` 把两个输入统一为 CuPy 数组。
3. 维数检查拒绝二维及更高维输入。
4. 在 period 升维前记录它是否原为标量。
5. 两个输入都升为二维行向量；检查所有周期严格为正。
6. 通过 `(P,1) @ (1,N)` 构造全部“周期—事件”组合的复相位矩阵。
7. 沿事件轴求复均值，再取模与辐角。
8. 标量周期拆箱；周期数组保持 `(P,)`；返回二元组。

标量周期 $T$ 时，核心表达式等价于

$$
\bar z=\frac{1}{N}\sum_{i=1}^{N}\exp\left(j\frac{2\pi t_i}{T}\right),
\qquad
R=|\bar z|,
\qquad
\phi=\arg\bar z.
$$

周期数组时只是把同一公式对 $p=1,\ldots,P$ 矩阵化，没有采用另一套数学算法。

## 6. 数学公式与代码逐项映射

| 数学对象 | 代码表达式 | 基准位置 | 说明 |
| --- | --- | --- | --- |
| 候选角频率 $\omega_p=2\pi/T_p$ | `2j * cp.pi / period.T` 去掉 `1j` 后的实系数 | `spectral.py:1539` | 转置产生列向量，`1j` 同时加入复指数所需虚单位 |
| 相位 $\theta_{pi}=2\pi t_i/T_p$ | 点积的实系数与 `events` 外积 | `spectral.py:1539` | 没有显式 `% T`；复指数周期性隐式折叠相位 |
| 单位向量 $z_{pi}=e^{j\theta_{pi}}$ | `cp.exp` 作用于完整点积结果 | `spectral.py:1539` | 每项理论模长为 1 |
| 一阶圆矩 $\bar z_p=N^{-1}\sum_i z_{pi}$ | `cp.mean(vectors, axis=1)` | `spectral.py:1543` | 沿事件维平均 |
| vector strength $R_p=|\bar z_p|$ | `cp.abs(vectormean)` | `spectral.py:1544` | 理论范围 $[0,1]$ |
| 首选相位 $\phi_p=\operatorname{atan2}(\Im\bar z_p,\Re\bar z_p)$ | `cp.angle(vectormean)` | `spectral.py:1545` | 采用正指数约定 |
| resonating vector strength | period 数组生成 `(P,N)` 矩阵 | `spectral.py:1533–1545` | 对每个候选周期独立计算 |

与 Fourier 负指数约定 $e^{-j2\pi ft}$ 相比，本实现使用正指数。两者的复结果互为共轭，因此 strength 相同，phase 符号相反。

## 7. CuPy 语法与底层机制

### 7.1 `cp.asarray` 与 `cp.atleast_2d`

`cp.asarray(x)` 将 Python 标量、列表、NumPy 数组或 CuPy 数组转换为 CuPy 数组。函数没有显式 `dtype=`，因此结果类型服从输入和 CuPy 的类型推导。`cp.atleast_2d` 不改变已有二维以上数组，但这些输入已在前面的 ndim 检查中被拒绝；对一维数组，它在最前面增加长度为 1 的轴。

### 7.2 `.T`、`cp.dot` 与外积

升维后：

```text
period       (1, P)
period.T     (P, 1)
events       (1, N)
dot result   (P, N)
```

这里的矩阵乘法数值上等价于外积。第 $p$ 行保存同一周期 $T_p$ 对所有事件的相位参数；第 $i$ 列保存同一事件在所有候选周期下的参数。

### 7.3 `cp.exp`、`cp.mean`、`cp.abs` 与 `cp.angle`

- `cp.exp` 是逐元素复指数运算，不是 FFT。
- `cp.mean(vectors, axis=1)` 是 GPU 归约，事件顺序不影响数学结果。
- `cp.abs` 对复数计算 $\sqrt{x^2+y^2}$。
- `cp.angle` 等价于复数虚部与实部的 `atan2`，输出弧度主值。

函数不显式同步 CUDA stream，也不把结果复制回 CPU；返回值仍在 GPU 侧。调用者若立即访问主机数值，才可能触发后续同步或数据传输。

## 8. 边界条件、异常与数值稳定性

| 情况 | 源码行为 | 数学或使用含义 |
| --- | --- | --- |
| `events.ndim > 1` | 抛 `ValueError` | 不支持批量事件矩阵 |
| `period.ndim > 1` | 抛 `ValueError` | 不支持二维候选周期网格 |
| 任一 `period <= 0` | 抛 `ValueError` | 周期必须严格为正 |
| `events` 为标量 | 接受并升为 `(1,1)` | 单事件总有理论 $R=1$，但不构成显著性证据 |
| `events` 为空 | 没有显式拒绝 | `mean` 的数学分母 $N=0$，结果未定义；调用者应避免 |
| `period` 为空数组 | 没有显式拒绝 | 返回空的周期结果没有分析意义 |
| 所有相位相同 | `strength` 接近或等于 1 | 完全相位锁定 |
| 复均值为零 | `strength` 为 0，仍调用 `cp.angle` | phase 数学上未定义；浮点返回值不应解释 |
| 很大的 $|t_i/T|$ | 直接计算大相位后取复指数 | 参数约简可能损失低位相位精度；源码未先做模周期 |
| 大量近抵消向量 | 直接浮点归约 | 舍入误差可能使理论零得到极小非零值，并令 phase 不稳定 |

另外，NaN/Inf 和复数输入没有专门语义检查。`period <= 0` 对 NaN 不会把它识别为非正；复数周期的有序比较可能不合法。接口在物理上应使用有限实数时刻和有限正实周期。

该实现没有执行：

- Rayleigh 显著性检验或 $p$ 值计算；
- 有限样本偏差修正；
- 权重归一化；
- 二阶或更高阶圆矩；
- 周期峰值搜索、插值、FFT 或分箱；
- 自动相位解缠。

## 9. 时间复杂度、空间复杂度与性能瓶颈

令候选周期数为 $P$、事件数为 $N$：

- 相位矩阵和复指数：时间 $O(PN)$；
- 沿事件轴求均值：时间 $O(PN)$；
- 取模和相角：时间 $O(P)$；
- `vectors` 主存储：空间 $O(PN)$；
- 输出：空间 $O(P)$。

标量周期时 $P=1$，时间和临时空间均为 $O(N)$。多周期模式的主要瓶颈不是最终归约，而是一次物化整个 `(P,N)` 复矩阵：当 $P$ 和 $N$ 都很大时，GPU 显存及复指数吞吐会成为限制。当前实现没有分块，也没有利用递推相位或 FFT 加速规则周期网格。

## 10. 现有测试证据与覆盖缺口

直接测试位于 `ZKX/cusignal-23.08.00/python/cusignal/test/test_spectral_analysis.py:319–346`。测试执行以下检查：

1. 生成 $N=2^4=16$ 个时间事件；
2. 分别使用标量周期 `0.75` 和 `5`；
3. CPU 参考调用 `scipy.signal.vectorstrength`；
4. GPU 调用 `cusignal.vectorstrength`；
5. 同步默认 CUDA stream，并用 `array_equal` 比较两个返回结果。

这证明测试意图是核对标量周期正常路径与 SciPy 的一致性，但当前片段没有覆盖：period 数组、事件标量、空输入、非法维度、非正周期、NaN/Inf、极大相位、近完全抵消以及不同 dtype。本文没有运行本地或 ZQ500 测试；根据学习任务约束，本阶段只做源码阅读和文档核对。

## 11. 建议阅读顺序与检查问题

建议按以下顺序亲自阅读：

1. 先看 `spectral.py:1476–1522` 的签名和 docstring，确认输入是事件时刻而非幅值序列。
2. 看 `spectral.py:1523–1536`，追踪标量/一维输入如何校验并统一 shape。
3. 单独手算 `spectral.py:1539` 的 shape：为什么 `(P,1) @ (1,N)` 得到 `(P,N)`？
4. 看 `spectral.py:1543–1545`，把 `mean → abs/angle` 对应到一阶圆矩。
5. 看 `spectral.py:1547–1551`，确认 scalar period 与 period 数组的返回 shape 差异。
6. 最后回到两个 `__init__.py`，理解导出层只改变访问路径，不参与核心计算。
7. 查看测试 `test_spectral_analysis.py:319–346`，区分“与 SciPy 一致”已经覆盖的情况和未覆盖边界。

阅读完成后应能回答：

1. 为什么源码不需要显式执行 `events % period`？
2. `scalarperiod` 为什么必须在 `cp.atleast_2d(period)` 之前保存？
3. 标量、单元素数组和多元素数组的返回 shape 有何区别？
4. `cp.dot` 在这里为什么实际上构造的是外积？
5. `axis=1` 为什么是事件轴而不是周期轴？
6. 为什么 `cp.abs` 和 `cp.angle` 必须作用于 `vectormean`，不能先分别作用于每个事件向量再平均？
7. 为什么 `strength=0` 时仍可能存在双峰周期结构？
8. 当前实现为何是 $O(PN)$ 时间和 $O(PN)$ 临时空间？
9. 顶层导出、数组工程处理和数学核心分别对应哪些行？
10. 现有测试为什么不能证明周期数组和异常路径正确？

### 11.1 参考答案

1. **复指数本身具有周期性。**若 $t=qT+r$，则 $e^{j2\pi t/T}=e^{j2\pi q}e^{j2\pi r/T}=e^{j2\pi r/T}$。因此基准 `spectral.py:1539` 的 `cp.exp` 已隐式完成相位折叠。
2. **升维会丢失“原来是否为标量”的信息。**零维 period 的 `ndim` 为 0，故基准 `spectral.py:1531` 先保存 `scalarperiod=True`；执行 `cp.atleast_2d` 后标量和单元素一维数组都会呈 `(1,1)`，无法再区分。
3. **标量和数组的 marker 不同。**标量 period 最终对 `(1,)` 的 strength/phase 取 `[0]`；单元素数组仍返回长度 1 数组；多元素数组返回长度 $P$ 数组。证据为基准 `spectral.py:1547–1551`。
4. **两个操作数都是二维行/列向量。**`period.T` 为 `(P,1)`，`events` 为 `(1,N)`，所以 `cp.dot` 按矩阵乘法生成 `(P,N)`；每项恰是 $(2j\pi/T_p)t_i$，数值上就是外积。
5. **`vectors` 的第二维存事件。**其 shape 为 `(P,N)`，第 0 维索引 period，第 1 维索引 event，所以基准 `spectral.py:1543` 的 `axis=1` 对每个 period 汇总全部事件。
6. **必须先做向量相消再分解。**若先对每个单位向量取模，所有模都为 1，平均后永远得到 1；若先取角再普通平均，又会破坏圆周拓扑。基准 `spectral.py:1543–1545` 正确地先求复均值，再取 `abs` 和 `angle`。
7. **一阶圆矩可能被对称多峰抵消。**例如相位 0 与 $\pi$ 等权时复向量和为 0，但仍有严格双峰周期结构；所以 `strength=0` 只说明当前一阶分量抵消。
8. **每个 period 都要处理每个 event。**相位矩阵有 $PN$ 项，构造、复指数与求和均需 $O(PN)$ 工作；源码还显式物化 `(P,N)` 的 `vectors`，故主要临时空间也是 $O(PN)$。
9. **三层职责可由位置区分。**两个 `__init__.py` 的 import 只暴露名称；基准 `spectral.py:1523–1536,1547–1550` 负责数组转换、验证和返回 shape；`:1539,1543–1545` 才是相位映射、圆均值、模与角的数学核心。
10. **测试只覆盖标量正常路径。**基准 `test_spectral_analysis.py:319–346` 只用 16 个事件和标量周期 `0.75`、`5` 对照 SciPy，没有调用 period 数组，也没有触发空事件、非法维数、非正周期或数值极端分支。

## 12. 阶段二完整性结论

- 已完整摘录两个必要公开导入语句和 `vectorstrength` 的完整定义、完整 docstring、全部函数体；docstring 本身没有示例。
- 摘录中的每一行非空内容都已按原顺序逐行解释；空行保持在源码摘录中。
- 没有读取或摘录同文件其他算子的实现，也没有把 CuPy 第三方内部源码纳入学习范围。
- 学习副本只新增独立的 `# <学习注释：说明文本>` 行；没有修改原代码、原注释、空行、缩进或顺序。
- 剥离全部合法学习注释后，三个相关学习文件与 `ZKX/cusignal-23.08.00` 对应只读基准逐字符一致。
