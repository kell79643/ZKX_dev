# `fm_demod` Python 源码算法

## 1. 代码定位与接口概览

### 1.1 公开导入路径

`fm_demod` 有两条公开访问路径：

```python
import cusignal
cusignal.fm_demod(x, axis=-1)

import cusignal.demod
cusignal.demod.fm_demod(x, axis=-1)
```

准确定位如下：

| 角色 | 学习副本位置 | 只读基准位置 | 符号 |
| --- | --- | --- | --- |
| 顶层公开导出 | `Learning/cusignal-23.08.00/python/cusignal/__init__.py:33` | `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:30` | `fm_demod` |
| 子包公开导出 | `Learning/cusignal-23.08.00/python/cusignal/demod/__init__.py:15` | `ZKX/cusignal-23.08.00/python/cusignal/demod/__init__.py:14` | `fm_demod` |
| CuPy 导入 | `Learning/cusignal-23.08.00/python/cusignal/demod/demod.py:15` | `ZKX/cusignal-23.08.00/python/cusignal/demod/demod.py:14` | `cp` |
| 函数定义 | `Learning/cusignal-23.08.00/python/cusignal/demod/demod.py:19-46` | `ZKX/cusignal-23.08.00/python/cusignal/demod/demod.py:17-38` | `fm_demod` |

学习副本行号比只读基准大，是因为学习副本插入了合法的 `<学习注释：...>` 独立注释行；原始源码没有被改写。

### 1.2 函数签名

```python
fm_demod(x, axis=-1)
```

| 项目 | 语义 |
| --- | --- |
| `x` | 单个复信号或一批复信号。函数先用 `cp.asarray` 转成 CuPy 数组。实数 dtype 被拒绝。 |
| `axis` | 执行相位展开与相邻差分的轴；默认 `-1`，即最后一轴。原 docstring 没有记录这个参数。 |
| 返回值 `y` | 展开相位沿 `axis` 的一阶差分，单位为 `rad/sample`。若 `x.shape[axis]=N`，则 `y.shape[axis]=max(N-1,0)`，其他轴不变。 |

若输入沿目标轴的相位为 $\theta_u[n]$，实际返回

$$
y[n]=\theta_u[n+1]-\theta_u[n].
$$

源码没有 `fs`、载频 `fc` 或频率灵敏度 `kf` 参数，所以它不会自动换算成 Hz，也不会自动恢复带绝对标度的消息。换算瞬时频率还需在调用方计算

$$
f_i[n]=\frac{f_s}{2\pi}y[n].
$$

## 2. 当前算子的完整相关源码

以下摘录严格使用 `ZKX/cusignal-23.08.00` 只读基准的原文，没有加入学习注释，也没有省略当前算子的 docstring、函数体或必要公开导出语句。

### 2.1 顶层公开导出

来源：`ZKX/cusignal-23.08.00/python/cusignal/__init__.py:30`

```python
from cusignal.demod.demod import fm_demod
```

### 2.2 `demod` 子包公开导出

来源：`ZKX/cusignal-23.08.00/python/cusignal/demod/__init__.py:14`

```python
from cusignal.demod.demod import fm_demod
```

### 2.3 必要模块导入与完整函数定义

来源：`ZKX/cusignal-23.08.00/python/cusignal/demod/demod.py:14-38`

```python
import cupy as cp


def fm_demod(x, axis=-1):
    """
    Demodulate Frequency Modulated Signal

    Parameters
    ----------
    x : ndarray
        Received complex valued signal or batch of signals

    Returns
    -------
    y : ndarray
        The demodulated output with the same shape as `x`.
    """

    x = cp.asarray(x)

    if cp.isrealobj(x):
        raise AssertionError("Input signal must be complex-valued")
    x_angle = cp.unwrap(cp.angle(x), axis=axis)
    y = cp.diff(x_angle, axis=axis)
    return y
```

`cp.asarray`、`cp.isrealobj`、`cp.angle`、`cp.unwrap` 和 `cp.diff` 都是第三方 CuPy API，不是 cuSignal 项目内 helper；因此没有需要继续摘录的本项目 kernel 或 helper。该函数也没有自定义 CUDA kernel 字符串。

## 3. docstring 逐行翻译与解释

本节逐行覆盖基准函数 docstring 中每一行非空内容。基准行号对应 `ZKX/cusignal-23.08.00/python/cusignal/demod/demod.py`；学习副本行号对应插入学习注释后的文件。

#### 文档字符串

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/demod/demod.py:20` 至 `:21`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/demod/demod.py:18` 至 `:19`。

```python
    """
    Demodulate Frequency Modulated Signal
```

逐行对应说明：

- 第 1 行：三个双引号开始函数 docstring。缩进表明这个字符串属于 `fm_demod` 函数体；Python 会把它保存为函数的 `__doc__`。
- 第 2 行：“解调频率调制信号”。这是功能摘要，但没有说明输出只是未缩放的相位增量。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/demod/demod.py:23` 至 `:26`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/demod/demod.py:21` 至 `:24`。

```python
    Parameters
    ----------
    x : ndarray
        Received complex valued signal or batch of signals
```

逐行对应说明：

- 第 1 行：NumPy 风格 docstring 的参数章节标题。
- 第 2 行：标题下划线，是文档格式标记，不参与程序运算。
- 第 3 行：参数 `x` 应是数组。这里只写 `ndarray`，函数实际上还会用 `cp.asarray` 接受可转换的 array-like 输入。
- 第 4 行：“接收到的复值信号或一批信号”。复值意味着每个样本通常承载 I/Q 分量，可由辐角获得相位。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/demod/demod.py:28` 至 `:32`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/demod/demod.py:26` 至 `:30`。

```python
    Returns
    -------
    y : ndarray
        The demodulated output with the same shape as `x`.
    """
```

逐行对应说明：

- 第 1 行：返回值章节标题。注意 docstring 没有单列 `axis` 参数，这是接口文档遗漏。
- 第 2 行：返回值标题下划线，仅用于格式。
- 第 3 行：返回数组名为 `y`。源码实际返回 CuPy 运算产生的数组。
- 第 4 行：“与 `x` 形状相同的解调输出”。这与实现不一致：第 37 行调用 `cp.diff`，会令目标轴长度减少 1。
- 第 5 行：结束 docstring；下一条可执行语句从 `cp.asarray` 开始。


原 docstring 没有示例，因此不存在需要摘录或逐行解释的 docstring 示例代码。

## 4. 源代码逐行解释

本节覆盖必要导入、两条公开导出以及函数定义和函数体的每一行非空内容。docstring 已在上一节逐行解释，这里不重复其内部文字，但仍说明函数定义与执行代码。

### 4.1 必要导入

#### 公开导入或依赖导入

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/demod/demod.py:15` 至 `:15`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/demod/demod.py:14` 至 `:14`。

```python
import cupy as cp
```

逐行对应说明：

- 第 1 行：`import` 加载 CuPy 包；`as cp` 建立短别名。此后 `cp.asarray` 等名字都从 CuPy 命名空间查找。CuPy 数组和运算面向 GPU，这是工程执行后端，不改变鉴频公式。


### 4.2 公开导出

#### 公开导入或依赖导入

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/__init__.py:34` 至 `:34`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:30` 至 `:30`。

```python
from cusignal.demod.demod import fm_demod
```

逐行对应说明：

- 第 1 行：绝对导入语句：从 `cusignal.demod.demod` 模块取出 `fm_demod` 并绑定到 `cusignal` 顶层包，使 `cusignal.fm_demod` 可用。它只改变 API 可见性，不执行解调。

#### 公开导入或依赖导入

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/demod/__init__.py:15` 至 `:15`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/demod/__init__.py:14` 至 `:14`。

```python
from cusignal.demod.demod import fm_demod
```

逐行对应说明：

- 第 1 行：同一个符号被绑定到 `cusignal.demod` 子包，使 `cusignal.demod.fm_demod` 可用。它也不是核心算法。


### 4.3 函数定义与函数体

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/demod/demod.py:19` 至 `:19`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/demod/demod.py:17` 至 `:17`。

```python
def fm_demod(x, axis=-1):
```

逐行对应说明：

- 第 1 行：`def` 创建函数对象；`x` 是必填位置或关键字参数；`axis=-1` 给出默认值，负轴索引 `-1` 指最后一维；冒号开始缩进函数体。定义函数时不会执行函数体。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/demod/demod.py:35` 至 `:35`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/demod/demod.py:32` 至 `:32`。

```python
    x = cp.asarray(x)
```

逐行对应说明：

- 第 1 行：先求右侧：调用 CuPy 的 `asarray` 把输入变成 CuPy 数组；再把结果重新绑定给局部变量 `x`。这使后续 API 接收到统一数组类型。若输入来自 CPU，转换可能伴随主机到设备的数据搬运；是否复制取决于输入类型、dtype 和兼容性。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/demod/demod.py:38` 至 `:44`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/demod/demod.py:34` 至 `:37`。

```python
    if cp.isrealobj(x):
        raise AssertionError("Input signal must be complex-valued")
    x_angle = cp.unwrap(cp.angle(x), axis=axis)
    y = cp.diff(x_angle, axis=axis)
```

逐行对应说明：

- 第 1 行：调用 `isrealobj` 判断数组对象的 dtype 是否为实数类型。若条件为真，执行下一行缩进块；复数 dtype 即使虚部数值全为零，通常仍属于复数对象，不进入该分支。
- 第 2 行：构造带英文消息的 `AssertionError` 并用 `raise` 抛出，函数立即终止。这里使用显式异常语句，不受 Python `-O` 删除 `assert` 语句的影响。
- 第 3 行：Python 先计算内层 `cp.angle(x)`，得到每个复样本的主值相角；再把结果传给 `cp.unwrap`，沿指定轴修正被判断为相位包裹的跳变；最后绑定到 `x_angle`。同一个 `axis` 以关键字参数传入。
- 第 4 行：沿同一轴计算一阶相邻差分。若展开相位为 $\theta_u[n]$，则输出为 $\theta_u[n+1]-\theta_u[n]$。目标轴长度减少 1，结果代表相邻采样区间的平均相位增量。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/demod/demod.py:46` 至 `:46`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/demod/demod.py:38` 至 `:38`。

```python
    return y
```

逐行对应说明：

- 第 1 行：把局部变量 `y` 的引用返回给调用者并结束函数。没有后续缩放、滤波或边界填充。


## 5. 调用链与按执行顺序拆解

### 5.1 调用链

```text
cusignal.fm_demod / cusignal.demod.fm_demod
    → cusignal.demod.demod.fm_demod
        → cp.asarray
        → cp.isrealobj
        → cp.angle
        → cp.unwrap
        → cp.diff
        → return
```

没有项目内 helper、Python 循环或自定义 kernel。GPU 上的具体 kernel 启动由 CuPy 内部实现管理，本阶段不把第三方 CuPy 内部源码扩展为学习对象。

### 5.2 执行顺序

1. **数组统一**：`cp.asarray(x)` 把输入置于 CuPy 数组语境。
2. **输入门禁**：若数组 dtype 是实数类型，抛出异常。
3. **取包裹相位**：对每个复数 $a+jb$ 求 $\operatorname{atan2}(b,a)$ 所表示的主值辐角。
4. **相位展开**：沿 `axis` 检查相邻相位跳变，通过补减 $2\pi$ 消除被认为由主值区间造成的跳变。
5. **一阶差分**：沿同一轴做相邻元素相减。
6. **返回**：输出相位增量，不做频率标定。

## 6. 数学公式与代码逐项映射

设复输入为

$$
x[n]=A[n]e^{j\theta[n]}.
$$

| 数学步骤 | 公式 | 代码 |
| --- | --- | --- |
| 主值相角 | $\theta_w[n]=\operatorname{Arg}(x[n])$ | `cp.angle(x)` |
| 相位展开 | $\theta_u[n]=\theta_w[n]+2\pi k[n]$，其中整数 $k[n]$ 用于消除包裹跳变 | `cp.unwrap(..., axis=axis)` |
| 前向差分 | $y[n]=\theta_u[n+1]-\theta_u[n]$ | `cp.diff(x_angle, axis=axis)` |
| Hz 换算 | $f_i[n]=f_s y[n]/(2\pi)$ | **源码未实现，需调用方完成** |
| 消除载频 | $f_i[n]-f_c$ | **源码未实现** |
| 恢复消息标度 | $m[n]=(f_i[n]-f_c)/k_f$ | **源码未实现** |

一阶差分近似的是

$$
\frac{d\theta}{dt}\bigg|_{n+1/2}
\approx f_s\left(\theta_u[n+1]-\theta_u[n]\right),
$$

所以返回值的位置更自然地理解为相邻样本的中点，而不是原输入每个采样时刻。源码没有生成中点时间轴。

## 7. shape、dtype 与批处理语义

### 7.1 shape

若

```text
x.shape = (D0, D1, ..., Daxis=N, ..., Dk)
```

则

```text
y.shape = (D0, D1, ..., max(N-1, 0), ..., Dk)
```

例如，`x.shape == (8, 4096)`：

- `axis=-1` 时，8 路信号分别沿时间轴解调，输出形状为 `(8, 4095)`；
- `axis=0` 时，算法沿批次维相减，输出形状为 `(7, 4096)`。后者只有当第 0 轴确实代表连续采样时才有物理意义。

### 7.2 dtype

- 输入必须具有复数 dtype；检查的是对象类型语义，不是“当前所有虚部是否恰好为零”。
- `cp.angle` 产生实数相位数组；后续 `unwrap` 和 `diff` 也处理实数相位。
- 精确输出 dtype 由当前 CuPy 版本及输入 dtype 的类型提升规则决定；源码没有显式 `astype`，因此不能从本函数本身承诺固定为 `float32` 或 `float64`。

## 8. 边界、异常与数值稳定性

### 8.1 已实现的检查

- 实数 dtype：抛出 `AssertionError("Input signal must be complex-valued")`。
- 非数组输入：先交给 `cp.asarray`；能否转换及异常类型由 CuPy 决定。
- 非法 `axis`：由 `cp.unwrap` 或 `cp.diff` 抛出轴越界异常，本函数不自行包装。

### 8.2 未显式处理的情况

- **零幅度或近零幅度**：此处相位不稳定，噪声可导致很大的相位误差。
- **单样本相位变化过大**：`unwrap` 无法在缺少先验时区分真实快速旋转和 $2\pi$ 包裹；采样混叠不会被修复。
- **NaN/Inf**：本函数没有清洗，传播行为由 CuPy 运算决定。
- **残余载频**：表现为输出直流偏置，不会自动扣除。
- **差分噪声放大**：一阶差分具有高通特性，相位噪声会被强化；函数没有平滑或低通滤波。
- **目标轴过短**：长度 0 或 1 时，差分结果在该轴长度为 0。
- **输出对齐**：函数不补首项或末项，因此不保持原 shape。

### 8.3 docstring 与实现不一致

docstring 基准第 29 行声称输出“with the same shape as `x`”，但基准第 37 行的 `cp.diff` 必然令目标轴减少一个元素。学习和调用时应以代码事实为准。

## 9. 复杂度、内存与性能瓶颈

设输入总元素数为 $M$，目标轴长度为 $N$。

| 项目 | 复杂度或开销 |
| --- | --- |
| `cp.asarray` | 已是兼容 CuPy 数组时可能为 $O(1)$ 视图/复用；若需转换或从主机搬运，则约为 $O(M)$ 数据量。 |
| `cp.isrealobj` | 主要检查 dtype 元数据，通常近似 $O(1)$。 |
| `cp.angle` | 对每个元素求相角，时间 $O(M)$，生成相位临时数组。 |
| `cp.unwrap` | 沿目标轴检查差分并累计 $2\pi$ 修正，时间 $O(M)$，需要中间数组；具体数量由 CuPy 实现决定。 |
| `cp.diff` | 相邻相减，时间 $O(M)$，输出约含 $M-M/N$ 个元素（规则批次情形）。 |
| 总体 | 渐近时间 $O(M)$，额外设备内存 $O(M)$。 |

可能的性能瓶颈：

1. CPU array-like 输入经 `cp.asarray` 搬到 GPU；对很短信号，搬运与 kernel 启动开销可能超过计算收益。
2. `angle`、`unwrap`、`diff` 是多个独立数组步骤，会产生中间结果和多次设备内存读写。
3. `unwrap` 沿指定轴包含相邻差分与累计修正，比单纯逐元素运算更复杂。
4. 函数没有融合成单个自定义 kernel；这属于工程实现选择，不改变数学算法。

## 10. 与阶段一通用原理的对应

| 阶段一内容 | Python 落实位置 | 关系 |
| --- | --- | --- |
| 原理 1：瞬时相位求导鉴频 | `demod.py:36-37`（学习副本 `:42-44`） | 直接采用：主值相角、展开相位、前向差分。 |
| 原理 2：相邻样本共轭乘积 | 无显式对应语句 | 在正常相位增量条件下与当前差分形式数学等价，但源码没有写共轭乘积。 |
| 原理 3：频率—幅度转换鉴频 | 无 | 未采用。 |
| 原理 4：PLL 跟踪鉴频 | 无 | 未采用。 |

## 11. 建议阅读顺序与观察问题

1. 先看两条 `__init__.py` 导出语句：区分“函数如何被找到”和“函数如何计算”。
2. 看签名 `def fm_demod(x, axis=-1)`：确认批处理数组的真正时间轴是哪一维。
3. 看 `cp.asarray` 与 `cp.isrealobj`：区分 GPU 工程接入和数学输入前提。
4. 单独手算 `cp.angle`：例如 $1,j,-1,-j$ 的主值相角分别是什么？
5. 构造跨越 $\pi$ 的相位序列，思考 `unwrap` 为什么需要补减 $2\pi$。
6. 手算 `cp.diff` 后核对长度为什么从 $N$ 变为 $N-1$。
7. 最后检查缺失的物理标定：为什么没有 `fs` 就不可能输出 Hz？为什么没有 `fc`、`kf` 就不一定等于消息？

## 12. 阅读检查题

1. `axis=-1` 在二维批量输入中通常代表哪一维？如果时间轴不是最后一维怎么办？
2. `cp.isrealobj` 检查的是 dtype 还是每个元素当前的虚部数值？
3. `cp.angle` 的结果为什么需要 `unwrap`？
4. `unwrap` 能否恢复每采样跨越多个完整周期的真实相位？为什么？
5. `cp.diff` 对 shape 和时间对齐分别有什么影响？
6. 怎样把返回值从 `rad/sample` 换算成 Hz？
7. 顶层导出语句为什么不是算子核心计算？
8. docstring 中哪一项与代码事实冲突？
9. 当前函数为什么不需要摘录任何 cuSignal 自定义 helper 或 CUDA kernel？
10. 哪些步骤是 $O(M)$，主要临时数组来自哪里？

### 参考答案

1. `axis=-1` 指最后一维，二维批量通常把它作为每路信号的时间轴；若时间轴在别处，应显式传入对应正轴或负轴编号。依据是基准 `demod.py:17,36-37` 将同一个 `axis` 交给 `unwrap` 和 `diff`。
2. 检查 dtype/数组对象是否为实数类型，不逐元素判断虚部。依据是基准 `demod.py:34` 调用 `cp.isrealobj(x)`。
3. `cp.angle` 返回主值相位，跨越 $\pm\pi$ 会出现假的近 $2\pi$ 跳变；基准 `:36` 先 `unwrap` 再交给差分。
4. 不能。若真实单步相位跨越一个或多个完整周期，离散样本缺少唯一判定应补多少个 $2\pi$ 的信息，`unwrap` 不能突破采样混叠。
5. 基准 `:37` 的 `cp.diff` 让目标轴从 $N$ 变为 $\max(N-1,0)$；每个结果对应相邻输入区间，时间位置自然在两采样点中间。
6. 计算 $f_i[n]=f_s y[n]/(2\pi)$。若要恢复消息，还需减载频并除以频率灵敏度。
7. 两条 `__init__.py` 导入只把同一函数绑定到公开命名空间，不计算 angle、unwrap 或 diff；核心计算在基准 `demod.py:32-38`。
8. 基准 docstring `demod.py:29` 声称输出与 `x` 同 shape，但实现 `:37` 的 `cp.diff` 会缩短目标轴。
9. 函数体只调用第三方 CuPy API，没有调用 cuSignal 项目内 helper、kernel 或下一层符号；因此当前算子边界内没有其他项目源码需要摘录。
10. `cp.asarray` 在需要转换/搬运时、`cp.angle`、`cp.unwrap`、`cp.diff` 都是 $O(M)$ 量级；主要临时数组是包裹相位、展开相位及差分输出，具体复用由 CuPy 内部决定。

