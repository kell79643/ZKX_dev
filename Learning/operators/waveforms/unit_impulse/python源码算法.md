# `unit_impulse` Python 源码算法

## 1. 本阶段范围与完整性基线

本阶段只阅读并注释 `unit_impulse` 的公开导出入口、函数定义和它直接调用的 `_unit_impulse_kernel`；为核对 docstring 契约，额外只读查看了本算子的直接测试 `test_waveforms.py:168-190`，没有扩展阅读其他算子、其余测试树或 `cusignal_cpp`。

开始注释前已逐行比较以下学习副本与只读基准：

| 学习副本                                                              | 只读基准                                                         | 比较结果                                                                            |
| --------------------------------------------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `Learning/cusignal-23.08.00/python/cusignal/__init__.py`            | `ZKX/cusignal-23.08.00/python/cusignal/__init__.py`            | 忽略合法学习注释后完全一致                                                          |
| `Learning/cusignal-23.08.00/python/cusignal/waveforms/__init__.py`  | `ZKX/cusignal-23.08.00/python/cusignal/waveforms/__init__.py`  | 忽略合法学习注释后完全一致                                                          |
| `Learning/cusignal-23.08.00/python/cusignal/waveforms/waveforms.py` | `ZKX/cusignal-23.08.00/python/cusignal/waveforms/waveforms.py` | 忽略既有`square` 的合法 `#`/`// <学习注释：...>` 后，602 个原始行逐行完全一致 |

本阶段只向 `Learning/cusignal-23.08.00` 插入新的独立学习注释，没有修改 `ZKX/cusignal-23.08.00`。

## 2. 公开导入路径与代码定位

用户通常通过以下路径调用：

```python
import cusignal
y = cusignal.unit_impulse(8)
```

公开导入链为：

```text
cusignal.unit_impulse
  → cusignal/__init__.py 导入 unit_impulse
  → cusignal.waveforms.waveforms.unit_impulse
  → _unit_impulse_kernel
```

双行号定位如下；“学习行号”包含新增学习注释，“基准行号”对应未加学习注释的只读原文件。

| 层次             | 共享根目录`ZKX_dev/` 下的定位                                       | 符号                         | 学习行号 | 基准行号 |
| ---------------- | --------------------------------------------------------------------- | ---------------------------- | -------: | -------: |
| 顶层公开入口     | `Learning/cusignal-23.08.00/python/cusignal/__init__.py`            | `unit_impulse` 导入项      |       85 |       83 |
| waveforms 包入口 | `Learning/cusignal-23.08.00/python/cusignal/waveforms/__init__.py`  | `unit_impulse` 导入项      |       21 |       19 |
| 第三方依赖导入   | `Learning/cusignal-23.08.00/python/cusignal/waveforms/waveforms.py` | `cp`、`np`               |    14-17 |    14-15 |
| GPU kernel 对象  | 同上                                                                  | `_unit_impulse_kernel`     |  569-588 |  522-534 |
| Python API       | 同上                                                                  | `unit_impulse`             |  591-680 |  537-602 |
| 实际控制流       | 同上                                                                  | `shape`/`idx` 处理与返回 |  665-680 |  593-602 |

导入文件不做核心计算，只负责把同一个函数对象暴露到更方便的命名空间。核心实现只在 `waveforms.py` 中。

## 3. 函数签名与接口语义

基准源码第 537 行、学习副本第 591 行：

```python
def unit_impulse(shape, idx=None, dtype=float):
```

### 3.1 Python 语法

- `def` 声明函数；
- `unit_impulse` 是函数名；
- `shape` 没有默认值，是必传参数；
- `idx=None` 表示不传 `idx` 时使用对象 `None`；
- `dtype=float` 表示签名宣称默认数据类型为 Python `float`；
- 末尾冒号 `:` 开始缩进函数体；
- 源码没有类型注解，参数约束来自 docstring 和实际执行行为。

### 3.2 参数的“文档语义”与“实际语义”

| 参数      | docstring 声称的语义               | 当前源码实际使用方式                      | 结论                       |
| --------- | ---------------------------------- | ----------------------------------------- | -------------------------- |
| `shape` | `int` 或 N-D shape tuple         | 经`np.atleast_1d` 后只读取 `shape[0]` | 实际输出长度只由第一维决定 |
| `idx`   | `None`、标量、tuple 或 `"mid"` | 会先正规化，但最终只读取`idx[0]`        | 实际只使用第一维目标索引   |
| `dtype` | 控制输出数据类型，默认`float64`  | 函数体完全没有读取`dtype`               | 当前参数无效               |

### 3.3 返回值、dtype 与 shape

kernel 输出声明固定为：

```python
"float64 out"
```

因此当前实现返回 CuPy `ndarray`，元素类型固定为 `float64`。调用时传入 `dtype=np.int8` 等值不会改变 kernel 声明。

最后调用为：

```python
return _unit_impulse_kernel(idx[0], size=shape[0])
```

由于没有数组输入或预分配的多维输出，`size=shape[0]` 只指定一维执行范围和输出元素数。因此可由源码确认的返回 shape 是：

$$
(\texttt{shape[0]},).
$$

例如 `shape=(3, 3)` 时，当前代码只请求 `size=3`，并不构造文档示例所示的 $3\times3$ 数组。这是 docstring 与实现之间的明确差异。

### 3.4 docstring 逐段解释

学习副本第 592-664 行包含新增解释注释和原始 docstring；只读基准原始 docstring 位于第 538-592 行。

#### 概要

```text
Unit impulse signal (discrete delta function) or unit basis vector.
```

中文含义是“单位冲激信号（离散 delta 函数）或单位基向量”。前半句采用信号处理视角，后半句采用线性代数视角；长度为 $N$、索引 $k$ 为 `1` 的数组同时就是 $\delta[n-k]$ 和 $\mathbf e_k$。

#### NumPy 风格章节语法

```text
Parameters
----------
Returns
-------
Notes
-----
Examples
--------
```

这些标题和下划线属于 NumPy 风格 docstring 标记，供 Sphinx 等文档工具识别，不是 Python 可执行语句。`参数名 : 类型` 后面的缩进行用于说明该参数。

#### `shape`

```text
shape : int or tuple of int
```

文档意图是：整数表示一维输出样本数，例如 `8`；整数 tuple 表示 N 维 shape，例如 `(3, 4)`。但当前实现正规化后只把 `shape[0]` 作为 `size`，所以 N-D 是未落实的接口说明。

#### `idx`

```text
idx : None or int or tuple of int or 'mid', optional
```

`optional` 表示可省略，省略时使用签名默认值 `None`。文档约定：

- `None`：目标为第 0 个元素；
- `"mid"`：每一维取 `shape // 2`；
- 标量整数：广播到所有维度；
- tuple：逐维指定目标坐标。

这里的“broadcasting”不是 kernel 对数组做普通广播，而是文档意图中的“把标量索引复制到每一维”，对应 Python 代码 `(idx,) * len(shape)`。当前返回语句仍只使用复制结果的首项。

#### `dtype`

```text
dtype : data-type, optional
```

文档声称可选择输出元素类型，例如 `numpy.int8` 是 8 位有符号整数，默认 `numpy.float64` 是 64 位浮点数。但函数体不读取 `dtype`，kernel 直接声明 `"float64 out"`，所以这段说明描述的能力没有在当前实现中落实。

#### 返回值

```text
y : ndarray
```

文档把返回对象命名为 `y`，类型写作通用的 `ndarray`。在 cuSignal/CuPy 语境下，预期是驻留在 GPU 侧的 CuPy `ndarray`。当前函数实际直接返回 `_unit_impulse_kernel(...)` 的结果，没有在 Python 层创建名为 `y` 的局部变量。

#### Notes

```text
The 1D case is also known as the Kronecker delta.
```

一维情况下，输出满足

$$
y[n]=\delta[n-k]=
\begin{cases}
1,&n=k,\\
0,&n\ne k.
\end{cases}
$$

因此它就是 Kronecker delta 的有限数组表示。

#### Sphinx 数学语法

```text
:math:`\\delta[n]`
```

`:math:`...`` 是 Sphinx/reStructuredText 的行内数学角色。源码处于 Python 三引号字符串中，所以写成 `\\delta`，由 Python 字符串转义后保留 LaTeX 所需的 `\delta`。它渲染为 $\delta[n]$。

#### 示例一：默认索引

```python
cusignal.unit_impulse(8)
```

`idx` 使用默认值 `None`，随后转换成 `(0,)`。当前代码调用 kernel 的等价参数是 `idx=0, size=8`，所以一维输出应在索引 0 为 `1`。该示例与当前函数体一致。

示例中的：

```python
import cupy as cp
```

没有被后续示例语句直接引用，可能只是强调 cuSignal 使用 CuPy/GPU 数组；它不是调用 `cusignal.unit_impulse` 的必要语句。

#### 示例二：移位两个样本

```python
cusignal.unit_impulse(7, 2)
```

标量 `2` 没有 `__iter__`，因此被转换成 `(2,)`。kernel 接收 `idx=2, size=7`，索引 2 写 `1`，对应 $\delta[n-2]$。该示例也与当前函数体一致。

#### 示例三：二维居中

```python
cusignal.unit_impulse((3, 3), "mid")
```

按 docstring 意图：`shape // 2` 得到 `(1,1)`，应生成中心坐标 `(1,1)` 为 `1` 的 $3\times3$ 数组。但当前调用只使用 `idx[0]=1` 和 `shape[0]=3`，没有第二维计算，因此示例输出不能由当前函数体推出。

#### 示例四：标量索引广播到二维

```python
cusignal.unit_impulse((4, 4), 2)
```

参数正规化确实会先得到 `idx=(2,2)`，符合文档所说的广播意图；但返回语句只使用第一个 `2` 和第一个 shape 值 `4`。因此文档展示的 $4\times4$ 输出同样没有在当前实现中落实。

总结：前两个一维示例与代码一致；两个二维示例及 `dtype` 说明只代表 docstring 所述接口，和当前函数体存在明确差异。

## 4. 从公开 API 到最终计算的调用链

```text
用户调用 cusignal.unit_impulse(shape, idx, dtype)
  ↓ Python 导入绑定
waveforms.py::unit_impulse
  ↓ np.atleast_1d(shape)
shape 至少变成一维 NumPy 数组
  ↓ idx 分支
None → 每维 0
"mid" → 每维 shape // 2
标量 → 复制到每一维
iterable → 原样保留
  ↓ 仅取第一项
idx[0] 与 shape[0]
  ↓ CuPy ElementwiseKernel 调用
_unit_impulse_kernel(idx[0], size=shape[0])
  ↓ 每个线性索引 i
i != idx → out = 0
i == idx → out = 1
  ↓
返回一维 float64 CuPy 数组
```

这个调用链没有 FFT、卷积、插值、递推或线性代数库调用。唯一核心计算是索引比较。

## 5. `_unit_impulse_kernel` 定义逐句解释

### 5.1 创建 `ElementwiseKernel`

学习副本第 569-570 行，基准第 522 行：

```python
_unit_impulse_kernel = cp.ElementwiseKernel(
```

- `cp` 是 `cupy` 的别名；
- `ElementwiseKernel` 创建一个逐元素 GPU kernel 对象；
- `_unit_impulse_kernel` 前导下划线表示模块内部实现约定，不是公开 API；
- 该语句在模块导入期间执行一次，得到可复用的 kernel 对象；具体设备代码通常在首次匹配参数调用时编译并缓存。

[CuPy 官方 `ElementwiseKernel` 文档](https://docs.cupy.dev/en/v14.0.0a1/reference/generated/cupy.ElementwiseKernel.html)说明，定义包含输入参数、输出参数、operation 和 kernel 名称；调用时 `size` 指定索引范围大小，特殊变量 `i` 表示循环内当前索引。

### 5.2 输入参数声明

学习副本第 571-572 行，基准第 523 行：

```python
"int32 idx",
```

这是传给 CuPy 的参数描述字符串，不是 Python 类型注解。它声明：

- 设备端变量名为 `idx`；
- 设备端类型为 32 位有符号整数 `int32`；
- 传入的 Python/NumPy 标量需要由 CuPy 按此声明转换；
- 函数没有在 Python 层验证 `idx` 是否为整数、是否溢出或是否在输出范围内。

### 5.3 输出参数声明

学习副本第 573-574 行，基准第 524 行：

```python
"float64 out",
```

它声明输出变量 `out` 的每个元素是 64 位浮点数。这个声明比 Python 签名中的 `dtype` 参数更接近最终执行，因此可以直接判定当前输出固定为 `float64`。

### 5.4 operation 字符串

学习副本第 575-583 行，基准第 525-531 行：

```cpp
if (i != idx) {
    out = 0;
} else {
    out = 1;
}
```

外层 Python 三引号把这段 C/C++ 风格代码保存为普通字符串，CuPy 随后把它嵌入生成的设备 kernel 中。

语法逐项如下：

- `if (条件) { ... } else { ... }` 是 C/C++ 条件分支；
- `i` 是 `ElementwiseKernel` 提供的当前线性元素索引，不是 Python 局部变量；
- `!=` 表示不等于；
- `=` 是赋值，不是数学等号；
- 每条赋值语句以分号 `;` 结束；
- 花括号 `{}` 限定分支代码块。

它逐项实现：

$$
y[i]=
\begin{cases}
0,&i\ne k,\\
1,&i=k,
\end{cases}
$$

其中设备端 `idx` 对应数学目标索引 $k$，`out` 对应 $y[i]$。

### 5.5 kernel 名称与编译选项

学习副本第 584-587 行，基准第 532-533 行：

```python
"_unit_impulse_kernel",
options=("-std=c++11",),
```

- 名称字符串用于标识生成的 kernel；
- `options=` 是关键字参数；
- `("-std=c++11",)` 是只有一个元素的 Python tuple，末尾逗号不能省略；
- `-std=c++11` 请求后端按 C++11 语言标准编译；
- 这些属于编译与缓存配置，不改变 Kronecker delta 的数学含义。

## 6. `unit_impulse` 按执行顺序解析

### 6.1 `shape` 正规化

学习副本第 665-666 行，基准第 593 行：

```python
shape = np.atleast_1d(shape)
```

`np.atleast_1d` 保证输入至少具有一个维度：

```text
8       → array([8])
(3, 4)  → array([3, 4])
```

这里使用的是 CPU 侧 NumPy，只处理很小的 shape 元数据；最终信号并不是 NumPy 数组。

赋值号右边先执行，再把结果重新绑定给局部变量 `shape`。因此从这一行之后，源码假设 `shape` 支持：

- `len(shape)`；
- `shape // 2`；
- `shape[0]`。

### 6.2 `idx is None` 分支

学习副本第 668-670 行，基准第 595-596 行：

```python
if idx is None:
    idx = (0,) * len(shape)
```

语法和逻辑：

- `is None` 检查对象身份，是判断缺省值的标准 Python 写法；
- `(0,)` 是单元素 tuple；若写成 `(0)`，结果只是整数 `0`；
- tuple 乘整数表示重复；
- `len(shape)` 在这里代表文档语义上的维数。

例如：

```text
shape=array([8])    → len(shape)=1 → idx=(0,)
shape=array([3,4])  → len(shape)=2 → idx=(0,0)
```

不过最终只读取 `idx[0]`，因此第二个及以后索引不会进入 kernel。

### 6.3 `idx == "mid"` 分支

学习副本第 671-673 行，基准第 597-598 行：

```python
elif idx == "mid":
    idx = tuple(shape // 2)
```

- `elif` 表示只有前一个 `if` 条件为假时才检查该条件；
- `==` 比较值是否相等；
- `//` 是向下取整除法，并对 NumPy 数组逐元素执行；
- `tuple(...)` 把结果数组转换为 Python tuple。

例如：

```text
shape=array([7])    → shape // 2=array([3])   → idx=(3,)
shape=array([3,4])  → shape // 2=array([1,2]) → idx=(1,2)
```

数学对应为：

$$
k_r=\left\lfloor\frac{N_r}{2}\right\rfloor.
$$

当前实现最终只使用 $k_0$。

### 6.4 标量 `idx` 分支

学习副本第 674-676 行，基准第 599-600 行：

```python
elif not hasattr(idx, "__iter__"):
    idx = (idx,) * len(shape)
```

- `hasattr(obj, name)` 检查对象是否具有指定属性；
- `__iter__` 是 Python 迭代协议方法；
- `not` 对布尔结果取反；
- 没有 `__iter__` 的对象被当作标量；
- 标量随后被复制成与 `shape` 元素数相同的 tuple。

例如：

```text
shape=array([4,4]), idx=2 → idx=(2,2)
```

这只是启发式的“是否可迭代”判断，不是严格的整数或索引类型验证。字符串本身可迭代，因此除 `"mid"` 外的字符串不会进入这个分支。

### 6.5 iterable `idx` 的隐式分支

代码没有写最后的 `else`。如果 `idx`：

- 不是 `None`；
- 不等于 `"mid"`；
- 具有 `__iter__`；

那么它保持原样。例如 tuple `(2, 3)` 会直接保留。随后仍只读取 `idx[0]`。

源码没有验证：

- iterable 是否为空；
- tuple 长度是否等于 shape 维数；
- 每项是否为整数；
- 每项是否在合法范围内。

### 6.6 调用 kernel 并返回

学习副本第 678-680 行，基准第 602 行：

```python
return _unit_impulse_kernel(idx[0], size=shape[0])
```

执行顺序是：

1. 计算 `idx[0]`；
2. 计算 `shape[0]`；
3. 以第一个值作为 kernel 输入参数 `idx`；
4. 以第二个值作为关键字参数 `size`；
5. CuPy 为 $i=0,1,\ldots,\texttt{size}-1$ 执行 operation；
6. 得到一维 `float64` CuPy 数组；
7. `return` 立即把数组返回给调用者并结束函数。

`size` 必不可少，因为这里只有标量输入 `idx`，没有数组输入可供 CuPy 通过广播推断输出长度。

### 6.7 直接核对 CuPy 13.6.0 的 `size` 定义

为避免只根据调用表面推测，本阶段进一步读取了当前工作区固定的 CuPy 13.6.0 直接依赖：

```text
cupy-13.6.0/cupy/_core/_kernel.pyx
```

关键执行链为：

1. `_kernel.pyx:850` 从关键字参数取出 `size`；
2. `_kernel.pyx:891-894` 在显式提供 `size` 时执行 `shape.assign(1, size)`，其中第一个参数 `1` 表示 shape 只有一个维度；
3. `_kernel.pyx:896-897` 把该 shape 交给 `_get_out_args_with_params`；
4. `_kernel.pyx:674-684` 在调用者没有提供输出数组时，按 `out_shape` 分配新的 `cupy.ndarray`；
5. `_kernel.pyx:900-903` 在只有一个输出参数且未要求 tuple 时，直接返回这个数组；
6. `_kernel.pyx:921-923` 以 `indexer.size` 启动 kernel，并返回前面创建的结果。

所以对本函数调用

```python
_unit_impulse_kernel(idx[0], size=shape[0])
```

CuPy 13.6.0 的直接定义给出：

$$
\text{out shape}=(\texttt{shape[0]},).
$$

这项结论现在有 CuPy 实现源码支撑，不再只是根据参数名推断。它仍属于静态源码证据，不等于已在当前 ZQ500 Python 环境中成功执行示例。

## 7. 数学公式与代码逐项映射

| 数学对象                         | Python/CuPy 代码                     | 学习副本位置             | 基准位置                 |
| -------------------------------- | ------------------------------------ | ------------------------ | ------------------------ |
| 有限序列长度$N$                | `shape[0]` 传给 `size`           | `waveforms.py:680`     | `waveforms.py:602`     |
| 目标索引$k$                    | `idx[0]`                           | `waveforms.py:680`     | `waveforms.py:602`     |
| 当前索引$i$                    | `ElementwiseKernel` 特殊变量 `i` | `waveforms.py:576-581` | `waveforms.py:526-530` |
| $i\ne k$ 时 $y[i]=0$         | `if (i != idx) { out = 0; }`       | `waveforms.py:577-579` | `waveforms.py:526-528` |
| $i=k$ 时 $y[i]=1$            | `else { out = 1; }`                | `waveforms.py:579-582` | `waveforms.py:528-530` |
| 默认$k=0$                      | `idx = (0,) * len(shape)`          | `waveforms.py:669-670` | `waveforms.py:595-596` |
| 中点$k_r=\lfloor N_r/2\rfloor$ | `idx = tuple(shape // 2)`          | `waveforms.py:672-673` | `waveforms.py:597-598` |

kernel 直接生成 Kronecker delta，没有借助第一个文档中的阶跃差分、卷积或傅里叶变换。那些原理解释冲激的性质和用途，不是该构造函数的内部步骤。

## 8. 边界处理、异常与类型风险

当前函数没有显式的 `raise`、范围判断或 dtype 处理。可由控制流直接确认的情况如下：

| 输入情况              | 源码路径                           | 可确认的结果或风险                                        |
| --------------------- | ---------------------------------- | --------------------------------------------------------- |
| `idx=None`          | 构造全 0 tuple，最终取首项         | 目标索引为 0                                              |
| `idx="mid"`         | `shape // 2` 后取首项            | 一维时目标为`shape[0] // 2`                             |
| `idx<0`             | 设备端没有任何非负`i` 与之相等   | 对通常正 size 输出全零                                    |
| `idx>=shape[0]`     | 同上                               | 输出全零，而非本函数主动报越界错误                        |
| 空 iterable`idx=()` | 访问`idx[0]`                     | Python 层会发生索引错误                                   |
| N-D`shape`/`idx`  | 只读取`[0]`                      | 丢弃其他维度，返回一维结果                                |
| 用户传入`dtype`     | 函数体不读取                       | 输出仍由`"float64 out"` 决定                            |
| 非整数标量`idx`     | 传入声明为`int32` 的 kernel 参数 | 转换行为交给 CuPy；本函数没有验证，不应依赖隐式截断或转换 |
| 非法 shape            | 先经 NumPy、再作为`size`         | 具体异常由 NumPy/CuPy 触发；本函数没有给出自己的错误消息  |

`idx` 的设备类型是 `int32`，所以极大索引还涉及 32 位范围限制。源码没有检查该限制；不能把超大数组索引支持写成既成事实。

## 9. 数值稳定性与精度

核心 operation 只进行整数索引比较并写入 `0` 或 `1`：

- 没有浮点累加；
- 没有除法、指数或三角函数；
- `0.0` 和 `1.0` 可由 IEEE `float64` 精确表示；
- 因此合法范围内的单位冲激构造本身没有舍入误差。

可能的问题来自接口和类型转换，而不是数值公式：

- `idx` 被声明为 `int32`；
- `dtype` 被忽略；
- 多维参数被截成第一维；
- 越界不会在 Python 层主动拒绝。

## 10. 复杂度、内存与性能

令实际输出长度为：

$$
N=\texttt{shape[0]}.
$$

### 10.1 时间复杂度

每个输出元素执行一次比较和一次写入，总工作量为：

$$
O(N).
$$

GPU 让多个位置并行执行，但不会减少必须写入的 $N$ 个结果，因此总工作复杂度仍是 $O(N)$。

### 10.2 空间复杂度

返回数组包含 $N$ 个 `float64` 元素，占用约：

$$
8N\ \text{bytes}.
$$

除输出外，host 侧 `shape` 和 `idx` 元数据规模与文档维数 $d$ 成正比，辅助空间约为 $O(d)$。

### 10.3 可能的性能瓶颈

- 首次调用可能包含 CuPy JIT 编译与 kernel 缓存开销；
- 对很短数组，GPU kernel 启动开销可能大于实际比较计算；
- 对大数组，计算极少，主要成本是分配并写入 `float64` 输出，容易受内存带宽影响；
- 固定 `float64` 会比 `int8` 或 `bool` 占用更多输出内存，但当前 `dtype` 参数不能改变这一点；
- host 侧 `np.atleast_1d` 只处理少量元数据，通常不是主要瓶颈。

本阶段没有在本地或 ZQ500 上运行性能测试，因此以上是由代码结构得到的复杂度与瓶颈分析，不是实测结论。

## 11. docstring 示例与实际代码的差异

docstring 中的一维示例与核心代码一致：

```python
unit_impulse(8)
unit_impulse(7, 2)
```

但以下两类说明与当前实现不一致：

1. `unit_impulse((3, 3), "mid")` 被展示成 $3\times3$ 数组；实际返回调用只使用 `size=shape[0]` 和 `idx[0]`。
2. `dtype` 被描述为可选输出类型；实际 kernel 输出固定声明为 `float64`，函数体不读取 `dtype`。

阅读源码时必须保留这一不一致。不能为了匹配 docstring 而假设存在源码中没有出现的 reshape、多维坐标比较或 dtype 分派。

### 11.1 2026-08-03 ZQ500 实操检查记录

本次按标准五阶段流程进入 `gpu_02`：

- 本地 ZKX 提交：`c1c37114639676266a08a251a065b0a7928a32be`，并如实保留无关 dirty 改动；
- 同步成功标志：远程路径 `/home/usr_02/ZKX_dev` 和 `Sync done`；
- 容器目录：`/tmp/ZKX_dev`；
- 设备：`ZQ500-Q QUAD-2`；
- SDK 初始化后 CMake：3.22.1；
- base Python：3.12.11；
- 当时错误执行了 `python -m pip show cupy`，它查询的是发行包名 `cupy`，不能用于否定发行包 `cupy-cuda11x` 所提供的同名 Python 导入模块；
- 用户随后提供的容器截图显示：`cupy-cuda11x` 已安装在 `/root/.local/lib/python3.12/site-packages`，版本为 13.6.0，同时已有 NumPy 2.4.6 和 fastlock 0.8.3；
- 工作区的 `cupy-13.6.0` 目录仍是可用于直接核对实现的源码副本，它与 site-packages 中已安装的发行包是两个不同位置。

因此此前“容器没有 CuPy”的结论是错误的，根因是把 Python 导入名和 pip 发行包名混为一谈。随后重新进入干净的持久交互会话，实际确认：

```text
python=3.12.11
cupy=13.6.0
```

但是 CuPy 对设备的实际探测返回：

```text
CUDARuntimeError: cudaErrorNoDevice: no CUDA-capable device is detected
```

同一容器中的 `dlsmi` 可以识别 `ZQ500-Q QUAD-2`，但已安装的通用 `cupy-cuda11x` 通过其 CUDA runtime 后端没有识别到该设备。因此“CuPy 包已安装”成立，“这份 NVIDIA CUDA CuPy 能在 ZQ500 上执行 kernel”不成立。

实际运行 docstring 四个示例，并额外传入一次 `dtype=cp.int8`，结果如下：

| 探针         | 调用                                           | 实际状态              | 失败位置                                                                        |
| ------------ | ---------------------------------------------- | --------------------- | ------------------------------------------------------------------------------- |
| 默认索引     | `cusignal.unit_impulse(8)`                   | `cudaErrorNoDevice` | `_unit_impulse_kernel` 调用进入 `ElementwiseKernel.__call__` 后获取设备失败 |
| 偏移索引     | `cusignal.unit_impulse(7, 2)`                | `cudaErrorNoDevice` | 同上                                                                            |
| 二维居中     | `cusignal.unit_impulse((3, 3), "mid")`       | `cudaErrorNoDevice` | 同上                                                                            |
| 二维标量广播 | `cusignal.unit_impulse((4, 4), 2)`           | `cudaErrorNoDevice` | 同上                                                                            |
| dtype 探针   | `cusignal.unit_impulse(5, 1, dtype=cp.int8)` | `cudaErrorNoDevice` | 同上                                                                            |

traceback 显示 cuSignal 调用到只读基准 `waveforms.py:602`：

```python
return _unit_impulse_kernel(idx[0], size=shape[0])
```

随后在已安装 CuPy 的 `_kernel.pyx:868` 获取 CUDA device 时失败。失败发生在输出数组和 kernel 结果可供检查之前，所以本次没有得到任何案例的 `repr(y)`、`y.shape` 或 `y.dtype`；不能用失败运行声称二维示例实际返回了一维，也不能声称 docstring 二维输出运行成功。

当前证据边界是：

- CuPy 13.6.0 静态实现明确把显式 `size` 设成一维 shape，因此源码语义支持“一维输出”判断；
- 当前 ZQ500 容器里的通用 `cupy-cuda11x` 无法识别 ZQ500，因而不能完成运行时结果验证；
- 若要获得成功运行证据，需要平台兼容的 CuPy 后端，而不是仅确认通用 wheel 已安装。

### 11.2 怎样判断 docstring 还是当前 cuSignal 实现正确

这里的“正确”需要分成两个层次：

1. **接口契约是否正确**：`shape`、`idx`、`dtype` 按文档应产生什么结果；
2. **当前实现是否遵守契约**：cuSignal 23.08.00 的函数体实际把什么参数传给 kernel。

#### 证据一：SciPy 实际执行验证 docstring 契约

在同一 `gpu_02` 容器中用 SciPy 1.17.1 运行四个 docstring 示例，并增加一次 dtype 探针，实际结果为：

| 调用                                         | 实际 shape | 实际 dtype  | 关键结果                       |
| -------------------------------------------- | ---------- | ----------- | ------------------------------ |
| `signal.unit_impulse(8)`                   | `(8,)`   | `float64` | 索引 0 为`1`                 |
| `signal.unit_impulse(7, 2)`                | `(7,)`   | `float64` | 索引 2 为`1`                 |
| `signal.unit_impulse((3, 3), "mid")`       | `(3, 3)` | `float64` | 坐标`(1,1)` 为 `1`         |
| `signal.unit_impulse((4, 4), 2)`           | `(4, 4)` | `float64` | 标量`2` 广播为坐标 `(2,2)` |
| `signal.unit_impulse(5, 1, dtype=np.int8)` | `(5,)`   | `int8`    | 值为`[0,1,0,0,0]`            |

这证明 docstring 描述的是一套可以实际成立的 SciPy 接口契约，不是数学上自相矛盾的说明。

#### 证据二：无 GPU 参数捕获验证当前 cuSignal Python 控制流

由于通用 `cupy-cuda11x` 不能在 ZQ500 上启动 kernel，本次只在测试进程内把模块全局变量 `_unit_impulse_kernel` 临时替换为记录 `args` 和 `kwargs` 的普通 Python 函数。没有修改源码文件，也没有模拟 kernel 输出；目的仅是观察 `unit_impulse` 最后一行到底传了什么。

同一组输入的实际捕获结果为：

| 调用                                  | 传给`_unit_impulse_kernel` 的位置参数 | 关键字参数           |
| ------------------------------------- | --------------------------------------- | -------------------- |
| `unit_impulse(8)`                   | `(0,)`                                | `size=np.int64(8)` |
| `unit_impulse(7, 2)`                | `(2,)`                                | `size=np.int64(7)` |
| `unit_impulse((3, 3), "mid")`       | `(np.int64(1),)`                      | `size=np.int64(3)` |
| `unit_impulse((4, 4), 2)`           | `(2,)`                                | `size=np.int64(4)` |
| `unit_impulse(5, 1, dtype=cp.int8)` | `(1,)`                                | `size=np.int64(5)` |

因此可执行的 Python 控制流证明确实只有 `idx[0]` 和 `shape[0]` 到达 kernel；第二维和 `dtype` 均未传递。

#### 证据三：CuPy 13.6.0 直接源码确定 `size` 的输出维度

上一节已经定位 `_kernel.pyx:891-897`：显式 `size` 会执行 `shape.assign(1, size)` 并按该一维 shape 分配输出。这把“捕获到 `size=3`”进一步落实为“CuPy 语义上输出 shape 为 `(3,)`”，而不是 `(3,3)`。

#### 证据四：cuSignal 自带测试的覆盖范围

当前算子的直接测试位于：

```text
ZKX/cusignal-23.08.00/python/cusignal/test/test_waveforms.py:168-190
```

它只参数化：

```python
num_samps = 2**14
idx = "mid"
```

然后比较一维 SciPy 结果与一维 cuSignal 结果。测试没有覆盖：

- tuple shape；
- tuple idx；
- 标量 idx 向多维广播；
- `dtype` 参数。

所以“一维测试能通过”不能证明 docstring 的 N-D 和 dtype 契约已经实现。

#### 最终判定

- **docstring 对接口契约的描述是正确的**：SciPy 实际运行给出了完全一致的四个示例和 dtype 行为；
- **当前 cuSignal 23.08.00 实现没有完整遵守这份契约**：参数捕获、返回语句和 CuPy 13.6.0 的 `size` 定义共同证明它只形成一维调用，并忽略 dtype；
- 因此不是“docstring 和代码解释二选一谁绝对正确”，而是“docstring 表示预期契约，当前代码存在实现缺口”；
- ZQ500 上的通用 NVIDIA CuPy kernel 仍因 `cudaErrorNoDevice` 无法成功执行，但这不推翻上述契约对照和 Python 参数传递证据。

## 12. 建议的亲自阅读顺序

### 第一步：公开入口

依次看：

1. `Learning/cusignal-23.08.00/python/cusignal/__init__.py:78-86`；
2. `Learning/cusignal-23.08.00/python/cusignal/waveforms/__init__.py:14-22`。

观察问题：两个文件有没有执行数学计算？为什么用户可以直接调用 `cusignal.unit_impulse`？

### 第二步：kernel 声明

看 `Learning/cusignal-23.08.00/python/cusignal/waveforms/waveforms.py:569-588`。

观察问题：

- 输入和输出类型各是什么？
- `i` 从哪里来？
- 哪两条赋值对应 Kronecker delta 的两个分支？
- `dtype` 有没有出现在 kernel 声明中？

### 第三步：函数签名和 docstring

看同文件 `:591-664`。

观察问题：文档承诺了哪些 shape、idx 和 dtype 行为？哪些承诺需要继续到函数体核实？

### 第四步：实际控制流

看同文件 `:665-680`。

观察问题：

- 标量 shape 如何变成数组？
- 三个显式 idx 分支分别做什么？
- iterable idx 为什么没有最终 `else`？
- 最后一行究竟使用了 shape 和 idx 的几个分量？

### 第五步：反向核对数学公式

回到 `Learning/operators/waveforms/unit_impulse/数学物理原理.md`，只把以下公式映射到 kernel：

$$
\delta[i-k]=
\begin{cases}
1,&i=k,\\
0,&i\ne k.
\end{cases}
$$

不要把冲激响应、卷积或 DTFT 误当作 `unit_impulse` 内部执行的算法。

## 13. 阶段二自检问题

1. `cusignal.unit_impulse` 如何从顶层包定位到 `waveforms.py`？

   **答案：**`Learning/cusignal-23.08.00/python/cusignal/__init__.py:78-86` 从 `cusignal.waveforms.waveforms` 导入 `unit_impulse`，把它绑定到顶层 `cusignal` 命名空间；`waveforms/__init__.py:14-22` 还把同一函数暴露到 `cusignal.waveforms`。真正定义和计算位于 `waveforms.py:569-680`。
2. `shape=8` 经过 `np.atleast_1d` 后是什么对象和 shape？

   **答案：**它变成一个一维 NumPy 数组，概念上为 `array([8])`，其数组 shape 是 `(1,)`、`len(shape)` 为 `1`，而 `shape[0]` 的值为 `8`。注意这里的 `(1,)` 是元数据数组自身的 shape，不是最终冲激输出长度；最终输出长度由值 `shape[0]=8` 决定。
3. 为什么 `(0,)` 中的逗号不能省略？

   **答案：**Python 中单元素 tuple 由逗号定义。`(0,)` 是只含整数 `0` 的 tuple，而 `(0)` 只是加括号的整数 `0`。源码需要 tuple 才能通过 `(0,) * len(shape)` 重复出每一维的默认索引。
4. `idx=None`、`idx="mid"`、标量 idx 和 tuple idx 分别经过哪个分支？

   **答案：**`None` 进入 `if idx is None`，生成全零 tuple；`"mid"` 进入第一个 `elif`，逐维计算 `shape // 2`；没有 `__iter__` 的标量进入第二个 `elif`，被复制到各维；普通 tuple 具有 `__iter__`，三个显式条件都不满足，因此保持原样。无论得到多少维索引，最后只读取 `idx[0]`。
5. `hasattr(idx, "__iter__")` 检查的是数学上的整数性吗？

   **答案：**不是。它只检查 Python 对象是否实现迭代协议，用于粗略区分标量和可迭代对象。它不会验证值是否为整数、tuple 长度是否匹配维数、索引是否越界，也不会拒绝除 `"mid"` 外的其他可迭代字符串。
6. `ElementwiseKernel` 中的 `i` 从哪里来？

   **答案：**`i` 不是 Python 局部变量，也不是显式 kernel 参数，而是 CuPy `ElementwiseKernel` 在生成逐元素循环代码时提供的特殊索引变量。它表示当前正在处理的线性输出位置，范围由本次调用的 `size` 决定。
7. 为什么 `size=shape[0]` 在本次调用中是必要的？

   **答案：**kernel 的显式输入只有标量 `idx`，没有数组输入供 CuPy 通过 shape 和广播推断输出范围。因此调用必须用 `size=shape[0]` 指定要处理并返回多少个元素，即让 `i` 覆盖 $0$ 到 `shape[0]-1`。
8. 哪几行直接实现 $\delta[i-k]$ 的分段定义？

   **答案：**学习副本 `waveforms.py:577-582`：`if (i != idx)` 分支写 `out = 0`，`else` 分支写 `out = 1`。对应只读基准原始位置为 `waveforms.py:526-530`。
9. 为什么当前返回值是 `float64`，而不是由函数参数 `dtype` 决定？

   **答案：**`_unit_impulse_kernel` 在学习副本第 574 行、基准第 524 行把输出参数声明为 `"float64 out"`；函数体从未读取或转发 `dtype`。因此最终设备输出类型由 kernel 声明固定，而不是由 Python 签名中的 `dtype` 控制。
10. 为什么 docstring 的二维示例不能作为当前实现已支持二维的证据？

    **答案：**示例和说明只是文档。实际返回语句为 `_unit_impulse_kernel(idx[0], size=shape[0])`，既没有使用 `idx[1:]`、`shape[1:]`，也没有多维坐标计算或 reshape。源码证据只能支持一维输出。
11. 越界 idx 为什么会得到全零，而不是由本函数主动抛出异常？

    **答案：**函数没有任何范围判断或显式 `raise`。当 `idx<0` 或 `idx>=shape[0]` 时，kernel 处理的非负索引 $i\in[0,\texttt{shape[0]})$ 没有一个等于 `idx`，所以每个位置都进入 `out = 0` 分支。这里讨论的是通常正输出长度下可由代码直接推出的行为。
12. 该算法的总工作量和输出空间为什么都是 $O(N)$？

    **答案：**长度为 $N=\texttt{shape[0]}$ 的输出必须为每个元素执行一次比较和一次写入，总工作量与 $N$ 成正比，即 $O(N)$；返回数组还必须保存 $N$ 个 `float64` 元素，约占 $8N$ 字节，所以输出空间也是 $O(N)$。GPU 并行缩短实际执行时间，但不减少总元素数和总工作量。

## 14. 本阶段结论

cuSignal 23.08.00 的 `unit_impulse` 由两层组成：

1. Python 层把 `shape` 和 `idx` 规范化；
2. CuPy `ElementwiseKernel` 对每个一维线性索引执行一次相等性判断。

真正落实数学公式的是：

```cpp
if (i != idx) {
    out = 0;
} else {
    out = 1;
}
```

当前源码只把 `idx[0]` 和 `shape[0]` 传给 kernel，并将输出固定声明为 `float64`。因此这份实现可确认的是一维单位冲激；docstring 中的 N-D 和可选 dtype 行为没有在当前函数体中落实。
