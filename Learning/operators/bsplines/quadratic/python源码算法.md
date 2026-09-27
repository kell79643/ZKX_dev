# quadratic Python 源码算法

## 1. 代码定位与接口概览

本阶段只研究 `quadratic` 的 Python/CuPy 实现，不进入 `cusignal_cpp`。

### 1.1 公开导入路径

用户可以通过以下两条公开路径访问同一个函数对象：

```python
import cusignal
cusignal.quadratic(x)

from cusignal.bsplines import quadratic
quadratic(x)
```

代码定位如下。学习副本行号会因为插入学习注释而增大；只读基准行号对应未修改原文。

| 职责 | 符号 | 学习副本位置 | 只读基准位置 |
| --- | --- | --- | --- |
| CuPy 依赖 | `cp` | `Learning/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:15` | `ZKX/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:14` |
| 顶层公开导出 | `quadratic` | `Learning/cusignal-23.08.00/python/cusignal/__init__.py:22` | `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:20` |
| `bsplines` 子包导出 | `quadratic` | `Learning/cusignal-23.08.00/python/cusignal/bsplines/__init__.py:16` | `ZKX/cusignal-23.08.00/python/cusignal/bsplines/__init__.py:14` |
| 逐元素核心计算 | `_quadratic_kernel` | `Learning/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:103-134` | `ZKX/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:79-95` |
| Python 包装函数 | `quadratic` | `Learning/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:138-149` | `ZKX/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:98-105` |

### 1.2 函数签名

```python
def quadratic(x):
```

- `x`：待求二次 B 样条值的坐标。源码没有类型注解、默认值、显式 shape 检查或显式 dtype 检查。
- 返回值：与 `cp.asarray(x)` 的 shape 逐元素对应的 CuPy 结果数组；标量输入先成为零维 CuPy 数组。
- dtype：`ElementwiseKernel` 的输入和输出都声明为 `T`，因此输出复用为输入选择的类型变量。源码没有主动提升整数输入到浮点类型；若需要保留分数，调用者应使用浮点输入。对于不能支持比较和 `abs` 表达式的类型，源码也没有预先给出友好异常，而会在 CuPy 类型解析或 kernel 编译/调用阶段失败。
- 数学定义域：实数坐标最符合该实现，因为 kernel 要对 `ax` 执行 `< 0.5` 和 `< 1.5` 比较。

### 1.3 本算子的直接依赖边界

`quadratic` 只直接依赖：

1. 第三方库 `cupy` 的 `asarray`；
2. 第三方库 `cupy.ElementwiseKernel`；
3. 本文件内定义的 `_quadratic_kernel`。

它不调用项目内的通用 `bspline`，不调用 FFT、插值求解器、卷积函数或另一个项目 helper。docstring 中的“等价于 `bspline(x, 2)`”是数学/API 语义说明，不是实际调用链。

## 2. 当前算子的完整相关源码

以下摘录严格采用 `ZKX/cusignal-23.08.00` 只读基准原文；没有加入学习注释，也没有用省略号删减当前算子的相关符号。原文件中同属 `bsplines.py` 的 `gauss_spline` 和 `cubic` 与当前算子无关，故不摘录。

### 2.1 CuPy 导入

基准位置：`ZKX/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:14`

```python
import cupy as cp
```

### 2.2 必要的公开导出语句

基准位置：`ZKX/cusignal-23.08.00/python/cusignal/bsplines/__init__.py:14`

```python
from cusignal.bsplines.bsplines import cubic, gauss_spline, quadratic
```

基准位置：`ZKX/cusignal-23.08.00/python/cusignal/__init__.py:20`

```python
from cusignal.bsplines.bsplines import cubic, gauss_spline, quadratic
```

这两条原始语句同时导入三个符号。为了保持源码原文，不能只截取逗号列表中的 `quadratic`。

### 2.3 完整 `_quadratic_kernel`

基准位置：`ZKX/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:79-95`

```python
_quadratic_kernel = cp.ElementwiseKernel(
    "T x",
    "T res",
    """
    const T ax { abs( x ) };

    if( ax < 0.5 ) {
        res = 0.75 - ax * ax;
    } else if( !( ax < 0.5 ) && ( ax < 1.5 ) ) {
        res = ( ( ax - 1.5 ) * ( ax - 1.5 ) ) * 0.5 ;
    } else {
        res = 0.0;
    }
    """,
    "_quadratic_kernel",
    options=("-std=c++11",),
)
```

### 2.4 完整 `quadratic`

基准位置：`ZKX/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:98-105`

```python
def quadratic(x):
    """A quadratic B-spline.

    This is a special case of `bspline`, and equivalent to ``bspline(x, 2)``.
    """
    x = cp.asarray(x)

    return _quadratic_kernel(x)
```

该 docstring 没有 `Parameters`、`Returns`、`Examples` 小节，因此不存在被省略的 docstring 参数说明或示例。

## 3. docstring 翻译与解释

### 二次 B 样条的定义说明

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:141` 至 `:144`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:99` 至 `:102`。

```python
    """A quadratic B-spline.

    This is a special case of `bspline`, and equivalent to ``bspline(x, 2)``.
    """
```

第一行开始三引号 docstring，并把 `quadratic` 定义为“二次 B 样条”。空行把简述与补充说明分开。第三行说明它是一般 `bspline` 的特例，数学上等价于 `bspline(x, 2)`；单反引号和双反引号都是 reStructuredText/Sphinx 的代码排版语法，不会触发函数调用。最后一行关闭 docstring。原文没有参数表、返回值表或示例，因此这里不存在被省略的 docstring 示例。

## 4. 源代码逐行解释

### 4.1 导入与公开导出

定位：实现模块学习副本 `Learning/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:15`、只读基准 `ZKX/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:14`。

```python
import cupy as cp
```

`import` 加载 CuPy 模块，`as cp` 把它绑定到短名称 `cp`。后面的 `cp.asarray` 和 `cp.ElementwiseKernel` 都通过这个名称查找。

定位：子包导出学习副本 `Learning/cusignal-23.08.00/python/cusignal/bsplines/__init__.py:16`、只读基准 `ZKX/cusignal-23.08.00/python/cusignal/bsplines/__init__.py:14`；顶层导出学习副本 `Learning/cusignal-23.08.00/python/cusignal/__init__.py:22`、只读基准 `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:20`。

```python
from cusignal.bsplines.bsplines import cubic, gauss_spline, quadratic
```

两处导出采用同一条 `from ... import ...` 语句：从实现模块取出三个名称，并分别绑定到 `cusignal.bsplines` 与 `cusignal` 命名空间。当前算子对应列表末尾的 `quadratic`；尾部没有逗号，因为它是该行最后一个名称。导出只决定用户能否调用 `cusignal.quadratic`，不执行样条计算。

### 4.2 构造逐元素 GPU kernel

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:103` 至 `:134`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:79` 至 `:95`。

```python
_quadratic_kernel = cp.ElementwiseKernel(
    "T x",
    "T res",
    """
    const T ax { abs( x ) };

    if( ax < 0.5 ) {
        res = 0.75 - ax * ax;
    } else if( !( ax < 0.5 ) && ( ax < 1.5 ) ) {
        res = ( ( ax - 1.5 ) * ( ax - 1.5 ) ) * 0.5 ;
    } else {
        res = 0.0;
    }
    """,
    "_quadratic_kernel",
    options=("-std=c++11",),
)
```

这是一个完整的 kernel 构造调用，而不是若干互不相关的单行语句：

- `_quadratic_kernel =` 保存 `cp.ElementwiseKernel(...)` 返回的可调用 kernel 对象；外层圆括号允许实参跨行书写。
- `"T x"` 声明输入元素 `x`，`"T res"` 声明输出元素 `res`。`T` 是由 CuPy 根据调用时输入 dtype 推导的占位类型。
- 第三个实参是 CUDA-C/C++ operation 字符串。`const T ax { abs(x) };` 使用 C++ 花括号初始化一个只读局部量，并借助绝对值利用偶对称性。
- `if (ax < 0.5)` 对应中央区间 $|x|<1/2$，赋值语句实现 $3/4-|x|^2$。
- `else if (!(ax < 0.5) && (ax < 1.5))` 只在前一条件失败后判断，因此对普通有限实数对应 $1/2\le |x|<3/2$；两个相同因子相乘实现 $\tfrac12(|x|-3/2)^2$。
- `else` 覆盖 $|x|\ge3/2$，把结果置零。三个分支合起来实现二次 B 样条的完整分段定义；各个 `}` 依次关闭对应代码块。
- operation 字符串后的 `"_quadratic_kernel"` 是生成 kernel 的内部名称。`options=("-std=c++11",)` 传入单元素 tuple；其中尾逗号是 Python 单元素 tuple 语法。最后的 `)` 结束构造调用。

### 4.3 Python 包装函数：输入正规化后调用 kernel

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:138` 至 `:149`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:98` 至 `:105`。

```python
def quadratic(x):
    """A quadratic B-spline.

    This is a special case of `bspline`, and equivalent to ``bspline(x, 2)``.
    """
    x = cp.asarray(x)

    return _quadratic_kernel(x)
```

`def quadratic(x):` 创建公开函数，`x` 是唯一的必填位置或关键字参数，冒号开始函数体。docstring 已在上一节完整解释。随后 `cp.asarray(x)` 把标量、列表、NumPy/CuPy 兼容对象统一成 CuPy 数组，并把结果重新绑定给局部变量 `x`；是否复制取决于输入的存储位置、dtype 和兼容性。最后一行先调用 `_quadratic_kernel(x)`，让每个 GPU 元素线程计算对应坐标的分段样条值，再由 `return` 把结果数组交给调用者。

这一组织方式仍覆盖了相关源码的每一行非空内容，但把同一个签名、kernel 构造、分支和包装函数保留为完整语义块，阅读时不必在几十个单行标题之间跳转。

## 5. 调用链与算法总结

### 5.1 调用链

```text
用户调用 cusignal.quadratic(x)
  → cusignal/__init__.py 的公开导出绑定
  → bsplines.py::quadratic(x)
  → cp.asarray(x)
  → bsplines.py::_quadratic_kernel(x)
  → 每个元素计算 ax = abs(x)
  → 按 0.5 和 1.5 分三段写入 res
  → 返回 CuPy 结果数组
```

`cusignal.bsplines.quadratic(x)` 从另一个公开导出入口到达同一个包装函数，后续调用链完全相同。

### 5.2 按执行时间区分构造与调用

模块首次导入时：

1. 导入 `cupy as cp`；
2. 执行 `cp.ElementwiseKernel(...)`，创建 `_quadratic_kernel` 对象；
3. 执行 `def quadratic(x): ...`，创建函数对象；
4. 包的 `__init__.py` 把函数对象绑定到公开命名空间。

用户每次调用时：

1. `cp.asarray(x)` 统一输入；
2. `_quadratic_kernel(x)` 执行逐元素分段计算；
3. 返回结果。

因此不能把 `ElementwiseKernel` 的构造语句误认为每个数组元素都重新创建一次 kernel 对象。

### 5.3 数学算法与工程实现的边界

| 类别 | 对应源码 |
| --- | --- |
| 数学核心 | `abs(x)`、两个区间判断、三个分段表达式 |
| Python API 包装 | `def quadratic(x)`、`return` |
| 数组适配 | `cp.asarray(x)` |
| GPU 工程实现 | `cp.ElementwiseKernel`、`T` 类型变量、操作字符串、kernel 名、C++11 选项 |
| 命名空间导出 | 两个 `__init__.py` 的 `from ... import ...` |

## 6. 数学公式与代码逐项映射

令 $a=|x|$。阶段一得到的中心化二次基数 B 样条为

$$
\beta^2(x)=
\begin{cases}
\frac34-a^2,&0\le a<\frac12,\\
\frac12(a-\frac32)^2,&\frac12\le a<\frac32,\\
0,&a\ge\frac32.
\end{cases}
$$

| 数学步骤 | 基准源码 | 学习副本 | 说明 |
| --- | --- | --- | --- |
| $a=|x|$ | `bsplines.py:83` | `bsplines.py:111` | `const T ax { abs(x) };` |
| $a<1/2$ | `:85` | `:114` | 中央区间判断 |
| $3/4-a^2$ | `:86` | `:116` | 中央二次式 |
| $1/2\le a<3/2$ | `:87` | `:118` | 外侧区间判断 |
| $\frac12(a-3/2)^2$ | `:88` | `:120` | 外侧二次式 |
| $a\ge3/2$ | `:89-90` | `:122-124` | 紧支撑外输出零 |

这属于“解析闭式的逐元素直接求值”。源码没有在运行时使用：

- Cox–de Boor 递归；
- 三次盒函数数值卷积；
- FFT；
- 插值系数求解；
- CUDA shared memory 或跨元素通信。

## 7. dtype、shape、边界与数值行为

### 7.1 shape

kernel 是逐元素运算，因此输入数组的每个坐标产生一个输出。不存在 reduction、reshape 或广播多个输入的情形；本函数这里只有一个数组输入。

### 7.2 dtype

输入声明与输出声明都使用 `T`：

```text
T x → T res
```

由此可知源码没有声明独立的浮点输出类型。重要影响是：

- 浮点输入适合保存 $0.75$、$0.5$、$0.125$ 等分数结果；
- 整数输入存在结果转换回整数类型、丢失小数的风险；
- 复数不能按普通方式与 `0.5`、`1.5` 做大小比较，因而不是该分支逻辑的有效数学输入；
- 源码没有 `dtype` 检查，也没有自动调用 `astype(float)`。

学习阶段只依据源码契约说明这些风险；本任务按规则没有在本地运行 Python/CUDA 测试。

### 7.3 精确边界

- $|x|=0.5$：第一条严格 `< 0.5` 为假，进入第二分支；结果为 $0.5(0.5-1.5)^2=0.5$。
- $|x|=1.5$：第二条严格 `< 1.5` 为假，进入 `else`；结果为 0。
- 两个选择都与相邻解析式的极限一致，所以边界函数值连续。

### 7.4 NaN 与无穷大

- 对 $x=\pm\infty$，`ax<0.5` 和 `ax<1.5` 都为假，通常进入 `else` 输出 0，这与“支撑外为零”的延拓一致。
- 对 NaN，所有有序比较通常为假，因此也会落到 `else` 输出 0，而不是传播 NaN。这是由具体条件写法产生的工程行为，不能误写成 B 样条数学定义的一部分。
- 源码没有显式处理或记录这两类特殊值。

### 7.5 数值稳定性

公式只含绝对值、比较、减法和乘法，没有除法、指数、迭代或大规模累加，通常不存在发散问题。外侧段在 $a$ 接近 $1.5$ 时先计算小差值再平方；结果本来就趋近零，主要风险是有限精度下的舍入，而不是算法不稳定。

## 8. 底层执行机制

`cp.ElementwiseKernel` 把同一段操作代码应用到每个元素。概念上，第 $i$ 个逻辑工作项执行：

$$
x_i\longmapsto\beta^2(x_i)=r_i.
$$

每个结果只读取一个输入元素，不依赖相邻元素。因此：

- 没有卷积窗口的数据复用；
- 没有线程间数据依赖；
- 没有算法层面的同步需求；
- 没有临时数组，除输出数组外只需每个元素的局部量 `ax`。

具体 grid、block、kernel 缓存和编译细节由 CuPy 管理，当前项目源码没有显式给出，不能从这几行代码虚构固定的线程块大小。

## 9. 复杂度与性能瓶颈

设输入共有 $N$ 个元素。

- 时间复杂度：$O(N)$。每个元素执行常数次绝对值、比较、乘减和一次输出写入。
- 额外输出空间：$O(N)$，用于保存结果。
- 每元素临时空间：$O(1)$，只有局部变量 `ax`。
- 跨元素辅助空间：$O(1)$，没有显式临时数组。

可能的性能因素：

1. 第一次遇到某个 dtype 时，CuPy 可能需要生成/编译对应 kernel，启动开销相对明显；
2. 对很小数组，kernel 启动开销可能大于算术本身；
3. 对大数组，计算量很低，性能更可能受输入读取、输出写入和分支执行影响；
4. 三个分支可能造成同一执行组内不同元素走不同路径，但每条路径都很短；
5. `cp.asarray` 是否发生数据复制或主机到设备传输，取决于输入对象，源码本身没有进一步控制。

以上是依据源码结构得出的复杂度和潜在瓶颈判断，不代表已经执行了 benchmark。

## 10. 异常检查与缺失机制

该函数没有显式：

- `try/except`；
- 输入范围检查；
- dtype 白名单；
- shape 限制；
- NaN 检查；
- 自定义错误消息；
- 边界条件参数。

发生错误时，异常主要来自 `cp.asarray`、CuPy 类型解析、kernel 编译或 kernel 调用。函数也不执行样条插值的有限区间边界延拓，因为它计算的是一个定义在整个实轴、支撑外为零的单个基函数。

## 11. 建议阅读顺序与检查问题

### 第一遍：只追踪公开调用链

1. 看两个 `__init__.py` 的导入行：确认为什么能调用 `cusignal.quadratic`。
2. 看 `def quadratic(x)`：确认包装函数只有转换和 kernel 调用。
3. 看 `_quadratic_kernel`：确认真正公式在哪里。

检查问题：导出语句是否执行了数学计算？`quadratic` 是否调用了一个名为 `bspline` 的函数？

### 第二遍：只追踪一个元素

假设当前元素分别为 $x=0.25$、$x=1$、$x=2$，逐条判断：

1. `ax` 是多少？
2. 哪个条件为真？
3. `res` 的最终值是多少？

预期分别得到 $0.6875$、$0.125$、$0$。

### 第三遍：区分两种语言

观察三引号边界：

- 三引号外是 Python；
- 三引号内是交给 CuPy 的 C/C++ 风格操作代码；
- 学习副本中，Python 行使用 `# <学习注释：...>`，kernel 字符串内部使用 `// <学习注释：...>`。

检查问题：`const`、花括号、分号为什么不是 Python 语法错误？

### 第四遍：检查边界与类型

1. 为什么 $|x|=0.5$ 进入第二分支？
2. 为什么 $|x|=1.5$ 进入 `else`？
3. 输入整数 dtype 时，`T res` 有何风险？
4. NaN 为什么可能被写成 0？

## 12. 阶段二自检

1. `cp.asarray` 是数学算法还是数组适配？
2. `T x` 和 `T res` 分别声明什么？
3. `_quadratic_kernel` 是何时创建、何时真正处理数据的？
4. `abs(x)` 如何对应偶对称性？
5. 三个源码分支分别对应哪个数学区间和公式？
6. 为什么源码等价于 `bspline(x, 2)`，却没有调用 `bspline`？
7. 两个 `__init__.py` 各自提供什么公开路径？
8. 为什么这段算法是 $O(N)$，而不是卷积常见的滑动窗口复杂度？
9. 当前实现有没有自动把整数输入提升到浮点？
10. 哪些 CuPy 内部执行细节无法从项目源码中确定？

### 12.1 参考答案

1. **`cp.asarray` 是数组适配。** 它把输入统一为 CuPy 数组，但不参与 $\beta^2(x)$ 的分段公式；数学核心在基准 `bsplines.py:83-91`。
2. **`T x` 声明当前输入元素，`T res` 声明对应输出元素。** 两者复用类型变量 `T`，所以源码没有声明独立的浮点输出 dtype。
3. **模块导入时创建 kernel 对象，调用函数时处理数据。** 基准 79-95 行执行 `cp.ElementwiseKernel(...)` 构造；基准 105 行 `_quadratic_kernel(x)` 才把数组交给它计算。
4. **`abs(x)` 把 $x$ 与 $-x$ 映射到同一个 $|x|$。** 因而所有后续条件和公式只依赖绝对值，落实 $\beta^2(-x)=\beta^2(x)$。
5. **三个分支依次是中央段、外侧段和支撑外。** 基准 85-86 行对应 $|x|<0.5$ 与 $0.75-|x|^2$；87-88 行对应 $0.5\le|x|<1.5$ 与 $0.5(|x|-1.5)^2$；89-90 行对应 $|x|\ge1.5$ 输出 0。
6. **等价是数学等价，不是调用等价。** docstring 基准 101 行声明 `bspline(x,2)`，但函数体只有 `cp.asarray` 和 `_quadratic_kernel`；闭式公式已经预先展开，所以无需调用通用递归函数。
7. **子包导出与顶层导出提供不同访问名。** `bsplines/__init__.py:14` 提供 `cusignal.bsplines.quadratic`，顶层 `cusignal/__init__.py:20` 提供 `cusignal.quadratic`；二者绑定同一个实现函数。
8. **每个元素只做常数次运算。** $N$ 个元素各自执行一次绝对值、有限比较和乘减，没有长度随 $N$ 增大的窗口，故时间复杂度为 $O(N)$。
9. **没有。** `cp.asarray(x)` 未传 `dtype=`，kernel 输入输出都写 `T`；因此源码没有显式 `astype(float)` 或独立浮点输出声明，整数结果存在写回整数而丢失分数的风险。
10. **无法从项目源码确定 CuPy 选择的具体 grid/block、缓存键、编译缓存位置及所有 dtype 的生成代码。** 项目只提供 `ElementwiseKernel` 声明；这些调度和生成细节由所用 CuPy 版本内部实现决定，不能凭本文件虚构。

## 13. 完整性说明

- 已完整摘录当前算子的 `quadratic` 定义、全部 docstring、完整函数体、直接使用的 `_quadratic_kernel`、CuPy 导入和必要公开导出语句。
- 当前 docstring 没有参数表、返回值小节或示例，因此没有可摘录的 docstring 示例。
- `Learning/cusignal-23.08.00` 只插入了合法的独立学习注释行；Python 上下文使用 `# <学习注释：...>`，kernel C/C++ 字符串内部使用 `// <学习注释：...>`。
- 完整性复核应在忽略上述两种合法学习注释行后，使学习副本与 `ZKX/cusignal-23.08.00` 只读基准逐行完全一致。
- 本阶段没有修改 `ZKX/cusignal-23.08.00`，没有进入 `cusignal_cpp`，也没有运行本地 Python、CUDA 或硬件测试。
