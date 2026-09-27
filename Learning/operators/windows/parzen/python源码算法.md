# parzen Python 源码算法

## 1. 代码定位与接口概览

### 1.1 当前阶段与源码边界

本阶段只研究 cuSignal 23.08.00 的 `parzen` Python/CuPy 实现。直接相关源码包括：

1. 顶层与 windows 子包的公开导出项；
2. `parzen(M, sym=True)`；
3. 它直接调用的 `_len_guards`、`_extend`、`_truncate`；
4. `_parzen_kernel` 的完整 `cp.ElementwiseKernel` 定义；
5. `get_window` 名称分发表中的 `parzen` 项。

没有读取或摘录同文件中的其他窗函数实现。测试只查看了 `TestParzen` 的直接范围，用于判断已有覆盖，不把 benchmark 包装误称为核心算法。

### 1.2 学习副本与只读基准

| 作用 | 共享根目录 `ZKX_dev/` 下的路径 |
| --- | --- |
| 带学习注释的目标 | `Learning/cusignal-23.08.00/python/cusignal/__init__.py` |
| 带学习注释的目标 | `Learning/cusignal-23.08.00/python/cusignal/windows/__init__.py` |
| 带学习注释的目标 | `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py` |
| 只读基准 | `ZKX/cusignal-23.08.00/python/cusignal/__init__.py` |
| 只读基准 | `ZKX/cusignal-23.08.00/python/cusignal/windows/__init__.py` |
| 只读基准 | `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py` |

写本文档前已机械复核：从三个学习文件中删除所有符合固定格式的 Python 或 C++ 学习注释独立行后，原始内容与对应只读基准逐行一致，差异为 0。

### 1.3 公开调用路径

可以从三个层级访问同一个函数对象：

```python
cusignal.parzen(M, sym=True)
cusignal.windows.parzen(M, sym=True)
cusignal.windows.windows.parzen(M, sym=True)
```

还可通过窗口工厂使用名称 `"parzen"`、`"parz"` 或 `"par"`。名称项后面的 `False` 表示该窗不需要类似 Kaiser `beta` 那样的额外形状参数。

### 1.4 函数签名

```python
def parzen(M, sym=True):
```

| 项目 | 语义 |
| --- | --- |
| `M` | 目标输出长度，API 意图是非负整数。`_len_guards` 拒绝负值和非整数值；代码只比较 `int(M) != M`，因此整数值浮点数可能通过本层检查，但不应视为正式支持的输入类型。 |
| `sym=True` | `True` 生成中心对称窗；`False` 生成适合长度 $M$ 周期延拓的窗。API 意图是布尔值，代码实际按 truthy/falsy 判断。 |
| 返回值 | 一维 `cupy.ndarray`，shape 为 `(M,)`。kernel 明确声明 `float64 w`；短窗路径 `cp.ones(M)` 默认也生成浮点数组。 |
| $M=0$ | 返回空数组。 |
| $M=1$ | 返回 `[1]`。 |
| 非法 `M` | 抛出 `ValueError("Window length M must be a non-negative integer")`。 |

### 1.5 docstring 状态

`parzen` 函数本身在此版本中没有 docstring，因此不存在可摘录的参数说明、返回值说明或示例。不能把 SciPy 文档或其他函数的 docstring 冒充为 cuSignal 源码。三个直接 helper 各有一行 docstring，本文在第 3 节完整翻译。

## 2. 当前算子的完整相关源码

以下内容严格摘自只读基准原文，没有加入学习注释、没有省略当前算子符号内部的任何行。

### 2.1 公开导出项

基准定位：

- `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:106`；学习副本当前为 `Learning/cusignal-23.08.00/python/cusignal/__init__.py:128`。
- `ZKX/cusignal-23.08.00/python/cusignal/windows/__init__.py:34`；学习副本当前为 `Learning/cusignal-23.08.00/python/cusignal/windows/__init__.py:36`。

两个 `parzen,` 都处在各自已有的带括号多行导入语句中。为避免复制同一导入语句内的其他算子，这里只摘录与当前算子直接对应的原始项：

```python
    parzen,
```

### 2.2 长度与周期处理 helper

基准：`ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:20-40`。

学习副本当前定位：`_len_guards` 从第 22 行开始，`_extend` 从第 34 行开始，`_truncate` 从第 47 行开始。

```python
def _len_guards(M):
    """Handle small or incorrect window lengths"""
    if int(M) != M or M < 0:
        raise ValueError("Window length M must be a non-negative integer")
    return M <= 1


def _extend(M, sym):
    """Extend window by 1 sample if needed for DFT-even symmetry"""
    if not sym:
        return M + 1, True
    else:
        return M, False


def _truncate(w, needed):
    """Truncate window by 1 sample if needed for DFT-even symmetry"""
    if needed:
        return w[:-1]
    else:
        return w
```

### 2.3 `_parzen_kernel` 完整定义

基准：`ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:279-318`，符号 `_parzen_kernel`。

学习副本当前：`Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:295-365`。

```python
_parzen_kernel = cp.ElementwiseKernel(
    "",
    "float64 w",
    """
    double n {};
    double temp {};
    double sizeS1 {};

    if ( odd ) {
        sizeS1 = s1 - start + 1.0;
    } else {
        s1 += 0.5;
        s2 += 0.5;
        sizeS1 = s1 - start;
    }

    double sizeS2 { s2 - start + 1.0 - sizeS1 };

    if ( i < sizeS1 ) {
        n = i + start;
        temp = 1.0 - abs( n ) * den;
        w = 2.0 * ( temp * temp * temp );
    } else if ( i >= sizeS1 && i < ( sizeS1 + sizeS2 ) ) {
        n = ( i - sizeS1 - s2 );
        temp = abs( n ) * den;
        w = 1.0 - 6.0 * temp * temp + 6.0 * temp * temp * temp;
    } else {
        n = -( i - sizeS2 + s1 + sizeS1 );
        temp = 1.0 - abs( n ) * den;
        w = 2.0 * temp * temp * temp;
    }
    """,
    "_parzen_kernel",
    options=("-std=c++11",),
    loop_prep="const double start { 0.5 * -( _ind.size () - 1 ) }; \
               const double den { 1.0 / ( 0.5 * _ind.size () ) }; \
               const bool odd { _ind.size() & 1 }; \
               double s1 { floor(-0.25 * ( _ind.size () - 1 ) ) }; \
               double s2 { floor(0.25 * ( _ind.size () - 1 ) ) };",
)
```

### 2.4 `parzen` 完整定义

基准：`ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:321-329`，符号 `parzen`。

学习副本当前：`Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:369-382`。

```python
def parzen(M, sym=True):

    if _len_guards(M):
        return cp.ones(M)
    M, needs_trunc = _extend(M, sym)

    w = _parzen_kernel(size=M)

    return _truncate(w, needs_trunc)
```

### 2.5 `get_window` 名称分发项

基准：`ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1991`；学习副本当前第 2103 行。

```python
    ("parzen", "parz", "par"): (parzen, False),
```

## 3. docstring 逐行翻译与解释

`parzen` 没有 docstring。直接 helper 的全部 docstring 如下：

#### `_len_guards` 的职责说明

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:23` 至 `:23`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:21` 至 `:21`。

```python
    """Handle small or incorrect window lengths"""
```

逐行对应说明：

- 第 1 行：“处理很短或不正确的窗口长度”。它说明 `_len_guards` 同时负责非法输入检查和 $M\le1$ 的快速路径判断。三对双引号构成单行 Python docstring。

#### `_extend` 的 DFT-even 扩展说明

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:35` 至 `:35`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:28` 至 `:28`。

```python
    """Extend window by 1 sample if needed for DFT-even symmetry"""
```

逐行对应说明：

- 第 1 行：“如果 DFT-even 对称需要，则把窗口扩展一个样本”。这里的扩展只用于周期窗构造。

#### `_truncate` 的末点截断说明

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:48` 至 `:48`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:36` 至 `:36`。

```python
    """Truncate window by 1 sample if needed for DFT-even symmetry"""
```

逐行对应说明：

- 第 1 行：“如果 DFT-even 对称需要，则截掉一个样本”。它与 `_extend` 的布尔标志配对。


## 4. 源代码逐行解释

空行保留在第 2 节源码排版中，但按规则无需单独解释。以下最小完整语义块覆盖所有非空原始行，包括续行、括号、docstring 和 kernel 字符串中的 C++/CUDA 语句。

### 4.1 公开导出与名称分发

#### 顶层 `cusignal.parzen` 导出项

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/__init__.py:128` 至 `:128`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:106` 至 `:106`。

```python
    parzen,
```

逐行对应说明：

- 第 1 行：这是顶层多行导入列表中的一个元素。结尾逗号允许列表继续，使 `parzen` 进入 `cusignal` 命名空间。

#### `cusignal.windows.parzen` 子包导出项

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/__init__.py:36` 至 `:36`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/__init__.py:34` 至 `:34`。

```python
    parzen,
```

逐行对应说明：

- 第 1 行：把同一个函数导入 `cusignal.windows` 包命名空间。缩进表明它位于括号包围的多行 import 列表内。

#### `get_window` 的名称与参数标志映射

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:2103` 至 `:2103`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:1991` 至 `:1991`。

```python
    ("parzen", "parz", "par"): (parzen, False),
```

逐行对应说明：

- 第 1 行：字典键是三个等价字符串组成的 tuple；值是函数对象和“是否需要额外参数”的标志。`False` 表示仅凭窗名和长度即可构造。


### 4.2 `_len_guards` 每一行

#### 定义长度守卫

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:22` 至 `:23`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:20` 至 `:21`。

```python
def _len_guards(M):
    """Handle small or incorrect window lengths"""
```

逐行对应说明：

- 第 1 行：`def` 定义函数；前导下划线表示模块内部 helper；`M` 是唯一形参；冒号开始缩进函数体。
- 第 2 行：函数 docstring，见第 3 节。

#### 拒绝负数与非整数值

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:25` 至 `:27`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:22` 至 `:23`。

```python
    if int(M) != M or M < 0:
        raise ValueError("Window length M must be a non-negative integer")
```

逐行对应说明：

- 第 1 行：`int(M) != M` 检查值是否具有整数性，`or` 与负值判断组合；任一条件为真就进入异常分支。短路求值意味着左侧异常时不会继续。
- 第 2 行：构造并抛出 `ValueError`，终止当前调用；字符串明确说明要求非负整数长度。

#### 标记零点或单点短窗

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:29` 至 `:29`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:24` 至 `:24`。

```python
    return M <= 1
```

逐行对应说明：

- 第 1 行：比较表达式产生布尔值。$M=0$ 或 $1$ 返回 `True`，更大合法长度返回 `False`。


注意：`int(M)` 本身可能对某些非数字对象抛出 `TypeError` 或其他转换异常；源码没有捕获。`M=3.0` 在数值比较上也可能通过这一行，因此“类型必须是 Python `int`”并不是本 helper 严格执行的检查。

### 4.3 `_extend` 每一行

#### 定义长度扩展 helper

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:34` 至 `:35`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:27` 至 `:28`。

```python
def _extend(M, sym):
    """Extend window by 1 sample if needed for DFT-even symmetry"""
```

逐行对应说明：

- 第 1 行：定义接收窗长和对称标志的内部 helper。
- 第 2 行：单行 docstring，见第 3 节。

#### 识别周期窗请求

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:37` 至 `:37`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:29` 至 `:29`。

```python
    if not sym:
```

逐行对应说明：

- 第 1 行：`not` 对 `sym` 做布尔取反；`sym` 为假时进入周期窗路径。

#### 周期窗返回扩展长度与截断标志

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:39` 至 `:39`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:30` 至 `:30`。

```python
        return M + 1, True
```

逐行对应说明：

- 第 1 行：返回二元 tuple；内部计算长度增加 1，`True` 记录最终需要截断。逗号负责 tuple 打包。

#### 进入对称窗分支

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:40` 至 `:40`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:31` 至 `:31`。

```python
    else:
```

逐行对应说明：

- 第 1 行：`sym` 为真时进入互斥分支。

#### 对称窗保持长度且无需截断

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:42` 至 `:42`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:32` 至 `:32`。

```python
        return M, False
```

逐行对应说明：

- 第 1 行：保持原长度并返回无需截断标志。


### 4.4 `_truncate` 每一行

#### 定义截断 helper

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:47` 至 `:48`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:35` 至 `:36`。

```python
def _truncate(w, needed):
    """Truncate window by 1 sample if needed for DFT-even symmetry"""
```

逐行对应说明：

- 第 1 行：定义接收窗口数组 `w` 和截断标志的 helper。
- 第 2 行：单行 docstring，见第 3 节。

#### 识别需要删除末点的周期路径

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:50` 至 `:50`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:37` 至 `:37`。

```python
    if needed:
```

逐行对应说明：

- 第 1 行：当 `_extend` 返回 `True` 时进入截断路径。

#### 切片删除扩展末点

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:52` 至 `:52`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:38` 至 `:38`。

```python
        return w[:-1]
```

逐行对应说明：

- 第 1 行：Python 切片省略起点，表示从第一个元素取到倒数第一个元素之前；因此删除末点。对 CuPy 数组通常形成切片视图。

#### 进入无需截断的对称路径

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:53` 至 `:53`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:39` 至 `:39`。

```python
    else:
```

逐行对应说明：

- 第 1 行：无需截断的互斥分支。

#### 原样返回对称窗数组

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:55` 至 `:55`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:40` 至 `:40`。

```python
        return w
```

逐行对应说明：

- 第 1 行：原样返回 kernel 生成的数组。


### 4.5 `_parzen_kernel` Python 构造行

#### 创建 `ElementwiseKernel` 并声明输入输出

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:295` 至 `:301`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:279` 至 `:282`。

```python
_parzen_kernel = cp.ElementwiseKernel(
    "",
    "float64 w",
    """
```

逐行对应说明：

- 第 1 行：调用 CuPy 的逐元素 kernel 构造器，并把返回的可调用 kernel 对象绑定到模块变量。左括号开始跨多行参数列表。
- 第 2 行：输入参数声明为空；kernel 不从用户数组读取每元素输入。尾逗号分隔下一个实参。
- 第 3 行：输出声明只有一个名为 `w` 的 `float64` 元素，因此返回 dtype 固定为双精度浮点。
- 第 4 行：开始多行 kernel operation 字符串；其中内容按 C++/CUDA 语法编译。

#### 结束 operation 并配置名称、编译选项与 `loop_prep`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:354` 至 `:365`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:310` 至 `:318`。

```python
    """,
    "_parzen_kernel",
    options=("-std=c++11",),
    loop_prep="const double start { 0.5 * -( _ind.size () - 1 ) }; \
               const double den { 1.0 / ( 0.5 * _ind.size () ) }; \
               const bool odd { _ind.size() & 1 }; \
               double s1 { floor(-0.25 * ( _ind.size () - 1 ) ) }; \
               double s2 { floor(0.25 * ( _ind.size () - 1 ) ) };",
)
```

逐行对应说明：

- 第 1 行：结束 operation 字符串，并用逗号分隔下一个构造参数。
- 第 2 行：指定 kernel 名称，供编译缓存、生成代码和诊断识别。
- 第 3 行：关键字参数 `options` 是单元素 tuple；内部字符串要求 C++11。单元素 tuple 必须保留逗号。
- 第 4 行：开始 `loop_prep` 字符串，计算中心坐标起点 $-(M-1)/2$；末尾反斜杠让 Python 源码中的字符串跨物理行连续。
- 第 5 行：计算 `den=1/(0.5M)=2/M`，把中心坐标换成归一化距离。
- 第 6 行：`_ind.size() & 1` 用最低位判断内部长度奇偶，非零转换为 `true`。
- 第 7 行：用 C++11 花括号初始化左分段边界，值为 $\lfloor-(M-1)/4\rfloor$。
- 第 8 行：初始化右分段边界 $\lfloor(M-1)/4\rfloor$，随后双引号关闭字符串。
- 第 9 行：关闭 `cp.ElementwiseKernel` 调用；构造出的 kernel 在模块导入期建立。


`loop_prep` 只在逐元素循环开始前计算这些公共量，避免每个输出位置重复计算窗口级常量。

### 4.6 kernel operation 字符串每一行

#### 声明每线程局部变量

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:303` 至 `:307`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:283` 至 `:285`。

```python
    double n {};
    double temp {};
    double sizeS1 {};
```

逐行对应说明：

- 第 1 行：C++11 值初始化，把双精度局部变量 `n` 初始化为 0；后续保存中心坐标。
- 第 2 行：初始化临时量；它在不同分支中保存 $1-|u|$ 或 $|u|$。
- 第 3 行：初始化左侧外段的元素计数；虽然计数概念是整数，源码使用 `double`。

#### 按内部长度奇偶划分左侧外段

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:310` 至 `:322`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:287` 至 `:293`。

```python
    if ( odd ) {
        sizeS1 = s1 - start + 1.0;
    } else {
        s1 += 0.5;
        s2 += 0.5;
        sizeS1 = s1 - start;
    }
```

逐行对应说明：

- 第 1 行：内部长度为奇数时进入该分支；花括号开始 C++ 复合语句。
- 第 2 行：由左边界、起点和包含端点的 `+1` 计算左段长度。这个 `+1` 对某些小奇数长度会产生后文分析的边界异常。
- 第 3 行：关闭奇数分支并开始偶数分支。
- 第 4 行：复合赋值，把左边界向右移动半个样本。
- 第 5 行：右边界同样移动半个样本。
- 第 6 行：偶数长度按半样本边界差计算左段元素数。
- 第 7 行：结束奇偶分支。

#### 计算中央内段元素数

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:325` 至 `:325`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:295` 至 `:295`。

```python
    double sizeS2 { s2 - start + 1.0 - sizeS1 };
```

逐行对应说明：

- 第 1 行：定义并初始化中间内段元素数：到右边界的累计元素数减去左外段数。

#### 按索引执行左外段、中央内段和右外段公式

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:328` 至 `:350`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:297` 至 `:308`。

```python
    if ( i < sizeS1 ) {
        n = i + start;
        temp = 1.0 - abs( n ) * den;
        w = 2.0 * ( temp * temp * temp );
    } else if ( i >= sizeS1 && i < ( sizeS1 + sizeS2 ) ) {
        n = ( i - sizeS1 - s2 );
        temp = abs( n ) * den;
        w = 1.0 - 6.0 * temp * temp + 6.0 * temp * temp * temp;
    } else {
        n = -( i - sizeS2 + s1 + sizeS1 );
        temp = 1.0 - abs( n ) * den;
        w = 2.0 * temp * temp * temp;
```

逐行对应说明：

- 第 1 行：CuPy 自动提供逐元素索引 `i`；左段索引进入外侧三次公式。
- 第 2 行：把从 0 开始的索引平移成以窗中心为原点的坐标。
- 第 3 行：先算 `temp = 1 - abs(n) * den`，即 $1-|u|$。
- 第 4 行：用三次连乘代替幂函数，落实外段 $2(1-|u|)^3$。
- 第 5 行：关闭左段并判断 `i` 是否落在 `[sizeS1,sizeS1+sizeS2)` 的半开中间区间。`&&` 是逻辑与。
- 第 6 行：根据中段局部索引恢复中心坐标。外层括号只用于分组。
- 第 7 行：计算归一化绝对距离 `temp = abs(n) * den`，即 $|u|$。
- 第 8 行：落实内段 $1-6|u|^2+6|u|^3$。
- 第 9 行：剩余索引进入右侧外段。
- 第 10 行：组合右段索引、两段大小和边界得到镜像坐标，再由一元负号改变符号。该式依赖前面段长计算正确。
- 第 11 行：与左外段相同，得到 $1-|u|$。
- 第 12 行：落实右外段公式；括号写法虽与左段不同，乘法结合结果相同。

#### 闭合三段位置分支

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:352` 至 `:352`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:309` 至 `:309`。

```python
    }
```

逐行对应说明：

- 第 1 行：结束三段位置分支。


### 4.7 `parzen` 每一行

#### 定义公开 `parzen` 接口

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:369` 至 `:369`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:321` 至 `:321`。

```python
def parzen(M, sym=True):
```

逐行对应说明：

- 第 1 行：定义公开函数；`sym` 的默认实参在函数定义时绑定为 `True`。冒号开始函数体。

#### 调用长度守卫

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:372` 至 `:372`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:323` 至 `:323`。

```python
    if _len_guards(M):
```

逐行对应说明：

- 第 1 行：调用长度 helper；返回 `True` 时进入短窗快速路径，非法输入则已在 helper 内抛异常。

#### 短窗返回与对称/周期长度扩展

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:374` 至 `:376`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:324` 至 `:325`。

```python
        return cp.ones(M)
    M, needs_trunc = _extend(M, sym)
```

逐行对应说明：

- 第 1 行：创建长度为 `M` 的全 1 CuPy 数组并立即返回；$M=0$ 得空数组，$M=1$ 得单点窗。
- 第 2 行：调用返回 tuple 的 helper，并用序列解包同时覆盖局部 `M`、创建 `needs_trunc`。周期模式此后的 `M` 已是用户长度加 1。

#### 启动无输入逐元素 kernel

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:379` 至 `:379`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:327` 至 `:327`。

```python
    w = _parzen_kernel(size=M)
```

逐行对应说明：

- 第 1 行：以关键字参数 `size` 指定输出元素数；因为没有输入数组，CuPy 无法从输入推断长度，必须显式提供。返回数组绑定为 `w`。

#### 按需截断并返回最终窗口

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/windows/windows.py:382` 至 `:382`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/windows/windows.py:329` 至 `:329`。

```python
    return _truncate(w, needs_trunc)
```

逐行对应说明：

- 第 1 行：对称模式直接返回，周期模式返回 `w[:-1]`，使最终逻辑长度恢复为用户请求的长度。


## 5. 调用链与执行顺序

### 5.1 直接调用

```text
cusignal.parzen / cusignal.windows.parzen
    → parzen(M, sym)
        → _len_guards(M)
        → 若 M <= 1：cp.ones(M) 后返回
        → _extend(M, sym)
        → _parzen_kernel(size=内部长度)
            → loop_prep 计算 start、den、odd、s1、s2
            → 每个索引 i 独立进入左外段、中央内段或右外段
        → _truncate(w, needs_trunc)
        → 返回长度 M 的 cupy.ndarray
```

### 5.2 `get_window` 间接调用

```text
get_window(窗名, Nx)
    → _win_equiv 名称查找
    → 得到 parzen 函数对象
    → 按窗口工厂的 fftbins/sym 约定调用 parzen
```

窗口工厂只负责解析名称和参数；分段三次核心仍全部在 `_parzen_kernel`。

### 5.3 对称模式

1. 用户请求长度 $M$；
2. `_extend` 返回 `(M, False)`；
3. kernel 直接生成 $M$ 点；
4. `_truncate` 原样返回。

### 5.4 周期模式

1. 用户请求长度 $M$；
2. `_extend` 返回 `(M+1, True)`；
3. kernel 生成 $M+1$ 点的内部窗；
4. `_truncate` 删除最后一点，返回前 $M$ 点。

## 6. 数学公式与代码映射

令 kernel 实际使用的内部长度为 $L$。`sym=True` 时 $L=M$；`sym=False` 时 $L=M+1$。

| 数学量或步骤 | 代码位置 | 对应关系 |
| --- | --- | --- |
| 中心起点 $-(L-1)/2$ | B313 | `start = 0.5 * -(size-1)` |
| 归一化因子 $2/L$ | B314 | `den = 1/(0.5*size)` |
| 奇偶性 | B315 | `size & 1` |
| 分段边界近似 $\pm(L-1)/4$ | B316-B317 | `floor` 产生 `s1`、`s2` |
| 外段 $2(1-|u|)^3$ | B299-B300、B307-B308 | `temp=1-abs(n)*den` 后三次连乘 |
| 内段 $1-6|u|^2+6|u|^3$ | B303-B304 | `temp=abs(n)*den` 后计算二次、三次项 |
| 周期窗扩展 | B29-B30 | `M+1, True` |
| 删除周期末点 | B37-B38 | `w[:-1]` |

数学母窗是阶段一的分段三次 Parzen 函数。kernel 没有真的进行“三角窗卷积”；它使用卷积结果的闭式多项式，因此每个输出元素都可独立计算。

## 7. 奇偶长度与静态边界追踪

### 7.1 正常的大长度路径

现有直接测试只参数化了 $2^{15}=32768$ 和 $2^{15}-1=32767$，分别覆盖一个偶数、一个奇数长度，并将 GPU 输出与 `scipy.signal.windows.parzen` 用 `array_equal` 比较：

`ZKX/cusignal-23.08.00/python/cusignal/test/test_windows.py:95-115`。

这些测试属于正确性比较与 benchmark 包装，不是核心实现。本文没有运行它们。

### 7.2 $M=5$ 暴露的源码不一致

逐语句静态代入 `sym=True, M=5`：

$$
start=-2,\quad den=0.4,\quad odd=true,\quad s1=-1,\quad s2=1,
$$

$$
sizeS1=2,\quad sizeS2=2.
$$

按当前 kernel 三段控制流逐个代入，会得到：

$$
[0.016,\ 0.432,\ 0.424,\ 1.0,\ -0.016].
$$

而按阶段一连续母窗在标准中心坐标 $[-2,-1,0,1,2]/2.5$ 上采样，理论序列应为：

$$
[0.016,\ 0.424,\ 1.0,\ 0.424,\ 0.016].
$$

因此，当前源码对某些小奇数长度的分段计数和右段坐标存在静态可见的不一致。最直接的触发点是奇数分支 B288 的 `+1.0` 与 B306 的右段坐标公式共同依赖边界取整。现有测试只覆盖两个很大的长度，未覆盖 $M=5$、其他小长度、`sym=False` 或异常边界。

这是一项 **静态源码推导**，不是已经执行的 GPU 测试结果。本学习任务不修改只读基准，也不擅自修复实现；如果要确认影响范围，应在独立开发任务中补充小长度参数化测试并在 ZQ500 规定环境验证。

## 8. dtype、shape、内存与并行机制

### 8.1 dtype 与 shape

- kernel 输出声明为 `float64 w`，所以主路径固定为双精度实数；
- 返回 shape 为 `(M,)`；
- 无复数输入、无 dtype 参数、无广播输入；
- $M\le1$ 走 `cp.ones(M)`，未显式指定 dtype，CuPy 默认浮点类型通常为 `float64`。

### 8.2 grid、block 与线程职责

Python 源码没有手写 grid、block 或线程块大小。`cp.ElementwiseKernel` 根据 `size=M` 选择启动配置。语义上，每个逻辑索引 `i` 负责且只负责一个 `w[i]`：

1. 读取公共的 `start`、`den`、`odd`、`s1`、`s2`；
2. 判断自己属于哪一段；
3. 做常数次标量算术；
4. 写一个连续输出元素。

不同元素之间没有数据依赖、共享临时缓冲区或 kernel 内同步。

### 8.3 内存布局与数据搬运

- 输出是连续的一维 CuPy device 数组；
- 没有 host 输入数组复制到 device，因为 kernel 没有输入数组；
- 对称模式分配 $M$ 个双精度元素；
- 周期模式先分配 $M+1$ 个元素，再用切片返回前 $M$ 个；切片通常共享原 device allocation；
- 函数不把结果自动复制回 NumPy/CPU。

## 9. 边界、异常与数值稳定性

| 情况 | 源码行为 | 风险或说明 |
| --- | --- | --- |
| $M<0$ | `_len_guards` 抛 `ValueError` | 不进入分配。 |
| 非整数数值 | `int(M) != M` 时抛 `ValueError` | 极大值、NaN、无穷或不可转换对象可能先由 `int(M)` 抛出其他异常。 |
| $M=0,1$ | 返回 `cp.ones(M)` | 不启动 `_parzen_kernel`，避免除以零。 |
| `sym=False` | 内部长度加 1 后切片 | 输出长度仍为用户请求的 $M$。 |
| 偶数对称窗 | 中心位于两样本之间 | 不一定出现精确峰值 1。 |
| 小奇数窗 | 当前段长/坐标公式可能不对称 | 第 7 节给出 $M=5$ 静态证据；现有直接测试未覆盖。 |
| 浮点运算 | 只有加减乘、绝对值和一次公共除法 | 没有迭代收敛、FFT 或病态线性代数问题；但分段边界错误不是浮点舍入问题。 |

## 10. 时间复杂度、空间复杂度与性能瓶颈

| 项目 | 复杂度 | 原因 |
| --- | --- | --- |
| 时间复杂度 | $O(M)$ | 每个输出位置执行常数次分支和算术。 |
| 输出空间 | $O(M)$ | 保存长度 $M$ 的双精度数组。 |
| 周期内部空间 | $O(M+1)$ | 先生成扩展数组。 |
| kernel 额外空间 | 每线程 $O(1)$ | 只有 `n`、`temp` 等标量局部量。 |

潜在性能因素：

- 第一次调用可能包含 CuPy JIT 编译或缓存查找开销；
- 三个位置分支可能造成同一 warp 内控制流分歧，但分段连续，主要只影响边界 warp；
- 算术强度低，较大 $M$ 时可能更接近 kernel 启动和 device 写带宽受限；
- 对很小 $M$，GPU 启动开销远大于几次算术；
- 周期模式多计算并存储一个最终被截掉的元素，这是统一构造换取的微小额外成本。

## 11. 与阶段一数学原理的对应与差异

| 阶段一原理 | Python/CuPy 落实 | 一致或差异 |
| --- | --- | --- |
| 时域窗权重生成 | `parzen` 返回一维权重数组 | 一致；本函数不执行乘窗或 FFT。 |
| 两三角窗卷积得到分段三次式 | kernel B299-B308 | 等价实现；直接计算闭式公式，不实际卷积。 |
| 偶对称 Parzen 母窗 | 左、中、右三段设计 | 设计意图一致；小奇数长度静态追踪显示实现可能偏离偶对称。 |
| 对称/周期采样 | `_extend` 与 `_truncate` | 周期构造采用标准的扩一位再截断。 |
| `float64` 数值生成 | kernel 输出声明 | 工程实现选择，不是数学原理。 |

必须保留这个差异：数学公式正确不代表所有离散分支都正确实现了它。阶段一的 $M=5$ 例子描述的是标准 Parzen 采样；当前 cuSignal kernel 的静态控制流并没有产生同一结果。

## 12. 建议亲自阅读源码的顺序

1. 先读 `parzen` B321-B329：只观察五个动作——guard、extend、kernel、truncate、return。
2. 再读三个 helper B20-B40：分别回答“何时短路”“何时加一”“何时删一”。
3. 读 `loop_prep` B313-B317：手写出 $L=5$、$L=6$ 时的五个公共变量。
4. 读 kernel 三段 B297-B309：为每个 `i` 标出进入哪一段以及对应的 `n`。
5. 把 B299-B304 与阶段一的两段公式逐项连线。
6. 最后读两个公开导出项和 B1991：区分 API 暴露、名称分发和核心计算。
7. 阅读直接测试 B95-B115：观察它只覆盖 32768、32767，思考还缺哪些长度和 `sym` 组合。

## 13. 阅读检查与自检问题

1. 为什么 `_parzen_kernel` 没有输入参数，却必须传 `size=M`？
2. `float64 w` 同时决定了什么？
3. `loop_prep` 与 operation 字符串分别何时执行、面向多少个元素？
4. `den=2/L` 如何把中心样本坐标变成阶段一的归一化 $u$？
5. 外段为什么用 `temp=1-|u|`，内段为什么直接用 `temp=|u|`？
6. `temp * temp * temp` 与 `pow(temp, 3)` 在这里数学上是否等价？工程上为什么常用连乘？
7. `M, needs_trunc = _extend(M, sym)` 使用了哪种 Python 语法？
8. `w[:-1]` 删除的是哪个元素？为什么周期窗需要它？
9. `parzen` 为什么没有处理 FFT、卷积或信号输入？
10. `M=0` 为什么不会让 B314 出现除以零？
11. 为什么说 `int(M) != M` 检查的是数值整数性，而不是严格类型？
12. 对 $M=5$，能否按 B313-B317 算出 `start`、`den`、`sizeS1`、`sizeS2`？
13. 当前 kernel 的 $M=5$ 静态结果为什么不对称？
14. 现有测试覆盖了哪些长度？哪些关键边界未覆盖？
15. 哪些代码属于公开 API、helper、GPU 调度和 device 核心计算？

## 14. 自检问题参考答案

1. 因为 kernel 没有数组输入可供 CuPy 推导输出长度；`size=M` 明确要求启动 $M$ 个逻辑元素并分配长度为 $M$ 的输出。
2. `"float64 w"` 同时声明输出变量名为 `w`，并把输出 dtype 固定为 `float64`。
3. `loop_prep` 中的量在一次 kernel 调用的循环准备阶段建立，供 operation 使用；operation 则对每个输出索引 `i` 执行一次。对应源码是 `_parzen_kernel` 构造中的两个字符串实参。
4. `start=-(L-1)/2` 先把数组索引平移成中心坐标 $n$；`den=2/L` 再令 $u=2n/L$。因此索引距离被归一化到阶段一分段公式使用的无量纲坐标。
5. 外段公式依赖到支撑端点的距离，所以用 $1-|u|$；内段公式直接以中心距离 $|u|$ 构造 $1-6u^2+6|u|^3$，因此两段对 `temp` 的定义不同。
6. 对这里的实数 `temp`，三次连乘与 `pow(temp, 3)` 数学等价。连乘避免通用幂函数的参数处理，编译器也更容易生成少量乘法指令。
7. 这是 Python 的 iterable unpacking（可迭代对象解包）：`_extend` 返回二元组，两个元素分别绑定给 `M` 和 `needs_trunc`。
8. `w[:-1]` 取从开头到倒数第一个元素之前的切片，即删除最后一个样本。周期窗先按 $M+1$ 生成对称窗，再删去与周期首点重复的末点。
9. `parzen` 的输入是窗长和对称性，而不是待处理信号；它直接生成窗系数，所以不需要 FFT、卷积或信号数组。
10. `_len_guards(M)` 在 kernel 启动前处理非正长度并提前返回空数组，因而不会执行含 `den=2/L` 的 kernel 准备代码。
11. `int(M) != M` 先做数值转换再比较，所以 `5.0` 会通过；它判断值能否无损表示为整数，不要求对象的 Python 类型必须是 `int`。
12. $M=5$ 时 `start=-(5-1)/2=-2`，`den=2/5=0.4`，`odd=true`，`s1=floor(-1)=-1`，`s2=floor(1)=1`。因此 `sizeS1=s1-start+1=2`，`sizeS2=s2-start+1-sizeS1=2`。
13. 把上一答案代入三个分支：$i=0,1$ 得 $0.016,0.432$；$i=2,3$ 得 $0.424,1$；$i=4$ 进入末段，源码算出 `n=-3`、`temp=-0.2`、`w=-0.016`。序列既不关于中心对称，末点也为负，因此与标准 Parzen 五点采样不一致；文档据此把它保留为源码事实与验证风险。
14. 现有直接测试覆盖长度 32768 与 32767，主要覆盖大规模偶数、奇数情况；短长度、`M=0/1`、负数、非整数以及 `sym=False` 等关键边界仍需补充。
15. `cusignal/__init__.py` 与 `windows/__init__.py` 的导入属于公开 API；`_len_guards`、`_extend`、`_truncate` 是长度 helper；Python `parzen` 包装函数负责参数处理和 kernel 调度；`_parzen_kernel` operation 字符串是每个 GPU 元素执行的数值核心。

## 15. 本阶段结论

`parzen` 的 Python 包装层非常短：它检查长度、决定对称或周期内部长度、启动一个无输入的 CuPy 逐元素 kernel，再按需删除末点。真正的数值核心位于 `_parzen_kernel`，每个逻辑线程直接计算一个分段三次权重，因此理想复杂度为 $O(M)$、无跨线程依赖。

源码深入追踪还揭示了不能从阶段一公式反推掩盖的差异：当前 kernel 对 $M=5$ 的静态控制流会生成非对称且含负端点的序列，与标准 Parzen 采样不一致；已有直接测试只覆盖两个大长度。该问题应保留为代码事实与验证风险，不在学习副本中擅自修复。
