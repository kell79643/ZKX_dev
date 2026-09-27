# cubic Python 源码算法

## 1. 代码定位与接口概览

公开访问方式：

~~~python
import cusignal
cusignal.cubic(x)

from cusignal.bsplines import cubic
cubic(x)
~~~

| 职责 | 学习副本位置 | 只读基准位置 | 符号 |
| --- | --- | --- | --- |
| 顶层导出 | <code>Learning/cusignal-23.08.00/python/cusignal/__init__.py:22</code> | <code>ZKX/cusignal-23.08.00/python/cusignal/__init__.py:20</code> | <code>cubic</code> |
| 包导出 | <code>Learning/cusignal-23.08.00/python/cusignal/bsplines/__init__.py:16</code> | <code>ZKX/cusignal-23.08.00/python/cusignal/bsplines/__init__.py:14</code> | <code>cubic</code> |
| kernel | <code>Learning/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:52</code> | <code>ZKX/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:50</code> | <code>_cubic_kernel</code> |
| 包装函数 | <code>Learning/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:87</code> | <code>ZKX/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:69</code> | <code>cubic</code> |

函数签名是 <code>def cubic(x):</code>。参数 <code>x</code> 没有默认值，可为能被 <code>cp.asarray</code> 接受的标量或数组对象。返回 CuPy 数组；这里只有一个输入，所以输出 shape 通常与转换后的输入相同。输入和输出都声明为类型变量 <code>T</code>，输出 dtype 跟随输入推导类型；函数没有强制浮点转换，整数输入会把浮点公式结果写回整数类型。

当前算子的完整源码边界只有两条导出、完整 <code>_cubic_kernel</code> 和完整 <code>cubic</code>。原 docstring 没有示例，也没有直接项目内 helper。CuPy 内部实现不属于摘录范围。

## 2. 当前算子的完整相关源码

以下内容严格摘自只读基准，未删减、未使用省略号。

### 2.1 两条公开导出

<code>ZKX/cusignal-23.08.00/python/cusignal/__init__.py:20</code>

~~~python
from cusignal.bsplines.bsplines import cubic, gauss_spline, quadratic
~~~

<code>ZKX/cusignal-23.08.00/python/cusignal/bsplines/__init__.py:14</code>

~~~python
from cusignal.bsplines.bsplines import cubic, gauss_spline, quadratic
~~~

### 2.2 _cubic_kernel 完整定义

<code>ZKX/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:50</code>

~~~python
_cubic_kernel = cp.ElementwiseKernel(
    "T x",
    "T res",
    """
    const T ax { abs( x ) };

    if( ax < 1 ) {
        res =  2.0 / 3 - 1.0 / 2  * ax * ax * ( 2.0 - ax );
    } else if( !( ax < 1 ) && ( ax < 2 ) ) {
        res = 1.0 / 6 * ( 2.0 - ax ) *  ( 2.0 - ax ) * ( 2.0 - ax );
    } else {
        res = 0.0;
    }
    """,
    "_cubic_kernel",
    options=("-std=c++11",),
)
~~~

### 2.3 cubic 完整定义与完整 docstring

<code>ZKX/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:69</code>

~~~python
def cubic(x):
    """A cubic B-spline.

    This is a special case of `bspline`, and equivalent to ``bspline(x, 3)``.
    """
    x = cp.asarray(x)

    return _cubic_kernel(x)
~~~

## 3. docstring 逐行翻译与解释

空行保持原排版，无需单独解释。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:91` 至 `:91`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:70` 至 `:70`。

```python
    """A cubic B-spline.
```

逐行对应说明：

- 第 1 行：开始三引号 docstring；翻译为“一个三次 B 样条”。三次指多项式次数 3。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:93` 至 `:94`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:72` 至 `:73`。

```python
    This is a special case of `bspline`, and equivalent to ``bspline(x, 3)``.
    """
```

逐行对应说明：

- 第 1 行：翻译为“这是 bspline 的特例，等价于 bspline(x, 3)”。反引号是文档排版标记；数字 3 是次数。源码并未调用通用 bspline。
- 第 2 行：关闭 docstring。原文没有 Parameters、Returns、Examples，因而没有遗漏示例。


## 4. 源代码逐行解释

空行无需解释；以下逐项覆盖全部非空源码行。

### 4.1 导出

#### 公开导入或依赖导入

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/__init__.py:22` 至 `:22`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:20` 至 `:20`。

```python
from cusignal.bsplines.bsplines import cubic, gauss_spline, quadratic
```

逐行对应说明：

- 第 1 行：把实现符号绑定到 <code>cusignal</code> 顶层，支持 <code>cusignal.cubic</code>；只是 API 层。

#### 公开导入或依赖导入

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/bsplines/__init__.py:16` 至 `:16`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/bsplines/__init__.py:14` 至 `:14`。

```python
from cusignal.bsplines.bsplines import cubic, gauss_spline, quadratic
```

逐行对应说明：

- 第 1 行：重新导出到 <code>cusignal.bsplines</code>；两条路径引用同一函数。


### 4.2 kernel

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:52` 至 `:60`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:50` 至 `:54`。

```python
_cubic_kernel = cp.ElementwiseKernel(
    "T x",
    "T res",
    """
    const T ax { abs( x ) };
```

逐行对应说明：

- 第 1 行：调用 CuPy 逐元素 kernel 构造器并赋给内部变量；左括号开始跨行调用。
- 第 2 行：输入声明字符串；<code>T</code> 由 dtype 推导，<code>x</code> 是当前元素。
- 第 3 行：输出声明；与输入共用 <code>T</code>，每个输入对应一个输出。
- 第 4 行：开始 operation 多行字符串；其中是要编译的 C/C++ 风格代码。
- 第 5 行：C++11 花括号初始化常量 <code>ax</code>；数学上 $a=|x|$，用绝对值落实偶对称性。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:63` 至 `:83`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:56` 至 `:66`。

```python
    if( ax < 1 ) {
        res =  2.0 / 3 - 1.0 / 2  * ax * ax * ( 2.0 - ax );
    } else if( !( ax < 1 ) && ( ax < 2 ) ) {
        res = 1.0 / 6 * ( 2.0 - ax ) *  ( 2.0 - ax ) * ( 2.0 - ax );
    } else {
        res = 0.0;
    }
    """,
    "_cubic_kernel",
    options=("-std=c++11",),
)
```

逐行对应说明：

- 第 1 行：中心区间条件 $|x|<1$；左花括号开始该分支。
- 第 2 行：计算 $2/3-a^2(2-a)/2$；<code>ax*ax</code> 是 $a^2$，浮点字面量避免整数除法。
- 第 3 行：<code>!</code> 是非，<code>&&</code> 是与；普通有限实数对应 $1\le a<2$。
- 第 4 行：三个相同因子直接相乘，计算 $(2-a)^3/6$，避免通用幂函数。
- 第 5 行：关闭第二分支并开始兜底；普通有限实数对应 $a\ge2$。
- 第 6 行：支撑外输出 0，体现紧支撑 $[-2,2]$。
- 第 7 行：关闭条件结构；每条路径都已写入 <code>res</code>。
- 第 8 行：关闭 operation 字符串，逗号结束该实参。
- 第 9 行：kernel 名称，用于编译缓存和诊断；不同于 Python 变量层次。
- 第 10 行：关键字实参；单元素元组要求尾随逗号；启用 C++11。
- 第 11 行：关闭构造调用；这里只创建 kernel 对象，尚未处理数组。


中心段代码展开为

$$
\frac23-\frac12a^2(2-a)=\frac23-a^2+\frac12a^3.
$$

### 4.3 包装函数

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:87` 至 `:87`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:69` 至 `:69`。

```python
def cubic(x):
```

逐行对应说明：

- 第 1 行：定义函数；只有必填参数 <code>x</code>；冒号开始缩进函数体。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:96` 至 `:96`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:74` 至 `:74`。

```python
    x = cp.asarray(x)
```

逐行对应说明：

- 第 1 行：转成 CuPy 数组或兼容视图；未指定 dtype；主机输入可能发生设备搬运。重新绑定的是局部变量。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:99` 至 `:99`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/bsplines/bsplines.py:76` 至 `:76`。

```python
    return _cubic_kernel(x)
```

逐行对应说明：

- 第 1 行：启动逐元素 kernel 并直接返回 CuPy 数组；无后处理或显式同步。


## 5. 调用链与算法总结

~~~text
cusignal.cubic(x)
  → cusignal.bsplines.bsplines.cubic(x)
  → cp.asarray(x)
  → _cubic_kernel(x)
  → 每元素计算 a=abs(x)
  → 三段条件求值
  → 返回 CuPy 数组
~~~

$$
y_i=
\begin{cases}
\dfrac23-\dfrac12a_i^2(2-a_i),&a_i<1,\\
\dfrac16(2-a_i)^3,&1\le a_i<2,\\
0,&a_i\ge2,
\end{cases}
\qquad a_i=|x_i|.
$$

没有跨元素依赖、循环归约、FFT、插值系数求解或运行时卷积。

## 6. 数学映射、边界与 dtype

| 数学对象 | 基准源码位置 | 说明 |
| --- | --- | --- |
| $a=|x|$ | <code>bsplines.py:54</code> | 偶对称 |
| $a&lt;1$ 与中心公式 | <code>bsplines.py:56-57</code> | 中心分段 |
| $1\le a&lt;2$ 与外侧公式 | <code>bsplines.py:58-59</code> | 支撑边缘 |
| $a\ge2$ 时为 0 | <code>bsplines.py:60-61</code> | 紧支撑外 |

四个盒函数卷积是阶段一中的数学构造；运行代码直接求与其等价的分段闭式公式。

- $x=0$ 输出 $2/3$。
- $|x|=1$ 进入第二分支，输出 $1/6$，与第一公式边界值一致。
- $|x|=2$ 进入兜底，输出 0，与第二公式边界值一致。
- $\pm\infty$ 输出 0。
- NaN 的两次“小于”比较均为假，因此按源码控制流输出 0；这不是一般数学约定。
- 浮点输入是正常用途。整数输入会将结果写回整数 <code>T</code>，可能截断。
- 源码没有声明复数支持，也不显式检查有限性、实数性或空数组。

数值上只使用绝对值、比较、加减乘除，没有迭代或累加。靠近 $a=2$ 时相对误差可能变大，但绝对值也趋近 0。

## 7. 复杂度、性能和阅读检查

输入共有 $N$ 个元素：

- 时间复杂度 $O(N)$；
- 输出空间 $O(N)$；
- 每线程额外空间 $O(1)$；
- 小数组可能由 kernel 启动延迟主导；
- 主机输入的设备搬运可能比计算昂贵；
- 分支不同可能造成 warp 分歧；
- 算术量小，可能受内存读写限制。

建议依次检查：

1. 两条导出为何只是名称绑定；
2. 包装层为何只有数组转换和 kernel 调用；
3. <code>T</code> 如何决定 dtype；
4. <code>abs(x)</code> 如何落实偶对称；
5. 手算 $x=0,1,1.5,2$；
6. 验证两个分段边界连续；
7. 区分卷积构造、闭式实际算法和 CuPy 工程步骤；
8. 解释整数输入风险；
9. 解释为何该 API 不负责求插值系数和边界条件。

## 8. 完整性核验记录

- 注释前，三个学习副本目标文件在忽略既有合法学习注释后与只读基准逐行一致。
- 新增内容仅为独立的 Python <code># &lt;学习注释：...&gt;</code>，或 kernel operation 中合法的 C++ <code>// &lt;学习注释：...&gt;</code> 行。
- 未改写、删除、移动或格式化任何原始行。
- 插入后剔除合法学习注释，三个文件仍与基准逐行一致。
- 未修改 <code>ZKX/cusignal-23.08.00</code>。
- 未运行 Python、CUDA、硬件测试或 ZQ500 环境；结论来自静态源码核对。

## 9. 自检问题与参考答案

1. **两个公开导出提供什么路径？**  
   答：顶层导出支持 <code>cusignal.cubic(x)</code>，包导出支持 <code>from cusignal.bsplines import cubic</code>；二者绑定同一函数。

2. **函数有哪些参数和默认值？**  
   答：签名为 <code>def cubic(x):</code>，只有必填参数 <code>x</code>，没有默认值。

3. **完整调用链是什么？**  
   答：<code>cubic(x) → cp.asarray(x) → _cubic_kernel(x) → 返回 CuPy 数组</code>，证据是学习副本第 96、99 行和基准第 74、76 行。

4. **<code>"T x"</code> 与 <code>"T res"</code> 表示什么？**  
   答：分别声明逐元素 kernel 输入和输出；共同使用 <code>T</code>，输出 dtype 跟随输入推导类型。

5. **公式如何映射到代码？**  
   答：基准第 54 行求 $a=|x|$；56-57 行处理 $a&lt;1$；58-59 行处理 $1\le a&lt;2$；60-61 行把其余输入置零。

6. **整数输入为何危险？**  
   答：输出仍为整数 <code>T</code>，而权重在 $[0,2/3]$，写回时会转换/截断，不能保持真实分数。

7. **三个特殊边界怎样分支？**  
   答：$|x|=1$ 走第二分支得 $1/6$；$|x|=2$ 走 else 得 0；NaN 的小于比较均为假，也走 else 得 0。

8. **是否使用 FFT 或运行时卷积？**  
   答：否。卷积是数学构造，源码直接使用等价分段闭式公式。

9. **复杂度是什么？**  
   答：$N$ 元素总工作量 $O(N)$，输出空间 $O(N)$，每线程额外空间 $O(1)$。

10. **为什么它不是完整插值器？**  
    答：它只求一个基函数的权重，没有接收样本值、求系数、处理结点或边界条件。
