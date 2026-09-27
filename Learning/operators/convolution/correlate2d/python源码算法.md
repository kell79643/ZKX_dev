# `correlate2d` Python 源码算法

## 1. 代码定位与接口概览

### 1.1 公开路径

用户调用路径为：

```python
import cusignal
out = cusignal.correlate2d(in1, in2, mode="full", boundary="fill", fillvalue=0)
```

| 层次 | 学习副本位置 | 只读基准位置 | 符号 |
| --- | --- | --- | --- |
| 顶层导出 | `Learning/cusignal-23.08.00/python/cusignal/__init__.py:32` | `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:29` | `correlate2d` |
| 子包导出 | `Learning/cusignal-23.08.00/python/cusignal/convolution/__init__.py:23` | `ZKX/cusignal-23.08.00/python/cusignal/convolution/__init__.py:22` | `correlate2d` |
| 公开函数 | `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:170` | `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:158` | `correlate2d` |
| `valid` 尺寸 helper | `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:49` | `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:40` | `_inputs_swap_needed` |
| 模式 helper | `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:213` | `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:204` | `_valfrommode` |
| 边界 helper | `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:221` | `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:211` | `_bvalfromboundary` |
| grid helper | `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:232` | `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:221` | `_iDivUp` |
| 二维 kernel 包装 | `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:143` | `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:142` | `_cupy_convolve_2d_wrapper` |
| kernel 缓存/包装 | `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:190,205` | `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:185,199` | `_populate_kernel_cache`、`_get_backend_kernel` |
| GPU 调度 | `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:280` | `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:268` | `_convolve2d_gpu` |
| 统一后端入口 | `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:467` | `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:441` | `_convolve2d` |
| fatbin 加载/调试 | `Learning/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:69,82` | `ZKX/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:68,78` | `_get_function`、`_print_atts` |
| CUDA 核心 | `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:439` | `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:426` | `_cupy_correlate2D<T>` |
| CUDA dtype 入口 | `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:510` 起 | `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:488` 起 | `_cupy_correlate2D_{int32,int64,float32,float64,complex64,complex128}` |

### 1.2 函数签名和数据语义

```python
correlate2d(in1, in2, mode="full", boundary="fill", fillvalue=0)
```

| 参数 | 默认值 | 语义 |
| --- | --- | --- |
| `in1` | 无 | 第一个二维 `array_like`；转为 CuPy 数组。`same` 输出 shape 与它相同。 |
| `in2` | 无 | 第二个二维 `array_like`；进入后端前执行 `in2.conj()`。 |
| `mode` | `"full"` | `full`、`same`、`valid`；决定输出 shape 和 padding。 |
| `boundary` | `"fill"` | `fill`、`wrap`、`symm`，还接受 helper 字典中的别名；决定边界扩展。 |
| `fillvalue` | `0` | `fill` 边界使用的标量。非标量会在后端报错。 |

返回值是二维 CuPy `ndarray`。后端先执行 `cp.promote_types(in1.dtype, in2.dtype)`，再把两输入转换为共同 dtype，并以该 dtype 分配输出。当前编译 kernel 仅支持 `int32`、`int64`、`float32`、`float64`、`complex64`、`complex128`；其他 dtype 即使能进入 promotion，也会在 kernel cache 层被拒绝。

若输入 shape 分别为 $(M,N)$ 与 $(P,Q)$：

- `full`：$(M+P-1,N+Q-1)$；
- `same`：$(M,N)$；
- `valid`：交换后保证第一输入逐维不小于第二输入，结果为 $(M-P+1,N-Q+1)$（这里的 $M,N$ 指交换后的第一输入）。

## 2. 当前算子的完整相关源码

以下摘录全部来自未加学习注释的只读基准，保持原文、原顺序和原排版；没有用省略号删减符号内容。

### 2.1 公开导出

`ZKX/cusignal-23.08.00/python/cusignal/__init__.py:29`

```python
from cusignal.convolution.correlate import correlate, correlate2d
```

`ZKX/cusignal-23.08.00/python/cusignal/convolution/__init__.py:22`

```python
from cusignal.convolution.correlate import correlate, correlate2d, correlation_lags
```

### 2.2 `correlate2d` 完整定义

`ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:158`

```python
def correlate2d(
    in1,
    in2,
    mode="full",
    boundary="fill",
    fillvalue=0,
):
    """
    Cross-correlate two 2-dimensional arrays.
    Cross correlate `in1` and `in2` with output size determined by `mode`, and
    boundary conditions determined by `boundary` and `fillvalue`.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
        ``full``
           The output is the full discrete linear cross-correlation
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    boundary : str {'fill', 'wrap', 'symm'}, optional
        A flag indicating how to handle boundaries:
        ``fill``
           pad input arrays with fillvalue. (default)
        ``wrap``
           circular boundary conditions.
        ``symm``
           symmetrical boundary conditions.
    fillvalue : scalar, optional
        Value to fill pad input arrays with. Default is 0.

    Returns
    -------
    correlate2d : ndarray
        A 2-dimensional array containing a subset of the discrete linear
        cross-correlation of `in1` with `in2`.

    Examples
    --------
    Use 2D cross-correlation to find the location of a template in a noisy
    image:
    >>> import cusignal
    >>> import cupy as cp
    >>> from scipy import misc
    >>> face = cp.asarray(misc.face(gray=True) - misc.face(gray=True).mean())
    >>> template = cp.copy(face[300:365, 670:750])  # right eye
    >>> template -= template.mean()
    >>> face = face + cp.random.randn(*face.shape) * 50  # add noise
    >>> corr = cusignal.correlate2d(face, template, boundary='symm', \
        mode='same')
    >>> y, x = cp.unravel_index(cp.argmax(corr), corr.shape)  # find the match
    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_template, ax_corr) =
    ...     plt.subplots(3, 1, figsize=(6, 15))
    >>> ax_orig.imshow(cp.asnumpy(face), cmap='gray')
    >>> ax_orig.set_title('Original')
    >>> ax_orig.set_axis_off()
    >>> ax_template.imshow(cp.asnumpy(template), cmap='gray')
    >>> ax_template.set_title('Template')
    >>> ax_template.set_axis_off()
    >>> ax_corr.imshow(cp.asnumpy(corr), cmap='gray')
    >>> ax_corr.set_title('Cross-correlation')
    >>> ax_corr.set_axis_off()
    >>> ax_orig.plot(cp.asnumpy(x), cp.asnumpy(y), 'ro')
    >>> fig.show()

    """

    in1 = cp.asarray(in1)
    in2 = cp.asarray(in2)

    if not in1.ndim == in2.ndim == 2:
        raise ValueError("correlate2d inputs must both be 2D arrays")

    swapped_inputs = _inputs_swap_needed(mode, in1.shape, in2.shape)
    if swapped_inputs:
        in1, in2 = in2, in1

    out = _convolution_cuda._convolve2d(
        in1,
        in2.conj(),
        0,
        mode,
        boundary,
        fillvalue,
    )

    if swapped_inputs:
        out = out[::-1, ::-1]

    return out
```

## 3. docstring 逐行翻译与解释

下表中的行号是只读基准行号；每个非空 docstring 行均单独对应。

#### 文档字符串

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:183` 至 `:186`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:165` 至 `:168`。

```python
    """
    Cross-correlate two 2-dimensional arrays.
    Cross correlate `in1` and `in2` with output size determined by `mode`, and
    boundary conditions determined by `boundary` and `fillvalue`.
```

逐行对应说明：

- 第 1 行：开始三引号普通字符串；它成为函数的 `__doc__`。
- 第 2 行：对两个二维数组作互相关。这里明确限定二维。
- 第 3 行：输出尺寸由 `mode` 决定。该行在下一行续写。
- 第 4 行：边界由 `boundary` 和 `fillvalue` 共同决定。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:188` 至 `:199`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:170` 至 `:181`。

```python
    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
        ``full``
           The output is the full discrete linear cross-correlation
           of the inputs. (Default)
        ``valid``
```

逐行对应说明：

- 第 1 行：参数章节标题。
- 第 2 行：NumPy docstring 风格的标题分隔线。
- 第 3 行：`in1` 可为能被 CuPy 转换的类数组对象。
- 第 4 行：它是第一个输入，也是 `same` shape 的基准。
- 第 5 行：声明第二输入。
- 第 6 行：文档说维数应与 `in1` 相同；实现进一步要求二者都恰为二维。
- 第 7 行：输出模式是三个字符串之一，参数可省略。
- 第 8 行：后续条目解释每种输出大小。
- 第 9 行：`full` 模式条目。
- 第 10 行：返回完整离散线性互相关。
- 第 11 行：说明默认模式就是 `full`。
- 第 12 行：`valid` 模式条目。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:200` 至 `:211`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:182` 至 `:193`。

```python
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    boundary : str {'fill', 'wrap', 'symm'}, optional
        A flag indicating how to handle boundaries:
        ``fill``
           pad input arrays with fillvalue. (default)
        ``wrap``
           circular boundary conditions.
```

逐行对应说明：

- 第 1 行：仅保留模板完全落入另一数组的结果。
- 第 2 行：这些元素不依赖零填充，并开始说明尺寸约束。
- 第 3 行：必须有一个输入在每个维度都不小于另一个。
- 第 4 行：`same` 模式条目。
- 第 5 行：输出 shape 与原始 `in1` 相同。
- 第 6 行：从 `full` 结果的中心区域取值。
- 第 7 行：边界模式公开列出 `fill`、`wrap`、`symm`。
- 第 8 行：后续条目解释越界访问。
- 第 9 行：常数填充模式条目。
- 第 10 行：用 `fillvalue` 填充，且这是默认边界。实际实现只 pad 第一输入。
- 第 11 行：周期边界条目。
- 第 12 行：数组首尾在每个维度周期连接。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:212` 至 `:215`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:194` 至 `:197`。

```python
        ``symm``
           symmetrical boundary conditions.
    fillvalue : scalar, optional
        Value to fill pad input arrays with. Default is 0.
```

逐行对应说明：

- 第 1 行：对称边界条目。
- 第 2 行：按边缘对称延拓；实现调用 `cp.pad(..., "symmetric")`。
- 第 3 行：声明常数填充值，应为标量。
- 第 4 行：默认值为 0，仅 `fill` 时影响数据。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:217` 至 `:221`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:199` 至 `:203`。

```python
    Returns
    -------
    correlate2d : ndarray
        A 2-dimensional array containing a subset of the discrete linear
        cross-correlation of `in1` with `in2`.
```

逐行对应说明：

- 第 1 行：返回值章节标题。
- 第 2 行：NumPy docstring 分隔线。
- 第 3 行：返回对象描述为 ndarray；运行时具体是 CuPy ndarray。
- 第 4 行：输出严格为二维，并可能只是完整相关的一部分。
- 第 5 行：明确输入顺序为 `in1` 与 `in2`。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:223` 至 `:234`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:205` 至 `:216`。

```python
    Examples
    --------
    Use 2D cross-correlation to find the location of a template in a noisy
    image:
    >>> import cusignal
    >>> import cupy as cp
    >>> from scipy import misc
    >>> face = cp.asarray(misc.face(gray=True) - misc.face(gray=True).mean())
    >>> template = cp.copy(face[300:365, 670:750])  # right eye
    >>> template -= template.mean()
    >>> face = face + cp.random.randn(*face.shape) * 50  # add noise
    >>> corr = cusignal.correlate2d(face, template, boundary='symm', \
```

逐行对应说明：

- 第 1 行：示例章节标题。
- 第 2 行：示例标题分隔线。
- 第 3 行：示例目标是在含噪图像中定位模板。
- 第 4 行：续完上一行语义。
- 第 5 行：导入待演示的库。
- 第 6 行：以 `cp` 名称导入 CuPy。
- 第 7 行：从 SciPy 导入示例图像来源；这是旧版示例接口。
- 第 8 行：读取灰度图、减去全图均值，再转为 CuPy 数组。减均值不是 API 内部行为。
- 第 9 行：从图像切出右眼区域并复制，shape 为 $65\times80$。
- 第 10 行：原地减去模板均值，降低直流亮度对未归一化相关的影响。
- 第 11 行：生成同 shape 高斯噪声，乘 50 后加到图像。`*face.shape` 把 shape 元组展开为位置参数。
- 第 12 行：调用顶层 API，使用对称边界；反斜杠显式续行。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:235` 至 `:246`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:217` 至 `:228`。

```python
        mode='same')
    >>> y, x = cp.unravel_index(cp.argmax(corr), corr.shape)  # find the match
    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_template, ax_corr) =
    ...     plt.subplots(3, 1, figsize=(6, 15))
    >>> ax_orig.imshow(cp.asnumpy(face), cmap='gray')
    >>> ax_orig.set_title('Original')
    >>> ax_orig.set_axis_off()
    >>> ax_template.imshow(cp.asnumpy(template), cmap='gray')
    >>> ax_template.set_title('Template')
    >>> ax_template.set_axis_off()
    >>> ax_corr.imshow(cp.asnumpy(corr), cmap='gray')
```

逐行对应说明：

- 第 1 行：要求相关图与 `face` 同 shape，并闭合调用。
- 第 2 行：先取扁平最大值索引，再转换为二维 `(y,x)` 坐标。
- 第 3 行：导入绘图库。
- 第 4 行：多目标解包赋值开始；表达式在下一行继续。按普通脚本语法看，此处没有显式括号包围右值或反斜杠，示例排版本身存在可执行性疑点。
- 第 5 行：doctest 续行提示，创建三行一列子图。
- 第 6 行：`cp.asnumpy` 把 GPU 图像复制到 CPU，再显示原图。
- 第 7 行：设置原图标题。
- 第 8 行：隐藏原图坐标轴。
- 第 9 行：把模板复制到 CPU 并显示。
- 第 10 行：设置模板标题。
- 第 11 行：隐藏模板坐标轴。
- 第 12 行：把相关结果复制到 CPU 并显示。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:247` 至 `:250`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:229` 至 `:232`。

```python
    >>> ax_corr.set_title('Cross-correlation')
    >>> ax_corr.set_axis_off()
    >>> ax_orig.plot(cp.asnumpy(x), cp.asnumpy(y), 'ro')
    >>> fig.show()
```

逐行对应说明：

- 第 1 行：设置相关图标题。
- 第 2 行：隐藏相关图坐标轴。
- 第 3 行：在原图的相关峰坐标画红色圆点；`'ro'` 表示 red circle marker。
- 第 4 行：请求显示整张图。

#### 文档字符串

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:252` 至 `:252`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:234` 至 `:234`。

```python
    """
```

逐行对应说明：

- 第 1 行：结束 docstring。


## 4. 公开函数源代码逐行解释

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:170` 至 `:182`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:158` 至 `:164`。

```python
def correlate2d(
    in1,
    in2,
    mode="full",
    boundary="fill",
    fillvalue=0,
):
```

逐行对应说明：

- 第 1 行：`def` 创建函数对象；左括号开启多行形参列表。
- 第 2 行：第一个必需位置或关键字参数；逗号表示后面还有形参。
- 第 3 行：第二个必需参数。
- 第 4 行：可选参数，默认字符串在函数定义时绑定。
- 第 5 行：可选边界参数。
- 第 6 行：可选标量填充值；尾逗号允许下一行闭合。
- 第 7 行：结束形参列表，冒号开始缩进函数体。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:255` 至 `:257`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:236` 至 `:237`。

```python
    in1 = cp.asarray(in1)
    in2 = cp.asarray(in2)
```

逐行对应说明：

- 第 1 行：将 `in1` 统一成 CuPy 数组。若输入已是兼容 GPU 数组，通常避免不必要复制。
- 第 2 行：同样转换第二输入。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:260` 至 `:262`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:239` 至 `:240`。

```python
    if not in1.ndim == in2.ndim == 2:
        raise ValueError("correlate2d inputs must both be 2D arrays")
```

逐行对应说明：

- 第 1 行：Python 链式比较等价于 `not (in1.ndim == in2.ndim and in2.ndim == 2)`。
- 第 2 行：任一输入不是二维时同步抛错。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:265` 至 `:265`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:242` 至 `:242`。

```python
    swapped_inputs = _inputs_swap_needed(mode, in1.shape, in2.shape)
```

逐行对应说明：

- 第 1 行：调用 helper 检查 `valid` 模式的逐维包含关系，并记录是否交换。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:267` 至 `:269`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:243` 至 `:244`。

```python
    if swapped_inputs:
        in1, in2 = in2, in1
```

逐行对应说明：

- 第 1 行：条件为真才执行下一缩进块。
- 第 2 行：元组打包/解包交换引用，不复制数组内容。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:272` 至 `:286`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:246` 至 `:253`。

```python
    out = _convolution_cuda._convolve2d(
        in1,
        in2.conj(),
        0,
        mode,
        boundary,
        fillvalue,
    )
```

逐行对应说明：

- 第 1 行：调用私有统一后端，返回值绑定到 `out`。
- 第 2 行：传入可能已交换的第一输入。
- 第 3 行：逐元素复共轭。对实数值不变；对复数实现相关定义中的 $H^*$。
- 第 4 行：位置参数 `use_convolve=0`，在 Python 条件判断中为假，选择相关 kernel。
- 第 5 行：原样传递模式字符串。
- 第 6 行：原样传递边界字符串。
- 第 7 行：原样传递填充值。
- 第 8 行：结束调用。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:289` 至 `:291`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:255` 至 `:256`。

```python
    if swapped_inputs:
        out = out[::-1, ::-1]
```

逐行对应说明：

- 第 1 行：若之前交换过输入，执行结果方向恢复。
- 第 2 行：两个维度都用负步长切片，生成反向 view；对应互相关交换输入后的位移反转。源码没有再取共轭。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:294` 至 `:294`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:258` 至 `:258`。

```python
    return out
```

逐行对应说明：

- 第 1 行：返回二维 GPU 数组，结束函数。


## 5. 参数与尺寸 helper：完整源码及逐行解释

### 5.1 常量和映射

`ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:19`

```python
FULL = 2
SAME = 1
VALID = 0

CIRCULAR = 8
REFLECT = 4
PAD = 0

_modedict = {"valid": 0, "same": 1, "full": 2}

_boundarydict = {
    "fill": 0,
    "pad": 0,
    "wrap": 2,
    "circular": 2,
    "symm": 1,
    "symmetric": 1,
    "reflect": 4,
}
```

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:20` 至 `:24`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:19` 至 `:21`。

```python
FULL = 2
SAME = 1
VALID = 0
```

逐行对应说明：

- 第 1 行：`FULL` 的内部模式码为 2。
- 第 2 行：`SAME` 的内部模式码为 1。
- 第 3 行：`VALID` 的内部模式码为 0。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:27` 至 `:31`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:23` 至 `:25`。

```python
CIRCULAR = 8
REFLECT = 4
PAD = 0
```

逐行对应说明：

- 第 1 行：`CIRCULAR=8` 是最终周期边界码。
- 第 2 行：`REFLECT=4` 是最终对称边界码。
- 第 3 行：`PAD=0` 是最终常数填充码。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:34` 至 `:34`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:27` 至 `:27`。

```python
_modedict = {"valid": 0, "same": 1, "full": 2}
```

逐行对应说明：

- 第 1 行：`_modedict` 把三个公开字符串映射到上述模式码。字典查找是精确且区分大小写的。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:37` 至 `:45`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:29` 至 `:37`。

```python
_boundarydict = {
    "fill": 0,
    "pad": 0,
    "wrap": 2,
    "circular": 2,
    "symm": 1,
    "symmetric": 1,
    "reflect": 4,
}
```

逐行对应说明：

- 第 1 行：开始多行边界映射字典。
- 第 2 行：`fill` 的基础值为 0。
- 第 3 行：`pad` 是 `fill` 的未写入公开 docstring 的别名。
- 第 4 行：`wrap` 的基础值为 2，稍后左移两位得到 8。
- 第 5 行：`circular` 是 `wrap` 的别名。
- 第 6 行：`symm` 的基础值为 1，左移后得到 4。
- 第 7 行：`symmetric` 是 `symm` 的别名。
- 第 8 行：`reflect` 映射为 4，左移后得到 16，但 `_convolve2d_gpu` 只接受 0、4、8；因此这个字典项对该路径会在后续边界校验失败，是实现与可见字典不完全一致之处。
- 第 9 行：结束字典。


### 5.2 `_inputs_swap_needed` 完整定义

`ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:40`

```python
def _inputs_swap_needed(mode, shape1, shape2):
    """
    If in 'valid' mode, returns whether or not the input arrays need to be
    swapped depending on whether `shape1` is at least as large as `shape2` in
    every dimension.

    This is important for some of the correlation and convolution
    implementations in this module, where the larger array input needs to come
    before the smaller array input when operating in this mode.

    Note that if the mode provided is not 'valid', False is immediately
    returned.
    """
    if mode == "valid":
        ok1, ok2 = True, True

        for d1, d2 in zip(shape1, shape2):
            if not d1 >= d2:
                ok1 = False
            if not d2 >= d1:
                ok2 = False

        if not (ok1 or ok2):
            raise ValueError(
                "For 'valid' mode, one must be at least "
                "as large as the other in every dimension"
            )

        return not ok1

    return False
```

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:49` 至 `:50`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:40` 至 `:41`。

```python
def _inputs_swap_needed(mode, shape1, shape2):
    """
```

逐行对应说明：

- 第 1 行：定义私有 helper，三个参数分别为模式和两个 shape。
- 第 2 行：开始 docstring。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:51` 至 `:53`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:42` 至 `:44`。

```python
    If in 'valid' mode, returns whether or not the input arrays need to be
    swapped depending on whether `shape1` is at least as large as `shape2` in
    every dimension.
```

逐行对应说明：

- 第 1 行：仅在 `valid` 模式判断是否需要交换。
- 第 2 行：判断依据是 `shape1` 是否逐维不小于 `shape2`；句子续行。
- 第 3 行：“每个维度”完成上一句。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:55` 至 `:57`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:46` 至 `:48`。

```python
    This is important for some of the correlation and convolution
    implementations in this module, where the larger array input needs to come
    before the smaller array input when operating in this mode.
```

逐行对应说明：

- 第 1 行：开始说明为什么需要这一检查。
- 第 2 行：某些相关/卷积后端要求较大输入在前；句子续行。
- 第 3 行：限定为当前模式，即 `valid`。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:59` 至 `:61`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:50` 至 `:52`。

```python
    Note that if the mode provided is not 'valid', False is immediately
    returned.
    """
```

逐行对应说明：

- 第 1 行：若不是 `valid`，立即返回 `False`；句子续行。
- 第 2 行：完成上一句。
- 第 3 行：结束 docstring。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:63` 至 `:65`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:53` 至 `:54`。

```python
    if mode == "valid":
        ok1, ok2 = True, True
```

逐行对应说明：

- 第 1 行：精确比较字符串 `"valid"`。非法 mode 在这里也走 `False`，稍后由 `_valfrommode` 报错。
- 第 2 行：同时把 `ok1`、`ok2` 初始化为真。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:68` 至 `:68`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:56` 至 `:56`。

```python
        for d1, d2 in zip(shape1, shape2):
```

逐行对应说明：

- 第 1 行：`zip` 逐维配对；公开函数已保证两个 shape 都有两维。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:70` 至 `:72`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:57` 至 `:58`。

```python
            if not d1 >= d2:
                ok1 = False
```

逐行对应说明：

- 第 1 行：检查第一 shape 当前维是否小于第二 shape。`not d1 >= d2` 等价于 `d1 < d2`。
- 第 2 行：任一维失败后，`ok1` 永久变假。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:74` 至 `:76`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:59` 至 `:60`。

```python
            if not d2 >= d1:
                ok2 = False
```

逐行对应说明：

- 第 1 行：对称检查第二 shape 当前维是否小于第一 shape。
- 第 2 行：任一维失败后，`ok2` 永久变假。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:79` 至 `:85`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:62` 至 `:65`。

```python
        if not (ok1 or ok2):
            raise ValueError(
                "For 'valid' mode, one must be at least "
                "as large as the other in every dimension"
```

逐行对应说明：

- 第 1 行：两种逐维包含关系都不成立时进入错误分支。
- 第 2 行：创建 `ValueError`，实参在后续行拼接。
- 第 3 行：第一段字符串以空格结束。
- 第 4 行：相邻字符串字面量由 Python 编译器自动拼接。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:81` 至 `:81`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:63` 至 `:63`。

```python
            raise ValueError(
```

逐行对应说明：

- 第 1 行：闭合 `ValueError` 调用。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:90` 至 `:90`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:68` 至 `:68`。

```python
        return not ok1
```

逐行对应说明：

- 第 1 行：`ok1` 为假且未报错意味着 `ok2` 为真，因此返回 `True` 要交换。相等 shape 返回 `False`。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:93` 至 `:93`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:70` 至 `:70`。

```python
    return False
```

逐行对应说明：

- 第 1 行：非 `valid` 模式无需交换。


### 5.3 模式、边界与 grid helper 完整定义

`ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:204`

```python
def _valfrommode(mode):
    try:
        return _modedict[mode]
    except KeyError:
        raise ValueError("Acceptable mode flags are 'valid'," " 'same', or 'full'.")


def _bvalfromboundary(boundary):
    try:
        return _boundarydict[boundary] << 2
    except KeyError:
        raise ValueError(
            "Acceptable boundary flags are 'fill', 'circular' "
            "(or 'wrap'), and 'symmetric' (or 'symm')."
        )


def _iDivUp(a, b):
    return (a // b + 1) if (a % b != 0) else (a // b)
```

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:` 至 `:`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:198` 至 `:198`。

```python
        return _boundarydict[boundary] << 2
```

逐行对应说明：

- 第 1 行：定义模式字符串转换 helper。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:` 至 `:`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:199` 至 `:199`。

```python
    except KeyError:
```

逐行对应说明：

- 第 1 行：`try` 开始捕获字典缺键。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:215` 至 `:215`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:191` 至 `:191`。

```python
        return _modedict[mode]
```

逐行对应说明：

- 第 1 行：返回 `_modedict[mode]` 的整数。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:` 至 `:`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:201` 至 `:202`。

```python
            "Acceptable boundary flags are 'fill', 'circular' "
            "(or 'wrap'), and 'symmetric' (or 'symm')."
```

逐行对应说明：

- 第 1 行：只捕获 `KeyError`。不可哈希的 mode 会产生其他异常。
- 第 2 行：把缺键转换成说明三种合法值的 `ValueError`；两个相邻字符串自动拼接。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:` 至 `:`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:205` 至 `:205`。

```python

```

逐行对应说明：

- 第 1 行：定义边界字符串转换 helper。

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:` 至 `:`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:206` 至 `:206`。

```python
def _iDivUp(a, b):
```

逐行对应说明：

- 第 1 行：开始捕获字典缺键。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:` 至 `:`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:207` 至 `:209`。

```python
    return (a // b + 1) if (a % b != 0) else (a // b)


```

逐行对应说明：

- 第 1 行：取基础值并按位左移 2 位，相当于乘 4。
- 第 2 行：捕获未知边界键。
- 第 3 行：开始构造多行 `ValueError`。

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:` 至 `:`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:210` 至 `:212`。

```python
def _reverse_and_conj(x):
    """
    Reverse array `x` in all dimensions and perform the complex conjugate
```

逐行对应说明：

- 第 1 行：错误消息第一段。
- 第 2 行：第二段与前段自动拼接。
- 第 3 行：结束异常构造。

#### 源码注释与相邻逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:` 至 `:`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/convolution_utils.py:215` 至 `:215`。

```python
    # return cp.flip(x, 0)
```

逐行对应说明：

- 第 1 行：定义整数向上除法 helper。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/correlate.py:` 至 `:`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/correlate.py:216` 至 `:216`。

```python
    >>> corr = cusignal.correlate2d(face, template, boundary='symm', \
```

逐行对应说明：

- 第 1 行：若 `a` 不能整除 `b`，返回商加一；否则返回整数商。用于 `ceil(a/b)`。


## 6. kernel 加载与 Python 包装：完整源码及逐行解释

### 6.1 `_cupy_convolve_2d_wrapper`

`ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:142`

```python
class _cupy_convolve_2d_wrapper(object):
    def __init__(self, grid, block, kernel):
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)

        self.grid = grid
        self.block = block
        self.kernel = kernel

    def __call__(
        self,
        d_inp,
        paddedW,
        paddedH,
        d_kernel,
        S0,
        S1,
        out,
        outW,
        outH,
        pick,
    ):

        kernel_args = (
            d_inp,
            paddedW,
            paddedH,
            d_kernel,
            d_kernel.shape[0],
            d_kernel.shape[1],
            S0,
            S1,
            out,
            outW,
            outH,
            pick,
        )

        self.kernel(self.grid, self.block, kernel_args)
```

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:143` 至 `:143`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:142` 至 `:142`。

```python
class _cupy_convolve_2d_wrapper(object):
```

逐行对应说明：

- 第 1 行：定义继承 `object` 的内部包装类。

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:144` 至 `:144`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:143` 至 `:143`。

```python
    def __init__(self, grid, block, kernel):
```

逐行对应说明：

- 第 1 行：构造函数接收 launch grid、block 和 RawKernel。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:145` 至 `:146`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:144` 至 `:145`。

```python
        if isinstance(grid, int):
            grid = (grid,)
```

逐行对应说明：

- 第 1 行：若 grid 是单个整数，先判断类型。
- 第 2 行：转成单元素 tuple，满足 CuPy launch 接口。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:147` 至 `:148`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:146` 至 `:147`。

```python
        if isinstance(block, int):
            block = (block,)
```

逐行对应说明：

- 第 1 行：对 block 做相同类型判断。
- 第 2 行：单整数 block 也转成 tuple。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:150` 至 `:152`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:149` 至 `:151`。

```python
        self.grid = grid
        self.block = block
        self.kernel = kernel
```

逐行对应说明：

- 第 1 行：保存 grid 为实例属性。
- 第 2 行：保存 block。
- 第 3 行：保存底层 kernel。

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:155` 至 `:166`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:153` 至 `:164`。

```python
    def __call__(
        self,
        d_inp,
        paddedW,
        paddedH,
        d_kernel,
        S0,
        S1,
        out,
        outW,
        outH,
        pick,
```

逐行对应说明：

- 第 1 行：定义 `__call__`，使实例可直接调用；形参列表续行。
- 第 2 行：`self` 是实例。
- 第 3 行：`d_inp` 是 padding 后 GPU 输入。
- 第 4 行：`paddedW` 是输入列数。
- 第 5 行：`paddedH` 是输入行数。
- 第 6 行：`d_kernel` 是已共轭模板。
- 第 7 行：`S0` 是半径或模板第一维尺寸。
- 第 8 行：`S1` 是非方形模板第二维尺寸；方形时通常保持 0。
- 第 9 行：`out` 是预分配 GPU 输出。
- 第 10 行：`outW` 是输出列数。
- 第 11 行：`outH` 是输出行数。
- 第 12 行：`pick` 选择奇方形、偶方形或非方形索引分支。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:167` 至 `:167`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:165` 至 `:165`。

```python
    ):
```

逐行对应说明：

- 第 1 行：结束形参列表。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:170` 至 `:181`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:167` 至 `:178`。

```python
        kernel_args = (
            d_inp,
            paddedW,
            paddedH,
            d_kernel,
            d_kernel.shape[0],
            d_kernel.shape[1],
            S0,
            S1,
            out,
            outW,
            outH,
```

逐行对应说明：

- 第 1 行：开始构造严格有序的 kernel 参数 tuple。
- 第 2 行：第 1 个实参：输入指针。
- 第 3 行：第 2 个：输入宽度。
- 第 4 行：第 3 个：输入高度。
- 第 5 行：第 4 个：模板指针。
- 第 6 行：第 5 个：模板 `shape[0]`，传给 CUDA 的 `kerW`；命名与 shape 语义存在互换。
- 第 7 行：第 6 个：模板 `shape[1]`，传给 CUDA 的 `kerH`。
- 第 8 行：第 7 个：`S0`。
- 第 9 行：第 8 个：`S1`。
- 第 10 行：第 9 个：输出指针。
- 第 11 行：第 10 个：输出宽度。
- 第 12 行：第 11 个：输出高度。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:182` 至 `:183`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:179` 至 `:180`。

```python
            pick,
        )
```

逐行对应说明：

- 第 1 行：第 12 个：分支编号。
- 第 2 行：结束参数 tuple。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:186` 至 `:186`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:182` 至 `:182`。

```python
        self.kernel(self.grid, self.block, kernel_args)
```

逐行对应说明：

- 第 1 行：按 `kernel(grid, block, args)` 形式异步启动 RawKernel。


### 6.2 kernel cache 与类型包装

`ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:185`

```python
def _populate_kernel_cache(np_type, k_type):

    if np_type not in _SUPPORTED_TYPES:
        raise ValueError("Datatype {} not found for '{}'".format(np_type, k_type))

    if (str(np_type), k_type) in _cupy_kernel_cache:
        return

    _cupy_kernel_cache[(str(np_type), k_type)] = _get_function(
        "/convolution/_convolution.fatbin",
        "_cupy_" + k_type + "_" + str(np_type),
    )


def _get_backend_kernel(
    dtype,
    grid,
    block,
    k_type,
):

    kernel = _cupy_kernel_cache[(str(dtype), k_type)]
    if kernel:
        if k_type == "convolve" or k_type == "correlate":
            return _cupy_convolve_wrapper(grid, block, kernel)
        elif k_type == "convolve2D" or k_type == "correlate2D":
            return _cupy_convolve_2d_wrapper(grid, block, kernel)
        elif k_type == "convolve1D2O":
            return _cupy_convolve_1d2o_wrapper(grid, block, kernel)
        elif k_type == "convolve1D3O":
            return _cupy_convolve_1d3o_wrapper(grid, block, kernel)
        else:
            raise NotImplementedError(
                "No CuPY kernel found for k_type {}, datatype {}".format(k_type, dtype)
            )
    else:
        raise ValueError("Kernel {} not found in _cupy_kernel_cache".format(k_type))
```

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:190` 至 `:190`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:185` 至 `:185`。

```python
def _populate_kernel_cache(np_type, k_type):
```

逐行对应说明：

- 第 1 行：定义按 dtype 与 kernel 类型填充缓存的函数。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:192` 至 `:193`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:187` 至 `:188`。

```python
    if np_type not in _SUPPORTED_TYPES:
        raise ValueError("Datatype {} not found for '{}'".format(np_type, k_type))
```

逐行对应说明：

- 第 1 行：检查 dtype 是否在六种编译支持类型中。
- 第 2 行：不支持时格式化并抛出 `ValueError`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:195` 至 `:195`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:190` 至 `:190`。

```python
    if (str(np_type), k_type) in _cupy_kernel_cache:
```

逐行对应说明：

- 第 1 行：以 `(dtype字符串,k_type)` 为 cache key 检查是否已加载。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:196` 至 `:196`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:191` 至 `:191`。

```python
        return
```

逐行对应说明：

- 第 1 行：已加载就提前返回，避免重复创建 RawModule。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:198` 至 `:201`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:193` 至 `:196`。

```python
    _cupy_kernel_cache[(str(np_type), k_type)] = _get_function(
        "/convolution/_convolution.fatbin",
        "_cupy_" + k_type + "_" + str(np_type),
    )
```

逐行对应说明：

- 第 1 行：未加载时给该 key 赋值，右侧调用 `_get_function`。
- 第 2 行：fatbin 相对包路径。
- 第 3 行：拼出符号名；本算子例如 `_cupy_correlate2D_float32`。
- 第 4 行：结束加载调用。

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:205` 至 `:210`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:199` 至 `:204`。

```python
def _get_backend_kernel(
    dtype,
    grid,
    block,
    k_type,
):
```

逐行对应说明：

- 第 1 行：定义把 RawKernel 包装为匹配调用约定的 helper；形参续行。
- 第 2 行：dtype 用于 cache key。
- 第 3 行：grid 传入包装器。
- 第 4 行：block 传入包装器。
- 第 5 行：k_type 决定包装器类别。
- 第 6 行：结束签名。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:212` 至 `:212`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:206` 至 `:206`。

```python
    kernel = _cupy_kernel_cache[(str(dtype), k_type)]
```

逐行对应说明：

- 第 1 行：从缓存直接索引取得 kernel；缺键会抛 `KeyError`，正常路径此前已 populate。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:213` 至 `:213`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:207` 至 `:207`。

```python
    if kernel:
```

逐行对应说明：

- 第 1 行：RawKernel truthy 时进入分派。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:214` 至 `:214`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:208` 至 `:208`。

```python
        if k_type == "convolve" or k_type == "correlate":
```

逐行对应说明：

- 第 1 行：一维卷积或相关使用普通一维 wrapper。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:215` 至 `:215`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:209` 至 `:209`。

```python
            return _cupy_convolve_wrapper(grid, block, kernel)
```

逐行对应说明：

- 第 1 行：返回一维 wrapper 实例。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:216` 至 `:216`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:210` 至 `:210`。

```python
        elif k_type == "convolve2D" or k_type == "correlate2D":
```

逐行对应说明：

- 第 1 行：二维卷积或相关使用二维 wrapper；本算子命中 `correlate2D`。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:217` 至 `:217`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:211` 至 `:211`。

```python
            return _cupy_convolve_2d_wrapper(grid, block, kernel)
```

逐行对应说明：

- 第 1 行：返回 `_cupy_convolve_2d_wrapper`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:218` 至 `:218`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:212` 至 `:212`。

```python
        elif k_type == "convolve1D2O":
```

逐行对应说明：

- 第 1 行：二阶一维卷积的分支，与当前算子不相交但属于完整函数原文。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:219` 至 `:219`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:213` 至 `:213`。

```python
            return _cupy_convolve_1d2o_wrapper(grid, block, kernel)
```

逐行对应说明：

- 第 1 行：返回其专用 wrapper。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:220` 至 `:220`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:214` 至 `:214`。

```python
        elif k_type == "convolve1D3O":
```

逐行对应说明：

- 第 1 行：三阶一维卷积分支。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:221` 至 `:221`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:215` 至 `:215`。

```python
            return _cupy_convolve_1d3o_wrapper(grid, block, kernel)
```

逐行对应说明：

- 第 1 行：返回其 wrapper。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:222` 至 `:225`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:216` 至 `:219`。

```python
        else:
            raise NotImplementedError(
                "No CuPY kernel found for k_type {}, datatype {}".format(k_type, dtype)
            )
```

逐行对应说明：

- 第 1 行：未识别的 truthy kernel 类型进入兜底。
- 第 2 行：开始构造 `NotImplementedError`。
- 第 3 行：格式化 kernel 类型与 dtype。
- 第 4 行：结束异常构造。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:226` 至 `:227`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:220` 至 `:221`。

```python
    else:
        raise ValueError("Kernel {} not found in _cupy_kernel_cache".format(k_type))
```

逐行对应说明：

- 第 1 行：kernel 值 falsy 时进入分支。
- 第 2 行：抛出 cache 中未找到有效 kernel 的 `ValueError`。


### 6.3 fatbin 加载与调试 helper

`ZKX/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:68`

```python
def _get_function(fatbin, func):

    dir = os.path.dirname(Path(__file__).parent)

    module = cp.RawModule(
        path=dir + fatbin,
    )
    return module.get_function(func)


def _print_atts(func):
    if os.environ.get("CUSIGNAL_DEV_DEBUG") == "True":
        print("name:", func.kernel.name)
        print("max_threads_per_block:", func.kernel.max_threads_per_block)
        print("num_regs:", func.kernel.num_regs)
        print(
            "max_dynamic_shared_size_bytes:",
            func.kernel.max_dynamic_shared_size_bytes,
        )
        print("shared_size_bytes:", func.kernel.shared_size_bytes)
        print(
            "preferred_shared_memory_carveout:",
            func.kernel.preferred_shared_memory_carveout,
        )
        print("const_size_bytes:", func.kernel.const_size_bytes)
        print("local_size_bytes:", func.kernel.local_size_bytes)
        print("ptx_version:", func.kernel.ptx_version)
        print("binary_version:", func.kernel.binary_version)
        print()
```

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:` 至 `:`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:64` 至 `:64`。

```python
def _get_function(fatbin, func):
```

逐行对应说明：

- 第 1 行：定义从 fatbin 获取函数的 helper。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:` 至 `:`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:66` 至 `:66`。

```python
    dir = os.path.dirname(Path(__file__).parent)
```

逐行对应说明：

- 第 1 行：从当前文件父目录计算包内二进制根目录，变量名 `dir` 会遮蔽 Python 内建 `dir`，但仅局部有效。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:` 至 `:`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:68` 至 `:70`。

```python
    module = cp.RawModule(
        path=dir + fatbin,
    )
```

逐行对应说明：

- 第 1 行：创建 CuPy `RawModule`；调用续行。
- 第 2 行：`path` 是根目录与 fatbin 字符串拼接结果。
- 第 3 行：结束构造。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:` 至 `:`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:71` 至 `:71`。

```python
    return module.get_function(func)
```

逐行对应说明：

- 第 1 行：按导出符号名取得 RawKernel。

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:` 至 `:`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:74` 至 `:74`。

```python
def _print_atts(func):
```

逐行对应说明：

- 第 1 行：定义可选调试打印 helper。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:` 至 `:`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:75` 至 `:86`。

```python
    if os.environ.get("CUSIGNAL_DEV_DEBUG") == "True":
        print("name:", func.kernel.name)
        print("max_threads_per_block:", func.kernel.max_threads_per_block)
        print("num_regs:", func.kernel.num_regs)
        print(
            "max_dynamic_shared_size_bytes:",
            func.kernel.max_dynamic_shared_size_bytes,
        )
        print("shared_size_bytes:", func.kernel.shared_size_bytes)
        print(
            "preferred_shared_memory_carveout:",
            func.kernel.preferred_shared_memory_carveout,
```

逐行对应说明：

- 第 1 行：只有环境变量文本严格等于 `"True"` 才打印。
- 第 2 行：打印 kernel 名称。
- 第 3 行：打印每 block 最大线程数。
- 第 4 行：打印寄存器数量。
- 第 5 行：开始多行 `print`。
- 第 6 行：输出动态共享内存上限标签。
- 第 7 行：读取对应 kernel 属性。
- 第 8 行：结束调用。
- 第 9 行：打印静态共享内存字节数。
- 第 10 行：开始打印共享内存 carveout。
- 第 11 行：输出属性标签。
- 第 12 行：读取属性值。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:` 至 `:`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:87` 至 `:92`。

```python
        )
        print("const_size_bytes:", func.kernel.const_size_bytes)
        print("local_size_bytes:", func.kernel.local_size_bytes)
        print("ptx_version:", func.kernel.ptx_version)
        print("binary_version:", func.kernel.binary_version)
        print()
```

逐行对应说明：

- 第 1 行：结束调用。
- 第 2 行：打印常量内存字节数。
- 第 3 行：打印本地内存字节数。
- 第 4 行：打印 PTX 版本。
- 第 5 行：打印二进制版本。
- 第 6 行：打印空行。整个 helper 只影响调试输出，不影响数值。


## 7. Python GPU 后端：完整源码及逐行解释

### 7.1 `_convolve2d` 统一入口

`ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:441`

```python
def _convolve2d(in1, in2, use_convolve, mode, boundary, fillvalue):

    val = _valfrommode(mode)
    bval = _bvalfromboundary(boundary)

    # Promote inputs
    promType = cp.promote_types(in1.dtype, in2.dtype)
    in1 = in1.astype(promType)
    in2 = in2.astype(promType)

    if (bval != PAD) and (bval != REFLECT) and (bval != CIRCULAR):
        raise Exception("Incorrect boundary value.")

    if (bval == PAD) and (fillvalue is not None):
        fill = np.array(fillvalue, in1.dtype)
        if fill is None:
            raise Exception("fill must no be None.")
        if fill.size != 1:
            if fill.size == 0:
                raise Exception("`fillvalue` cannot be an empty array.")
            raise Exception("`fillvalue` must be scalar or an array with one element")
    else:
        fill = np.zeros(1, in1.dtype)
        if fill is None:
            raise Exception("Unable to create fill array")

    # Create empty array to hold number of aout dimensions
    out_dimens = np.empty(in1.ndim, int)
    if val == VALID:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i] - in2.shape[i] + 1
            if out_dimens[i] < 0:
                raise Exception(
                    "no part of the output is valid, use option 1 (same) or 2 \
                     (full) for third argument"
                )
    elif val == SAME:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i]
    elif val == FULL:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i] + in2.shape[i] - 1
    else:
        raise Exception("mode must be 0 (valid), 1 (same), or 2 (full)")

    # Create empty array out on GPU
    out = cp.empty(out_dimens.tolist(), in1.dtype)

    out = _convolve2d_gpu(
        in1,
        out,
        in2,
        val,
        bval,
        use_convolve,
        fill,
    )

    return out
```

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:467` 至 `:467`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:441` 至 `:441`。

```python
def _convolve2d(in1, in2, use_convolve, mode, boundary, fillvalue):
```

逐行对应说明：

- 第 1 行：定义二维统一后端；`in2` 在相关路径已由公开函数取共轭。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:470` 至 `:472`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:443` 至 `:444`。

```python
    val = _valfrommode(mode)
    bval = _bvalfromboundary(boundary)
```

逐行对应说明：

- 第 1 行：将公开 mode 转为整数 `val`。
- 第 2 行：将边界字符串转为整数 `bval`。

#### 源码注释与相邻逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:474` 至 `:480`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:446` 至 `:449`。

```python
    # Promote inputs
    promType = cp.promote_types(in1.dtype, in2.dtype)
    in1 = in1.astype(promType)
    in2 = in2.astype(promType)
```

逐行对应说明：

- 第 1 行：原注释：下面提升输入类型。
- 第 2 行：根据两输入 dtype 求共同类型。
- 第 3 行：第一输入用 `astype` 转为共同 dtype。默认 `copy=True` 的具体行为由 CuPy 决定。
- 第 4 行：第二输入转为相同 dtype。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:482` 至 `:483`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:451` 至 `:452`。

```python
    if (bval != PAD) and (bval != REFLECT) and (bval != CIRCULAR):
        raise Exception("Incorrect boundary value.")
```

逐行对应说明：

- 第 1 行：检查最终边界码必须是 0、4、8 之一。
- 第 2 行：不合法时抛普通 `Exception`，不是更具体的 `ValueError`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:485` 至 `:486`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:454` 至 `:455`。

```python
    if (bval == PAD) and (fillvalue is not None):
        fill = np.array(fillvalue, in1.dtype)
```

逐行对应说明：

- 第 1 行：只有常数填充且 `fillvalue` 非 `None` 时使用用户值。
- 第 2 行：在 CPU 上按输入 dtype 创建 NumPy 标量/数组 `fill`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:487` 至 `:488`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:456` 至 `:457`。

```python
        if fill is None:
            raise Exception("fill must no be None.")
```

逐行对应说明：

- 第 1 行：检查 `fill is None`；`np.array` 正常不会返回 `None`，属于防御性但基本不可达的检查。
- 第 2 行：若不可达条件成立则抛错；消息有 `no`/`not` 拼写问题。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:489` 至 `:489`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:458` 至 `:458`。

```python
        if fill.size != 1:
```

逐行对应说明：

- 第 1 行：要求 `fill` 总元素数等于 1。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:490` 至 `:492`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:459` 至 `:461`。

```python
            if fill.size == 0:
                raise Exception("`fillvalue` cannot be an empty array.")
            raise Exception("`fillvalue` must be scalar or an array with one element")
```

逐行对应说明：

- 第 1 行：单独检查空数组。
- 第 2 行：空数组得到专门异常。
- 第 3 行：多元素值抛标量约束异常。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:493` 至 `:494`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:462` 至 `:463`。

```python
    else:
        fill = np.zeros(1, in1.dtype)
```

逐行对应说明：

- 第 1 行：非 PAD 或 `fillvalue is None` 走这里。
- 第 2 行：创建一个零元素的 NumPy 数组作为占位填充值。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:495` 至 `:496`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:464` 至 `:465`。

```python
        if fill is None:
            raise Exception("Unable to create fill array")
```

逐行对应说明：

- 第 1 行：再次进行基本不可达的 `None` 检查。
- 第 2 行：不可达时抛错。

#### 源码注释与相邻逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:498` 至 `:499`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:467` 至 `:468`。

```python
    # Create empty array to hold number of aout dimensions
    out_dimens = np.empty(in1.ndim, int)
```

逐行对应说明：

- 第 1 行：原注释：创建保存输出维度的空数组；`aout` 是原文拼写。
- 第 2 行：在 CPU 上创建长度等于输入 ndim 的未初始化整数数组。公开入口下长度为 2。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:501` 至 `:501`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:469` 至 `:469`。

```python
    if val == VALID:
```

逐行对应说明：

- 第 1 行：`valid` 分支。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:502` 至 `:503`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:470` 至 `:471`。

```python
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i] - in2.shape[i] + 1
```

逐行对应说明：

- 第 1 行：遍历每一维。
- 第 2 行：按 $n_1-n_2+1$ 计算该维长度。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:504` 至 `:508`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:472` 至 `:476`。

```python
            if out_dimens[i] < 0:
                raise Exception(
                    "no part of the output is valid, use option 1 (same) or 2 \
                     (full) for third argument"
                )
```

逐行对应说明：

- 第 1 行：防御性检查负长度；正常公开路径已通过 swap helper 保证非负。
- 第 2 行：开始构造异常。
- 第 3 行：第一行字符串用反斜杠续到下一源码行。
- 第 4 行：完成消息，建议 same/full。字符串缩进空格也进入消息。
- 第 5 行：结束异常构造。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:510` 至 `:510`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:477` 至 `:477`。

```python
    elif val == SAME:
```

逐行对应说明：

- 第 1 行：`same` 分支。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:511` 至 `:512`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:478` 至 `:479`。

```python
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i]
```

逐行对应说明：

- 第 1 行：遍历每一维。
- 第 2 行：输出该维等于第一输入该维。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:514` 至 `:514`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:480` 至 `:480`。

```python
    elif val == FULL:
```

逐行对应说明：

- 第 1 行：`full` 分支。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:515` 至 `:516`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:481` 至 `:482`。

```python
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i] + in2.shape[i] - 1
```

逐行对应说明：

- 第 1 行：遍历每一维。
- 第 2 行：输出该维为 $n_1+n_2-1$。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:517` 至 `:518`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:483` 至 `:484`。

```python
    else:
        raise Exception("mode must be 0 (valid), 1 (same), or 2 (full)")
```

逐行对应说明：

- 第 1 行：理论兜底分支；合法映射只产生前三种值。
- 第 2 行：抛出内部模式码错误。

#### 源码注释与相邻逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:520` 至 `:522`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:486` 至 `:487`。

```python
    # Create empty array out on GPU
    out = cp.empty(out_dimens.tolist(), in1.dtype)
```

逐行对应说明：

- 第 1 行：原注释：在 GPU 创建输出。
- 第 2 行：把 NumPy shape 转 list，并以 `in1.dtype` 分配未初始化 CuPy 数组。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:525` 至 `:527`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:489` 至 `:491`。

```python
    out = _convolve2d_gpu(
        in1,
        out,
```

逐行对应说明：

- 第 1 行：调用 GPU 准备/调度层，实参续行。
- 第 2 行：传第一输入。
- 第 3 行：传预分配输出。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:514` 至 `:514`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:492` 至 `:492`。

```python
                                                                             const int kerW,
```

逐行对应说明：

- 第 1 行：传已共轭第二输入作为 `ker`。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:529` 至 `:533`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:493` 至 `:497`。

```python
        val,
        bval,
        use_convolve,
        fill,
    )
```

逐行对应说明：

- 第 1 行：传整数模式。
- 第 2 行：传整数边界。
- 第 3 行：传 `use_convolve=0`。
- 第 4 行：传单元素 NumPy fill 数组。
- 第 5 行：结束调用。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:536` 至 `:536`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:499` 至 `:499`。

```python
    return out
```

逐行对应说明：

- 第 1 行：返回 GPU 调度层返回的输出。


### 7.2 `_convolve2d_gpu` 完整定义

`ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:268`

```python
def _convolve2d_gpu(
    inp,
    out,
    ker,
    mode,
    boundary,
    use_convolve,
    fillvalue,
):

    if (boundary != PAD) and (boundary != REFLECT) and (boundary != CIRCULAR):
        raise Exception("Invalid boundary flag")

    S = np.zeros(2, dtype=int)

    # If kernel is square and odd
    if ker.shape[0] == ker.shape[1]:  # square
        if ker.shape[0] % 2 == 1:  # odd
            pick = 1
            S[0] = (ker.shape[0] - 1) // 2
            if mode == 2:  # full
                P1 = P2 = P3 = P4 = S[0] * 2
            else:  # same/valid
                P1 = P2 = P3 = P4 = S[0]
        else:  # even
            pick = 2
            S[0] = ker.shape[0] // 2
            if mode == 2:  # full
                P1 = P2 = P3 = P4 = S[0] * 2 - 1
            else:  # same/valid
                if use_convolve:
                    P1 = P2 = P3 = P4 = S[0]
                else:
                    P1 = P3 = S[0] - 1
                    P2 = P4 = S[0]
    else:  # Non-square
        pick = 3
        S[0] = ker.shape[0]
        S[1] = ker.shape[1]
        if mode == 2:  # full
            P1 = S[0] - 1
            P2 = S[0] - 1
            P3 = S[1] - 1
            P4 = S[1] - 1
        else:  # same/valid
            if use_convolve:
                P1 = S[0] // 2
                P2 = S[0] // 2 if (S[0] % 2) else S[0] // 2 - 1
                P3 = S[1] // 2
                P4 = S[1] // 2 if (S[1] % 2) else S[1] // 2 - 1
            else:
                P1 = S[0] // 2 if (S[0] % 2) else S[0] // 2 - 1
                P2 = S[0] // 2
                P3 = S[1] // 2 if (S[1] % 2) else S[1] // 2 - 1
                P4 = S[1] // 2

    if mode == 1:  # SAME
        pad = ((P1, P2), (P3, P4))  # 4x5
        if boundary == REFLECT:
            inp = cp.pad(inp, pad, "symmetric")
        if boundary == CIRCULAR:
            inp = cp.pad(inp, pad, "wrap")
        if boundary == PAD:
            inp = cp.pad(inp, pad, "constant", constant_values=(fillvalue))

    if mode == 2:  # FULL
        pad = ((P1, P2), (P3, P4))
        if boundary == REFLECT:
            inp = cp.pad(inp, pad, "symmetric")
        if boundary == CIRCULAR:
            inp = cp.pad(inp, pad, "wrap")
        if boundary == PAD:
            inp = cp.pad(inp, pad, "constant", constant_values=(fillvalue))

    paddedW = inp.shape[1]
    paddedH = inp.shape[0]

    outW = out.shape[1]
    outH = out.shape[0]

    d_inp = cp.asarray(inp)
    d_kernel = cp.asarray(ker)

    threadsperblock = (16, 16)
    blockspergrid = (
        _iDivUp(out.shape[1], threadsperblock[0]),
        _iDivUp(out.shape[0], threadsperblock[1]),
    )

    if use_convolve:
        k_type = "convolve2D"

        _populate_kernel_cache(out.dtype, k_type)

        kernel = _get_backend_kernel(
            out.dtype,
            blockspergrid,
            threadsperblock,
            k_type,
        )
    else:
        k_type = "correlate2D"

        _populate_kernel_cache(out.dtype, k_type)

        kernel = _get_backend_kernel(
            out.dtype,
            blockspergrid,
            threadsperblock,
            k_type,
        )

    kernel(d_inp, paddedW, paddedH, d_kernel, S[0], S[1], out, outW, outH, pick)

    _print_atts(kernel)

    return out
```

#### 签名、检查与模板分类逐行解释

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:280` 至 `:288`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:268` 至 `:276`。

```python
def _convolve2d_gpu(
    inp,
    out,
    ker,
    mode,
    boundary,
    use_convolve,
    fillvalue,
):
```

逐行对应说明：

- 第 1 行：定义函数，形参在后续行展开。
- 第 2 行：`inp`：第一输入。
- 第 3 行：`out`：预分配输出。
- 第 4 行：`ker`：已共轭第二输入。
- 第 5 行：`mode`：0/1/2。
- 第 6 行：`boundary`：0/4/8。
- 第 7 行：`use_convolve`：真假分派标志。
- 第 8 行：`fillvalue`：单元素 NumPy 数组。
- 第 9 行：结束签名。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:291` 至 `:292`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:278` 至 `:279`。

```python
    if (boundary != PAD) and (boundary != REFLECT) and (boundary != CIRCULAR):
        raise Exception("Invalid boundary flag")
```

逐行对应说明：

- 第 1 行：再次验证边界码。
- 第 2 行：非法时抛普通异常。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:295` 至 `:295`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:281` 至 `:281`。

```python
    S = np.zeros(2, dtype=int)
```

逐行对应说明：

- 第 1 行：创建两个整数零的 `S`。

#### 源码注释与相邻逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:297` 至 `:297`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:283` 至 `:283`。

```python
    # If kernel is square and odd
```

逐行对应说明：

- 第 1 行：原注释称下面判断“方形且奇数”，实际下一行先只判断方形。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:299` 至 `:299`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:284` 至 `:284`。

```python
    if ker.shape[0] == ker.shape[1]:  # square
```

逐行对应说明：

- 第 1 行：比较模板两维是否相等；行尾原注释标记 square。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:300` 至 `:302`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:285` 至 `:287`。

```python
        if ker.shape[0] % 2 == 1:  # odd
            pick = 1
            S[0] = (ker.shape[0] - 1) // 2
```

逐行对应说明：

- 第 1 行：方形模板再判断边长是否奇数。
- 第 2 行：`pick=1` 代表奇数方形。
- 第 3 行：`S[0]=(K-1)//2` 是整数中心半径。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:303` 至 `:304`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:288` 至 `:289`。

```python
            if mode == 2:  # full
                P1 = P2 = P3 = P4 = S[0] * 2
```

逐行对应说明：

- 第 1 行：`full` 分支。
- 第 2 行：四侧 padding 都取 $2S_0=K-1$。链式赋值让四变量相等。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:305` 至 `:306`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:290` 至 `:291`。

```python
            else:  # same/valid
                P1 = P2 = P3 = P4 = S[0]
```

逐行对应说明：

- 第 1 行：`same` 与 `valid` 共用此分支。
- 第 2 行：四侧 padding 候选都取半径 $S_0$；`valid` 后面不会实际调用 `cp.pad`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:307` 至 `:309`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:292` 至 `:294`。

```python
        else:  # even
            pick = 2
            S[0] = ker.shape[0] // 2
```

逐行对应说明：

- 第 1 行：偶数方形分支。
- 第 2 行：`pick=2`。
- 第 3 行：`S[0]=K//2`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:310` 至 `:311`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:295` 至 `:296`。

```python
            if mode == 2:  # full
                P1 = P2 = P3 = P4 = S[0] * 2 - 1
```

逐行对应说明：

- 第 1 行：偶方形 `full`。
- 第 2 行：四侧均取 $2S_0-1=K-1$。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:312` 至 `:312`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:297` 至 `:297`。

```python
            else:  # same/valid
```

逐行对应说明：

- 第 1 行：偶方形 `same/valid`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:313` 至 `:314`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:298` 至 `:299`。

```python
                if use_convolve:
                    P1 = P2 = P3 = P4 = S[0]
```

逐行对应说明：

- 第 1 行：卷积与相关对偶数中心采用不同偏移。
- 第 2 行：卷积路径四侧先设为 $S_0$。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:315` 至 `:317`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:300` 至 `:302`。

```python
                else:
                    P1 = P3 = S[0] - 1
                    P2 = P4 = S[0]
```

逐行对应说明：

- 第 1 行：相关路径。
- 第 2 行：相关的上、左侧取 $S_0-1$。变量与实际轴的对应由后续 pad tuple 决定。
- 第 3 行：下、右侧取 $S_0$，体现偶数模板中心偏向。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:318` 至 `:321`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:303` 至 `:306`。

```python
    else:  # Non-square
        pick = 3
        S[0] = ker.shape[0]
        S[1] = ker.shape[1]
```

逐行对应说明：

- 第 1 行：非方形分支；包括任意矩形，即使两维奇偶相同。
- 第 2 行：`pick=3`。
- 第 3 行：`S[0]` 保存模板第一维长度。
- 第 4 行：`S[1]` 保存模板第二维长度。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:322` 至 `:326`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:307` 至 `:311`。

```python
        if mode == 2:  # full
            P1 = S[0] - 1
            P2 = S[0] - 1
            P3 = S[1] - 1
            P4 = S[1] - 1
```

逐行对应说明：

- 第 1 行：非方形 `full`。
- 第 2 行：第一维前侧 padding 为 $P-1$。
- 第 3 行：第一维后侧相同。
- 第 4 行：第二维前侧为 $Q-1$。
- 第 5 行：第二维后侧相同。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:327` 至 `:327`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:312` 至 `:312`。

```python
        else:  # same/valid
```

逐行对应说明：

- 第 1 行：非方形 `same/valid`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:328` 至 `:332`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:313` 至 `:317`。

```python
            if use_convolve:
                P1 = S[0] // 2
                P2 = S[0] // 2 if (S[0] % 2) else S[0] // 2 - 1
                P3 = S[1] // 2
                P4 = S[1] // 2 if (S[1] % 2) else S[1] // 2 - 1
```

逐行对应说明：

- 第 1 行：卷积路径。
- 第 2 行：第一维前侧为 floor($P/2$)。
- 第 3 行：第一维后侧：奇数相同，偶数减一。条件表达式只求被选分支。
- 第 4 行：第二维前侧为 floor($Q/2$)。
- 第 5 行：第二维后侧按奇偶调整。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:333` 至 `:337`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:318` 至 `:322`。

```python
            else:
                P1 = S[0] // 2 if (S[0] % 2) else S[0] // 2 - 1
                P2 = S[0] // 2
                P3 = S[1] // 2 if (S[1] % 2) else S[1] // 2 - 1
                P4 = S[1] // 2
```

逐行对应说明：

- 第 1 行：相关路径。
- 第 2 行：第一维前侧：奇数为 floor($P/2$)，偶数少一。
- 第 3 行：第一维后侧为 floor($P/2$)。
- 第 4 行：第二维前侧作同样奇偶调整。
- 第 5 行：第二维后侧为 floor($Q/2$)。


#### padding、launch 与返回逐行解释

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:340` 至 `:341`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:324` 至 `:325`。

```python
    if mode == 1:  # SAME
        pad = ((P1, P2), (P3, P4))  # 4x5
```

逐行对应说明：

- 第 1 行：只有 `mode==SAME` 执行本块。
- 第 2 行：构造 `((axis0_before,axis0_after),(axis1_before,axis1_after))`；`# 4x5` 是原注释示意。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:342` 至 `:343`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:326` 至 `:327`。

```python
        if boundary == REFLECT:
            inp = cp.pad(inp, pad, "symmetric")
```

逐行对应说明：

- 第 1 行：对称边界分支。
- 第 2 行：CuPy `symmetric` padding 会重复边缘值地镜像扩展。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:344` 至 `:345`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:328` 至 `:329`。

```python
        if boundary == CIRCULAR:
            inp = cp.pad(inp, pad, "wrap")
```

逐行对应说明：

- 第 1 行：周期边界分支。三个 `if` 互斥是因为 boundary 只能取一个码。
- 第 2 行：用 `wrap` 周期扩展。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:346` 至 `:347`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:330` 至 `:331`。

```python
        if boundary == PAD:
            inp = cp.pad(inp, pad, "constant", constant_values=(fillvalue))
```

逐行对应说明：

- 第 1 行：常数边界分支。
- 第 2 行：用 `constant` 扩展，并传入单元素 `fillvalue`。多余括号不构成 tuple。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:350` 至 `:351`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:333` 至 `:334`。

```python
    if mode == 2:  # FULL
        pad = ((P1, P2), (P3, P4))
```

逐行对应说明：

- 第 1 行：`full` 模式独立执行相同类型的 padding。
- 第 2 行：用 full 分支之前算出的四侧宽度建立 pad tuple。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:352` 至 `:353`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:335` 至 `:336`。

```python
        if boundary == REFLECT:
            inp = cp.pad(inp, pad, "symmetric")
```

逐行对应说明：

- 第 1 行：full 的对称分支。
- 第 2 行：对称 padding。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:354` 至 `:355`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:337` 至 `:338`。

```python
        if boundary == CIRCULAR:
            inp = cp.pad(inp, pad, "wrap")
```

逐行对应说明：

- 第 1 行：full 的周期分支。
- 第 2 行：wrap padding。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:356` 至 `:357`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:339` 至 `:340`。

```python
        if boundary == PAD:
            inp = cp.pad(inp, pad, "constant", constant_values=(fillvalue))
```

逐行对应说明：

- 第 1 行：full 的常数分支。
- 第 2 行：constant padding。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:359` 至 `:360`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:342` 至 `:343`。

```python
    paddedW = inp.shape[1]
    paddedH = inp.shape[0]
```

逐行对应说明：

- 第 1 行：padding 后输入 `shape[1]` 保存为 `paddedW`。
- 第 2 行：`shape[0]` 保存为 `paddedH`。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:362` 至 `:363`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:345` 至 `:346`。

```python
    outW = out.shape[1]
    outH = out.shape[0]
```

逐行对应说明：

- 第 1 行：输出第二维保存为 `outW`。
- 第 2 行：输出第一维保存为 `outH`。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:365` 至 `:366`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:348` 至 `:349`。

```python
    d_inp = cp.asarray(inp)
    d_kernel = cp.asarray(ker)
```

逐行对应说明：

- 第 1 行：再用 `cp.asarray` 确保输入是 CuPy 数组；通常已经是。
- 第 2 行：确保模板是 CuPy 数组。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:369` 至 `:373`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:351` 至 `:355`。

```python
    threadsperblock = (16, 16)
    blockspergrid = (
        _iDivUp(out.shape[1], threadsperblock[0]),
        _iDivUp(out.shape[0], threadsperblock[1]),
    )
```

逐行对应说明：

- 第 1 行：固定线程块为 `(16,16)`，共 256 线程。
- 第 2 行：开始构造二维 grid。
- 第 3 行：grid x 覆盖输出列数，向上取整除以 block x。
- 第 4 行：grid y 覆盖输出行数。
- 第 5 行：结束 grid tuple。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:376` 至 `:377`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:357` 至 `:358`。

```python
    if use_convolve:
        k_type = "convolve2D"
```

逐行对应说明：

- 第 1 行：根据 `use_convolve` 分派；本算子传 0，所以不走此分支。
- 第 2 行：卷积 kernel 类型字符串。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:379` 至 `:379`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:360` 至 `:360`。

```python
        _populate_kernel_cache(out.dtype, k_type)
```

逐行对应说明：

- 第 1 行：为输出 dtype 加载/命中卷积 kernel 缓存。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:381` 至 `:386`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:362` 至 `:367`。

```python
        kernel = _get_backend_kernel(
            out.dtype,
            blockspergrid,
            threadsperblock,
            k_type,
        )
```

逐行对应说明：

- 第 1 行：开始取得包装后卷积 kernel。
- 第 2 行：dtype。
- 第 3 行：grid。
- 第 4 行：block。
- 第 5 行：类型字符串。
- 第 6 行：结束调用。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:388` 至 `:389`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:368` 至 `:369`。

```python
    else:
        k_type = "correlate2D"
```

逐行对应说明：

- 第 1 行：`use_convolve` 为假走相关路径。
- 第 2 行：选择 `correlate2D`。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:391` 至 `:391`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:371` 至 `:371`。

```python
        _populate_kernel_cache(out.dtype, k_type)
```

逐行对应说明：

- 第 1 行：按输出 dtype 加载 `_cupy_correlate2D_<dtype>`。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:393` 至 `:396`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:373` 至 `:376`。

```python
        kernel = _get_backend_kernel(
            out.dtype,
            blockspergrid,
            threadsperblock,
```

逐行对应说明：

- 第 1 行：开始取得二维 wrapper。
- 第 2 行：传 dtype。
- 第 3 行：传 grid。
- 第 4 行：传 block。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:389` 至 `:389`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:369` 至 `:369`。

```python
        k_type = "correlate2D"
```

逐行对应说明：

- 第 1 行：传 `correlate2D`。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:398` 至 `:398`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:378` 至 `:378`。

```python
        )
```

逐行对应说明：

- 第 1 行：结束调用。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:401` 至 `:401`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:380` 至 `:380`。

```python
    kernel(d_inp, paddedW, paddedH, d_kernel, S[0], S[1], out, outW, outH, pick)
```

逐行对应说明：

- 第 1 行：调用 wrapper，传输入/模板/尺寸/输出/`pick`，最终启动 CUDA。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:403` 至 `:403`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:382` 至 `:382`。

```python
    _print_atts(kernel)
```

逐行对应说明：

- 第 1 行：可选打印 kernel 属性；默认环境下无输出。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:405` 至 `:405`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/convolution/_convolution_cuda.py:384` 至 `:384`。

```python
    return out
```

逐行对应说明：

- 第 1 行：返回已被 kernel 写入的 `out`。代码没有显式同步，后续依赖由 CuPy stream 语义保证。


## 8. CUDA 直接互相关：完整源码及逐行解释

### 8.1 模板 device 核心

`ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:425`

```cpp
template<typename T>
__device__ void _cupy_correlate2D( const T *__restrict__ inp,
                                   const int inpW,
                                   const int inpH,
                                   const T *__restrict__ kernel,
                                   const int kerW,
                                   const int kerH,
                                   const int S0,
                                   const int S1,
                                   T *__restrict__ out,
                                   const int outW,
                                   const int outH,
                                   const int pick ) {

    const int ty { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int tx { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };

    int i {};
    if ( pick != 3 ) {
        i = tx + S0;
    } else {
        i = tx + S1;
    }
    int j { ty + S0 };

    int2 oPixelPos { tx, ty };
    if ( ( tx < outH ) && ( ty < outW ) ) {
        T temp {};

        // Odd
        if ( pick == 1 ) {
            for ( int k = -S0; k < ( S0 + 1 ); k++ ) {
                for ( int l = -S0; l < ( S0 + 1 ); l++ ) {
                    int2 iPixelPos { ( i + k ), ( j + l ) };
                    int2 coefPos { ( k + S0 ), ( l + S0 ) };
                    temp += inp[iPixelPos.x * inpW + iPixelPos.y] * kernel[coefPos.x * kerW + coefPos.y];
                }
            }

            // Even
        } else if ( pick == 2 ) {
            for ( int k = -S0; k < S0; k++ ) {
                for ( int l = -S0; l < S0; l++ ) {
                    int2 iPixelPos { ( i + k ), ( j + l ) };  // iPixelPos[1], [0]
                    int2 coefPos { ( k + S0 ), ( l + S0 ) };
                    temp += inp[iPixelPos.x * inpW + iPixelPos.y] * kernel[coefPos.x * kerW + coefPos.y];
                }
            }

            // Non-squares
        } else {
            for ( int k = 0; k < S0; k++ ) {
                for ( int l = 0; l < S1; l++ ) {
                    int2 iPixelPos { ( i + k - S1 ), ( j + l - S0 ) };
                    int2 coefPos { k, l };
                    temp += inp[iPixelPos.x * inpW + iPixelPos.y] * kernel[coefPos.x * kerH + coefPos.y];
                }
            }
        }
        out[oPixelPos.x * outW + oPixelPos.y] = temp;
    }
}
```

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:437` 至 `:449`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:425` 至 `:436`。

```python
template<typename T>
__device__ void _cupy_correlate2D( const T *__restrict__ inp,
                                   const int inpW,
                                   const int inpH,
                                   const T *__restrict__ kernel,
                                   const int kerW,
                                   const int kerH,
                                   const int S0,
                                   const int S1,
                                   T *__restrict__ out,
                                   const int outW,
                                   const int outH,
```

逐行对应说明：

- 第 1 行：`template<typename T>` 声明类型模板。源码没有在 `template` 与 `<` 间留空格。
- 第 2 行：`__device__` 表明函数只能从 GPU 代码调用；输入指针是只读且 `__restrict__` 声明不与别的指针别名。
- 第 3 行：`inpW` 是 padding 后输入宽度，用作行主序步长。
- 第 4 行：`inpH` 是输入高度；核心函数签名接收但函数体没有直接使用。
- 第 5 行：`kernel` 指向已共轭模板，类型与输入相同。
- 第 6 行：`kerW` 接收 Python wrapper 的 `d_kernel.shape[0]`，命名与通常“宽=shape[1]”习惯相反。
- 第 7 行：`kerH` 接收 `shape[1]`。
- 第 8 行：`S0`：方形半径或非方形第一维长度。
- 第 9 行：`S1`：非方形第二维长度。
- 第 10 行：`out` 是可写输出指针。
- 第 11 行：`outW` 是输出行主序步长。
- 第 12 行：`outH` 用于线程边界检查。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:450` 至 `:450`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:437` 至 `:437`。

```python
                                   const int pick ) {
```

逐行对应说明：

- 第 1 行：`pick` 决定索引分支；右花括号前的 `{` 开始函数体。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:453` 至 `:455`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:439` 至 `:440`。

```python
    const int ty { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int tx { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };
```

逐行对应说明：

- 第 1 行：`ty` 由 grid/block 的 x 轴得到，实际用作输出列。花括号是 C++ 列表初始化。
- 第 2 行：`tx` 由 y 轴得到，实际用作输出行。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:457` 至 `:457`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:442` 至 `:442`。

```python
    int i {};
```

逐行对应说明：

- 第 1 行：值初始化 `i` 为 0。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:458` 至 `:463`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:443` 至 `:448`。

```python
    if ( pick != 3 ) {
        i = tx + S0;
    } else {
        i = tx + S1;
    }
    int j { ty + S0 };
```

逐行对应说明：

- 第 1 行：方形分支（pick 1 或 2）。
- 第 2 行：输入中心行设为输出行加 `S0`。
- 第 3 行：非方形分支。
- 第 4 行：输入中心/起算行设为输出行加 `S1`。这进一步反映 `S0/S1` 的轴命名不直观。
- 第 5 行：结束 if/else。
- 第 6 行：输入中心列 `j=ty+S0`。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:465` 至 `:465`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:450` 至 `:450`。

```python
    int2 oPixelPos { tx, ty };
```

逐行对应说明：

- 第 1 行：用 CUDA `int2` 保存输出 `(row,column)`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:467` 至 `:469`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:451` 至 `:452`。

```python
    if ( ( tx < outH ) && ( ty < outW ) ) {
        T temp {};
```

逐行对应说明：

- 第 1 行：丢弃落在输出高或宽之外的尾部线程。
- 第 2 行：`T temp {}` 值初始化为该类型的零，作为局部累加器。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:471` 至 `:471`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:454` 至 `:454`。

```python
        // Odd
```

逐行对应说明：

- 第 1 行：原注释：奇数方形分支。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:472` 至 `:472`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:455` 至 `:455`。

```python
        if ( pick == 1 ) {
```

逐行对应说明：

- 第 1 行：`pick==1`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:473` 至 `:473`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:456` 至 `:456`。

```python
            for ( int k = -S0; k < ( S0 + 1 ); k++ ) {
```

逐行对应说明：

- 第 1 行：外层模板行偏移从 $-S_0$ 到 $S_0$（含），共 $2S_0+1$ 次。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:474` 至 `:480`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:457` 至 `:462`。

```python
                for ( int l = -S0; l < ( S0 + 1 ); l++ ) {
                    int2 iPixelPos { ( i + k ), ( j + l ) };
                    int2 coefPos { ( k + S0 ), ( l + S0 ) };
                    temp += inp[iPixelPos.x * inpW + iPixelPos.y] * kernel[coefPos.x * kerW + coefPos.y];
                }
            }
```

逐行对应说明：

- 第 1 行：内层列偏移同范围。
- 第 2 行：计算当前输入二维坐标。
- 第 3 行：把有符号偏移转换成从 0 开始的模板坐标。
- 第 4 行：两个二维坐标按行主序展平，乘积累加到 `temp`。Python 已先共轭模板，所以这里不调用 `conj`。
- 第 5 行：结束内层循环。
- 第 6 行：结束外层循环。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:482` 至 `:483`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:464` 至 `:465`。

```python
            // Even
        } else if ( pick == 2 ) {
```

逐行对应说明：

- 第 1 行：原注释：偶数方形。它缩进在奇数分支末尾，但注释不影响控制流。
- 第 2 行：`else if pick==2`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:484` 至 `:484`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:466` 至 `:466`。

```python
            for ( int k = -S0; k < S0; k++ ) {
```

逐行对应说明：

- 第 1 行：外层从 $-S_0$ 到 $S_0-1$，共 $2S_0$ 次。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:485` 至 `:491`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:467` 至 `:472`。

```python
                for ( int l = -S0; l < S0; l++ ) {
                    int2 iPixelPos { ( i + k ), ( j + l ) };  // iPixelPos[1], [0]
                    int2 coefPos { ( k + S0 ), ( l + S0 ) };
                    temp += inp[iPixelPos.x * inpW + iPixelPos.y] * kernel[coefPos.x * kerW + coefPos.y];
                }
            }
```

逐行对应说明：

- 第 1 行：内层相同。
- 第 2 行：计算输入坐标；行尾注释是原源码留下的轴顺序提醒。
- 第 3 行：计算模板坐标。
- 第 4 行：与奇数分支相同的直接乘加。
- 第 5 行：结束内层循环。
- 第 6 行：结束外层循环。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:493` 至 `:494`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:474` 至 `:475`。

```python
            // Non-squares
        } else {
```

逐行对应说明：

- 第 1 行：原注释：非方形模板。
- 第 2 行：其余 `pick` 都进入这里；正常只有 3。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:495` 至 `:495`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:476` 至 `:476`。

```python
            for ( int k = 0; k < S0; k++ ) {
```

逐行对应说明：

- 第 1 行：`k` 遍历 `[0,S0)`，即模板第一维。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:496` 至 `:507`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:477` 至 `:486`。

```python
                for ( int l = 0; l < S1; l++ ) {
                    int2 iPixelPos { ( i + k - S1 ), ( j + l - S0 ) };
                    int2 coefPos { k, l };
                    temp += inp[iPixelPos.x * inpW + iPixelPos.y] * kernel[coefPos.x * kerH + coefPos.y];
                }
            }
        }
        out[oPixelPos.x * outW + oPixelPos.y] = temp;
    }
}
```

逐行对应说明：

- 第 1 行：`l` 遍历 `[0,S1)`，即模板第二维。
- 第 2 行：计算输入坐标。代入 `i=tx+S1`、`j=ty+S0` 后得到 `(tx+k,ty+l)`，因此实质仍是从输出位置开始取窗口。
- 第 3 行：模板坐标就是 `(k,l)`。
- 第 4 行：输入仍按 `inpW` 展平；模板却按 `coefPos.x * kerH + coefPos.y` 展平。由于 `kerH` 接收 `shape[1]`，数值上它恰是行主序所需列步长，虽然变量命名反常。因此这是命名混淆，不应仅凭名称断定 bug。
- 第 5 行：结束内层循环。
- 第 6 行：结束外层循环。
- 第 7 行：结束 `pick` 分支。
- 第 8 行：按 `row*outW+column` 写回该线程唯一负责的输出元素。
- 第 9 行：结束线程合法性 if。
- 第 10 行：结束 device 函数。


> 复核说明：阶段一曾把 `kerH` 行步长标成潜在风险。沿 Python wrapper 的真实传参继续追踪后，`kerH=d_kernel.shape[1]`，数值上正是行主序列数，因此乘法本身正确；真正的问题是 `kerW/kerH` 命名与通常宽高含义相反，容易误读。这里以完整调用链证据修正阶段一的初步风险判断。

### 8.2 六种 dtype 的完整导出入口

`ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:488`

```cpp
extern "C" __global__ void __launch_bounds__( 256 ) _cupy_correlate2D_int32( const int *__restrict__ inp,
                                                                             const int inpW,
                                                                             const int inpH,
                                                                             const int *__restrict__ kernel,
                                                                             const int kerW,
                                                                             const int kerH,
                                                                             const int S0,
                                                                             const int S1,
                                                                             int *__restrict__ out,
                                                                             const int outW,
                                                                             const int outH,
                                                                             const int pick ) {
    _cupy_correlate2D<int>( inp, inpW, inpH, kernel, kerW, kerH, S0, S1, out, outW, outH, pick );
}

extern "C" __global__ void __launch_bounds__( 256 ) _cupy_correlate2D_int64( const long int *__restrict__ inp,
                                                                             const int inpW,
                                                                             const int inpH,
                                                                             const long int *__restrict__ kernel,
                                                                             const int kerW,
                                                                             const int kerH,
                                                                             const int S0,
                                                                             const int S1,
                                                                             long int *__restrict__ out,
                                                                             const int outW,
                                                                             const int outH,
                                                                             const int pick ) {
    _cupy_correlate2D<long int>( inp, inpW, inpH, kernel, kerW, kerH, S0, S1, out, outW, outH, pick );
}

extern "C" __global__ void __launch_bounds__( 256 ) _cupy_correlate2D_float32( const float *__restrict__ inp,
                                                                               const int inpW,
                                                                               const int inpH,
                                                                               const float *__restrict__ kernel,
                                                                               const int kerW,
                                                                               const int kerH,
                                                                               const int S0,
                                                                               const int S1,
                                                                               float *__restrict__ out,
                                                                               const int outW,
                                                                               const int outH,
                                                                               const int pick ) {
    _cupy_correlate2D<float>( inp, inpW, inpH, kernel, kerW, kerH, S0, S1, out, outW, outH, pick );
}

extern "C" __global__ void __launch_bounds__(256 ) _cupy_correlate2D_float64( const double *__restrict__ inp,
                                                                               const int inpW,
                                                                               const int inpH,
                                                                               const double *__restrict__ kernel,
                                                                               const int kerW,
                                                                               const int kerH,
                                                                               const int S0,
                                                                               const int S1,
                                                                               double *__restrict__ out,
                                                                               const int outW,
                                                                               const int outH,
                                                                               const int pick ) {
    _cupy_correlate2D<double>( inp, inpW, inpH, kernel, kerW, kerH, S0, S1, out, outW, outH, pick );
}

extern "C" __global__ void __launch_bounds__(256 )
    _cupy_correlate2D_complex64( const thrust::complex<float> *__restrict__ inp,
                                 const int inpW,
                                 const int inpH,
                                 const thrust::complex<float> *__restrict__ kernel,
                                 const int kerW,
                                 const int kerH,
                                 const int S0,
                                 const int S1,
                                 thrust::complex<float> *__restrict__ out,
                                 const int outW,
                                 const int outH,
                                 const int pick ) {
    _cupy_correlate2D<thrust::complex<float>>( inp, inpW, inpH, kernel, kerW, kerH, S0, S1, out, outW, outH, pick );
}

extern "C" __global__ void __launch_bounds__( 256 )
    _cupy_correlate2D_complex128( const thrust::complex<double> *__restrict__ inp,
                                  const int inpW,
                                  const int inpH,
                                  const thrust::complex<double> *__restrict__ kernel,
                                  const int kerW,
                                  const int kerH,
                                  const int S0,
                                  const int S1,
                                  thrust::complex<double> *__restrict__ out,
                                  const int outW,
                                  const int outH,
                                  const int pick ) {
    _cupy_correlate2D<thrust::complex<double>>( inp, inpW, inpH, kernel, kerW, kerH, S0, S1, out, outW, outH, pick );
}
```

每个 wrapper 的非空行逐行对应如下；六组结构相同，但仍逐行列出，不以“其余类似”代替。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:510` 至 `:521`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:488` 至 `:499`。

```python
extern "C" __global__ void __launch_bounds__( 256 ) _cupy_correlate2D_int32( const int *__restrict__ inp,
                                                                             const int inpW,
                                                                             const int inpH,
                                                                             const int *__restrict__ kernel,
                                                                             const int kerW,
                                                                             const int kerH,
                                                                             const int S0,
                                                                             const int S1,
                                                                             int *__restrict__ out,
                                                                             const int outW,
                                                                             const int outH,
                                                                             const int pick ) {
```

逐行对应说明：

- 第 1 行：以 C linkage 导出 `int32` global kernel，`__launch_bounds__(256)` 与 Python 的 256-thread block 一致；开始 `inp` 参数。
- 第 2 行：`inpW`。
- 第 3 行：`inpH`。
- 第 4 行：`int` 模板指针。
- 第 5 行：`kerW`。
- 第 6 行：`kerH`。
- 第 7 行：`S0`。
- 第 8 行：`S1`。
- 第 9 行：`int` 输出指针。
- 第 10 行：`outW`。
- 第 11 行：`outH`。
- 第 12 行：`pick` 并开始函数体。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:523` 至 `:524`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:500` 至 `:501`。

```python
    _cupy_correlate2D<int>( inp, inpW, inpH, kernel, kerW, kerH, S0, S1, out, outW, outH, pick );
}
```

逐行对应说明：

- 第 1 行：调用 `_cupy_correlate2D<int>`，参数原样转发。
- 第 2 行：结束 int32 wrapper。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:527` 至 `:538`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:503` 至 `:514`。

```python
extern "C" __global__ void __launch_bounds__( 256 ) _cupy_correlate2D_int64( const long int *__restrict__ inp,
                                                                             const int inpW,
                                                                             const int inpH,
                                                                             const long int *__restrict__ kernel,
                                                                             const int kerW,
                                                                             const int kerH,
                                                                             const int S0,
                                                                             const int S1,
                                                                             long int *__restrict__ out,
                                                                             const int outW,
                                                                             const int outH,
                                                                             const int pick ) {
```

逐行对应说明：

- 第 1 行：导出 `int64` wrapper；C++ 侧使用 `long int`。
- 第 2 行：int64 的 `inpW`。
- 第 3 行：`inpH`。
- 第 4 行：`long int` 模板指针。
- 第 5 行：`kerW`。
- 第 6 行：`kerH`。
- 第 7 行：`S0`。
- 第 8 行：`S1`。
- 第 9 行：`long int` 输出指针。
- 第 10 行：`outW`。
- 第 11 行：`outH`。
- 第 12 行：`pick` 并开始函数体。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:539` 至 `:540`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:515` 至 `:516`。

```python
    _cupy_correlate2D<long int>( inp, inpW, inpH, kernel, kerW, kerH, S0, S1, out, outW, outH, pick );
}
```

逐行对应说明：

- 第 1 行：调用 `_cupy_correlate2D<long int>`。
- 第 2 行：结束 int64 wrapper。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:543` 至 `:554`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:518` 至 `:529`。

```python
extern "C" __global__ void __launch_bounds__( 256 ) _cupy_correlate2D_float32( const float *__restrict__ inp,
                                                                               const int inpW,
                                                                               const int inpH,
                                                                               const float *__restrict__ kernel,
                                                                               const int kerW,
                                                                               const int kerH,
                                                                               const int S0,
                                                                               const int S1,
                                                                               float *__restrict__ out,
                                                                               const int outW,
                                                                               const int outH,
                                                                               const int pick ) {
```

逐行对应说明：

- 第 1 行：导出 `float32` wrapper，元素类型为 `float`。
- 第 2 行：`inpW`。
- 第 3 行：`inpH`。
- 第 4 行：`float` 模板指针。
- 第 5 行：`kerW`。
- 第 6 行：`kerH`。
- 第 7 行：`S0`。
- 第 8 行：`S1`。
- 第 9 行：`float` 输出指针。
- 第 10 行：`outW`。
- 第 11 行：`outH`。
- 第 12 行：`pick` 并开始函数体。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:555` 至 `:556`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:530` 至 `:531`。

```python
    _cupy_correlate2D<float>( inp, inpW, inpH, kernel, kerW, kerH, S0, S1, out, outW, outH, pick );
}
```

逐行对应说明：

- 第 1 行：调用 `_cupy_correlate2D<float>`。
- 第 2 行：结束 float32 wrapper。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:559` 至 `:570`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:533` 至 `:544`。

```python
extern "C" __global__ void __launch_bounds__(256 ) _cupy_correlate2D_float64( const double *__restrict__ inp,
                                                                               const int inpW,
                                                                               const int inpH,
                                                                               const double *__restrict__ kernel,
                                                                               const int kerW,
                                                                               const int kerH,
                                                                               const int S0,
                                                                               const int S1,
                                                                               double *__restrict__ out,
                                                                               const int outW,
                                                                               const int outH,
                                                                               const int pick ) {
```

逐行对应说明：

- 第 1 行：导出 `float64` wrapper，元素类型为 `double`；宏括号空格与前面略有不同但语义相同。
- 第 2 行：`inpW`。
- 第 3 行：`inpH`。
- 第 4 行：`double` 模板指针。
- 第 5 行：`kerW`。
- 第 6 行：`kerH`。
- 第 7 行：`S0`。
- 第 8 行：`S1`。
- 第 9 行：`double` 输出指针。
- 第 10 行：`outW`。
- 第 11 行：`outH`。
- 第 12 行：`pick` 并开始函数体。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:571` 至 `:572`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:545` 至 `:546`。

```python
    _cupy_correlate2D<double>( inp, inpW, inpH, kernel, kerW, kerH, S0, S1, out, outW, outH, pick );
}
```

逐行对应说明：

- 第 1 行：调用 `_cupy_correlate2D<double>`。
- 第 2 行：结束 float64 wrapper。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:575` 至 `:586`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:548` 至 `:559`。

```python
extern "C" __global__ void __launch_bounds__(256 )
    _cupy_correlate2D_complex64( const thrust::complex<float> *__restrict__ inp,
                                 const int inpW,
                                 const int inpH,
                                 const thrust::complex<float> *__restrict__ kernel,
                                 const int kerW,
                                 const int kerH,
                                 const int S0,
                                 const int S1,
                                 thrust::complex<float> *__restrict__ out,
                                 const int outW,
                                 const int outH,
```

逐行对应说明：

- 第 1 行：开始 complex64 导出声明，函数名移到下一行。
- 第 2 行：函数名及 `thrust::complex<float>` 输入指针。
- 第 3 行：`inpW`。
- 第 4 行：`inpH`。
- 第 5 行：complex64 模板指针。
- 第 6 行：`kerW`。
- 第 7 行：`kerH`。
- 第 8 行：`S0`。
- 第 9 行：`S1`。
- 第 10 行：complex64 输出指针。
- 第 11 行：`outW`。
- 第 12 行：`outH`。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:587` 至 `:589`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:560` 至 `:562`。

```python
                                 const int pick ) {
    _cupy_correlate2D<thrust::complex<float>>( inp, inpW, inpH, kernel, kerW, kerH, S0, S1, out, outW, outH, pick );
}
```

逐行对应说明：

- 第 1 行：`pick` 并开始函数体。
- 第 2 行：调用 `_cupy_correlate2D<thrust::complex<float>>`。共轭已在 Python 完成，device 只执行复乘加。
- 第 3 行：结束 complex64 wrapper。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:592` 至 `:603`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:564` 至 `:575`。

```python
extern "C" __global__ void __launch_bounds__( 256 )
    _cupy_correlate2D_complex128( const thrust::complex<double> *__restrict__ inp,
                                  const int inpW,
                                  const int inpH,
                                  const thrust::complex<double> *__restrict__ kernel,
                                  const int kerW,
                                  const int kerH,
                                  const int S0,
                                  const int S1,
                                  thrust::complex<double> *__restrict__ out,
                                  const int outW,
                                  const int outH,
```

逐行对应说明：

- 第 1 行：开始 complex128 导出声明。
- 第 2 行：函数名及 `thrust::complex<double>` 输入指针。
- 第 3 行：`inpW`。
- 第 4 行：`inpH`。
- 第 5 行：complex128 模板指针。
- 第 6 行：`kerW`。
- 第 7 行：`kerH`。
- 第 8 行：`S0`。
- 第 9 行：`S1`。
- 第 10 行：complex128 输出指针。
- 第 11 行：`outW`。
- 第 12 行：`outH`。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:604` 至 `:606`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/convolution/_convolution.cu:576` 至 `:578`。

```python
                                  const int pick ) {
    _cupy_correlate2D<thrust::complex<double>>( inp, inpW, inpH, kernel, kerW, kerH, S0, S1, out, outW, outH, pick );
}
```

逐行对应说明：

- 第 1 行：`pick` 并开始函数体。
- 第 2 行：调用 `_cupy_correlate2D<thrust::complex<double>>`。
- 第 3 行：结束 complex128 wrapper。


## 9. 调用链与算法总结

```text
cusignal.correlate2d
  → cp.asarray(in1), cp.asarray(in2)
  → _inputs_swap_needed（仅 valid 可能交换）
  → in2.conj()
  → _convolve2d
      → _valfrommode / _bvalfromboundary
      → cp.promote_types + astype
      → 计算输出 shape、cp.empty
      → _convolve2d_gpu
          → 计算 pick、S0/S1、四侧 padding
          → cp.pad（same/full；valid 不 pad）
          → 计算 16×16 block 和二维 grid
          → _populate_kernel_cache
              → _get_function(fatbin, dtype 符号)
          → _get_backend_kernel
              → _cupy_convolve_2d_wrapper
          → _cupy_correlate2D_<dtype> global kernel
              → _cupy_correlate2D<T> device 核心
                  → 每线程一个输出位置
                  → 模板双重循环
                  → 对应元素直接乘加
  → valid 若交换过输入：out[::-1, ::-1]
  → 返回 CuPy ndarray
```

执行顺序的本质是：

1. Python 层负责 API 语义、二维约束、输入交换和第二输入共轭。
2. Python GPU 后端负责 dtype、输出 shape、边界扩展、kernel 查找和 launch geometry。
3. CUDA global dtype wrapper 只负责提供可从 fatbin 按名称取得的入口。
4. CUDA device 模板函数负责真正的二维直接乘加。

该实现没有调用 FFT、插值、卷积库或线性代数库；底层机制是 CuPy 数组管理、`cp.pad` 和预编译 CUDA fatbin。

## 10. 数学公式与代码映射

采用阶段一约定：

$$
R_{XH}[u,v]=\sum_{a=0}^{P-1}\sum_{b=0}^{Q-1}
\widetilde X[u+a,v+b]H^*[a,b].
$$

| 数学元素 | Python/CUDA 位置 | 对应关系 |
| --- | --- | --- |
| $H^*$ | `correlate.py:248` | `in2.conj()` 在 kernel 前一次性产生共轭模板。 |
| $\widetilde X$ | `_convolution_cuda.py:324–340` | `cp.pad` 把 `fill/wrap/symm` 变成常数、周期、对称延拓。 |
| 位移 $(u,v)$ | `_convolution.cu:439–450` | `tx,ty` 是一个输出位置，也就是一个离散位移存储位置。 |
| 模板索引 $(a,b)$ | `_convolution.cu:456–479` | `k,l` 双重循环覆盖每个模板元素。 |
| $X\,H^*$ | `_convolution.cu:460,470,480` | 展平后的输入元素乘以已共轭模板元素。 |
| 双重求和 | `T temp {}; temp += ...` | 每线程的局部变量从零累加整个模板。 |
| 输出 $R[u,v]$ | `_convolution.cu:484` | 累加值写回唯一输出位置。 |
| `full` shape | `_convolution_cuda.py:480–482` | 每维 $n_1+n_2-1$。 |
| `same` shape | `_convolution_cuda.py:477–479` | 每维等于第一输入。 |
| `valid` shape | `_convolution_cuda.py:469–471` | 每维 $n_1-n_2+1$。 |

### 10.1 方形与非方形分支的等价性

奇数和偶数方形分支以中心为基准，用有符号的 `k,l`；非方形分支从 0 开始遍历模板。它们的索引表达式不同，但都执行同一数学运算。`pick` 是工程索引分派，不是三种不同的相关算法。

特别注意参数命名链：Python 传 `shape[0]` 到 CUDA 的 `kerW`，传 `shape[1]` 到 `kerH`。因此非方形分支用 `kerH` 作模板行主序步长，在数值上等于列数 `shape[1]`，是正确步长；名称容易令人误以为宽高写反。

## 11. 边界、异常与数值风险

### 11.1 边界行为

- `valid` 不调用 `cp.pad`，所以 `boundary` 和 `fillvalue` 不影响正常数值结果，但边界字符串仍会被解析和校验。
- `same` 与 `full` 显式 pad 第一输入；模板保持不变。
- `fill` 采用 `cp.pad(..., "constant")`；`fillvalue` 必须恰有一个元素。
- `wrap` 采用周期扩展。
- `symm` 采用 `cp.pad(..., "symmetric")`，边缘样本参与镜像。
- 公开 docstring 只列三种名称，但 `_boundarydict` 还含 `pad/circular/symmetric/reflect`。其中前三个别名能映射到 0/8/4；`reflect` 经过左移得到 16，随后被后端拒绝，因此不能视为当前路径的有效别名。

### 11.2 异常检查

- 非二维输入：公开函数抛 `ValueError`。
- `valid` 两个 shape 交叉占优：`_inputs_swap_needed` 抛 `ValueError`。
- 未知 mode/boundary：映射 helper 抛 `ValueError`。
- 非标量 `fillvalue`：后端抛普通 `Exception`。
- 不支持 dtype：kernel cache 抛 `ValueError`。

### 11.3 dtype 与数值稳定性

- 两输入先提升到共同 dtype，输出和 CUDA 累加器都使用该 dtype。
- `int32/int64` 乘加没有扩大累加类型，存在整数溢出风险。
- `float32/float64` 采用固定循环顺序的朴素累加，没有 Kahan 补偿、pairwise reduction 或更高精度 accumulator；模板越大，舍入误差越可能积累。
- `complex64/complex128` 的实部和虚部同样按对应精度累加。
- 没有归一化，输出幅度随模板面积、输入幅度和边界重叠方式变化。

### 11.4 `valid` 交换在复数输入下的风险

互相关交换关系是：

$$
R_{HX}[u,v]=R_{XH}^*[-u,-v].
$$

源码在交换输入后只执行 `out[::-1, ::-1]`，没有对结果再取共轭。因此：

- 实数输入时反转足够；
- 复数输入且 `valid` 触发交换时，按标准关系还应考虑共轭；当前代码可能返回目标结果的复共轭。

这是从源码和数学关系推出的风险，本阶段没有运行测试验证，不能把它表述为已经由实验复现的失败。

## 12. 时间复杂度、空间复杂度与性能瓶颈

设输出尺寸为 $O_y\times O_x$，模板尺寸为 $P\times Q$。

- CUDA 核心总工作量：$\Theta(O_yO_xPQ)$。
- 单线程工作量：$\Theta(PQ)$。
- 输出空间：$\Theta(O_yO_x)$。
- `same/full` 的显式 padded 输入：额外 $\Theta((M+p_y)(N+p_x))$ GPU 空间。
- `in2.conj()`、`astype` 可能产生额外 GPU 临时数组。
- 每个输出线程顺序遍历整个模板；模板大时单线程循环很长。
- 相邻输出线程读取高度重叠的输入窗口，但源码没有显式 shared-memory tile；缓存命中取决于硬件缓存行为。
- 每个输出都重复读取模板，没有在源码中把模板放入 constant/shared memory。
- 不存在 direct/FFT 自动选择，大模板、大图像时 $PQ$ 因子可能成为主要瓶颈。
- 首次某 dtype 调用还包含 fatbin module 加载和符号查询开销，后续由 cache 避免重复加载。

## 13. 源码阅读顺序与观察问题

建议严格按以下顺序亲自阅读：

1. `correlate.py:158–258`：先找出 API 做了哪些事、没有做哪些事。重点观察 `in2.conj()`、`0` 和交换后反转。
2. `convolution_utils.py:40–70`：用几个 shape 手算 `ok1/ok2`，特别是 `(3,5)` 与 `(4,4)` 为什么报错。
3. `_convolution_cuda.py:441–499`：手算三种 mode 的输出 shape，并跟踪 dtype。
4. `_convolution_cuda.py:268–384`：分别代入 $3\times3$、$4\times4$、$3\times4$ 模板，计算 `pick/S/P1...P4`。
5. `_cupy_convolve_2d_wrapper:142–182`：逐项将 Python 参数与 CUDA 形参对齐，理解反常的 `kerW/kerH` 命名。
6. `_convolution.cu:425–486`：选一个输出线程，手算 `tx,ty,i,j,k,l` 与展平地址。
7. `_convolution.cu:488–578`：确认 dtype wrapper 不含算法，只做模板实例转发。

### 13.1 自检问题

1. 为什么 `in2.conj()` 在 Python 做一次，而不是在 CUDA 最内层循环中反复做？
2. `use_convolve=0` 经过哪两个分支最终变成 `_cupy_correlate2D_float32` 之类的符号？
3. 为什么 `valid` 不需要 pad？此时 boundary 参数还有什么作用？
4. 偶数模板为什么相关和卷积的 padding 方向相反？
5. `blockspergrid` 为什么把输出 `shape[1]` 放在第一维？它怎样对应 `ty`？
6. 非方形分支中 `kerH` 为什么数值上是正确的行主序步长？
7. 哪些语句证明没有 FFT 和归一化？
8. 复数 `valid` 输入交换时为什么可能缺少一次结果共轭？

### 13.2 参考答案

1. `in2.conj()` 在 `correlate.py` 基准第 248 行一次性生成共轭数组，CUDA 最内层只需乘加；若在模板双循环内逐元素反复调用共轭，会把同一模板元素的转换成本重复到每个输出位置。
2. `correlate.py:249` 的 `use_convolve=0` 先让 `_convolve2d_gpu` 在基准第 368–378 行选择 `k_type="correlate2D"`，再让 `_populate_kernel_cache` 拼出 `_cupy_correlate2D_<dtype>`，最后 `_get_backend_kernel` 返回二维 wrapper。
3. `valid` 只保留模板完全落入输入的窗口，所以所有索引都在数组内，无需虚拟边界。`boundary` 仍会在 `_bvalfromboundary` 中接受或拒绝字符串，但不进入 `cp.pad`，`fillvalue` 也不参与结果。
4. 偶数模板没有唯一整数中心。`_convolve2d_gpu:298–302` 让卷积与相关分别把多出的一个 padding 样点放在相反侧，以维持各自翻转/不翻转的 `same` 对齐约定。
5. `_convolve2d_gpu:351–355` 用输出列数计算 grid x；CUDA `_cupy_correlate2D:439` 又用 block/thread x 得到 `ty`，而 `ty` 最终作为输出列。因此两个层次的轴映射一致，只是变量名不直观。
6. Python wrapper 的 `d_kernel.shape[1]` 传入 CUDA 形参 `kerH`；对 row-major 矩阵，展平行步长正是列数 `shape[1]`。所以非方形分支的 `coefPos.x * kerH + coefPos.y` 数值正确，问题只在宽高命名反常。
7. `correlate2d` 固定传 `use_convolve=0`，后端只加载 `correlate2D` fatbin kernel；调用链没有 `fft2/ifft2`、频谱乘法、局部均值、范数或除法，证明既没有 FFT 路线也没有 NCC/ZNCC。
8. 标准交换关系为 $R_{HX}[u,v]=R_{XH}^*[-u,-v]$。公开函数在基准第 255–256 行交换后只执行双轴反转，没有再取共轭，所以复数且 `valid` 由第二输入逐维支配时存在返回目标结果共轭值的源码风险；实数输入不受影响。

## 14. 完整性与验证说明

- 已完整摘录公开导出、完整签名、完整 docstring、全部 docstring 示例、函数体、输入交换 helper、模式/边界/grid helper、kernel cache、二维 wrapper、fatbin loader、调试 helper、GPU 调度函数、CUDA device 核心和六种 dtype 入口。
- 每一行非空摘录均在对应逐行表中解释；空行仅保留在源码块中，不单独解释。
- 学习副本只插入独立的 `# <学习注释：...>` 或 `// <学习注释：...>` 行，没有改写原代码、原注释或顺序。
- `ZKX/cusignal-23.08.00` 只读基准未修改。
- 按项目规则，本阶段没有在本地运行 Python、CUDA 或测试，也没有连接 ZQ500；所有运行时结论均明确限定为源码推导。
