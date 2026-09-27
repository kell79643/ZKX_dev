# upfirdn 算子学习 — Python 源码算法

## 一、代码定位与接口概览

### 1.1 公开导入路径

```python
from cusignal import upfirdn
```

### 1.2 定义文件与符号

| 项目 | 值 |
|------|-----|
| 模块名 | `cusignal.filtering.resample` |
| 定义文件（学习副本） | `Learning/cusignal-23.08.00/python/cusignal/filtering/resample.py` |
| 定义文件（只读基准） | `ZKX/cusignal-23.08.00/python/cusignal/filtering/resample.py` |
| 公开函数 | `upfirdn()` |
| 行号（只读基准） | 448-539 |
| 核心实现类 | `_UpFIRDn` |
| 核心实现文件 | `Learning/cusignal-23.08.00/python/cusignal/filtering/_upfirdn_cuda.py` |
| 核心实现行号（只读基准） | 164-261 |

### 1.3 函数签名

```python
def upfirdn(
    h,
    x,
    up=1,
    down=1,
    axis=-1,
):
```

### 1.4 参数语义

| 参数 | 类型 | 默认值 | 语义 |
|------|------|--------|------|
| `h` | array_like | - | 一维 FIR 滤波器系数数组 |
| `x` | array_like | - | 输入信号数组（可多维） |
| `up` | int | 1 | 上采样因子 $L$ |
| `down` | int | 1 | 下采样因子 $M$ |
| `axis` | int | -1 | 滤波轴（默认最后一个轴） |

### 1.5 返回值

| 项目 | 说明 |
|------|------|
| 类型 | `ndarray` |
| dtype | `pp.result_type(h.dtype, x.dtype, pp.float32)` |
| shape | 与输入 `x` 相同，但 `axis` 轴长度变化 |
| 长度 | $\lfloor ((N_x - 1)L + N_h - 1) / M \rfloor + 1$ |

---

## 二、当前算子的完整相关源码

### 2.1 公开 API 函数定义（resample.py:448-539）

**学习副本路径**：`Learning/cusignal-23.08.00/python/cusignal/filtering/resample.py:448-539`

**只读基准路径**：`ZKX/cusignal-23.08.00/python/cusignal/filtering/resample.py:448-539`

```python
def upfirdn(
    h,
    x,
    up=1,
    down=1,
    axis=-1,
):
    """
    Upsample, FIR filter, and downsample.

    Parameters
    ----------
    h : array_like
        1-dimensional FIR (finite-impulse response) filter coefficients.
    x : array_like
        Input signal array.
    up : int, optional
        Upsampling rate. Default is 1.
    down : int, optional
        Downsampling rate. Default is 1.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis. Default is -1.

    Returns
    -------
    y : ndarray
        The output signal array. Dimensions will be the same as `x` except
        for along `axis`, which will change size according to the `h`,
        `up`,  and `down` parameters.

    Notes
    -----
    The algorithm is an implementation of the block diagram shown on page 129
    of the Vaidyanathan text [1]_ (Figure 4.3-8d).

    The direct approach of upsampling by factor of P with zero insertion,
    FIR filtering of length ``N``, and downsampling by factor of Q is
    O(N*Q) per output sample. The polyphase implementation used here is
    O(N/P).

    Examples
    --------
    Simple operations:
    >>> from cusignal import upfirdn
    >>> upfirdn([1, 1, 1], [1, 1, 1])   # FIR filter
    array([ 1.,  2.,  3.,  2.,  1.])
    >>> upfirdn([1], [1, 2, 3], 3)  # upsampling with zeros insertion
    array([ 1.,  0.,  0.,  2.,  0.,  0.,  3.,  0.,  0.])
    >>> upfirdn([1, 1, 1], [1, 2, 3], 3)  # upsampling with sample-and-hold
    array([ 1.,  1.,  1.,  2.,  2.,  2.,  3.,  3.,  3.])
    >>> upfirdn([.5, 1, .5], [1, 1, 1], 2)  # linear interpolation
    array([ 0.5,  1. ,  1. ,  1. ,  1. ,  1. ,  0.5,  0. ])
    >>> upfirdn([1], cp.arange(10), 1, 3)  # decimation by 3
    array([ 0.,  3.,  6.,  9.])
    >>> upfirdn([.5, 1, .5], cp.arange(10), 2, 3)  # linear interp, rate 2/3
    array([ 0. ,  1. ,  2.5,  4. ,  5.5,  7. ,  8.5,  0. ])
    Apply a single filter to multiple signals:
    >>> x = cp.reshape(cp.arange(8), (4, 2))
    >>> x
    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7]])
    Apply along the last dimension of ``x``:
    >>> h = [1, 1]
    >>> upfirdn(h, x, 2)
    array([[ 0.,  0.,  1.,  1.],
           [ 2.,  2.,  3.,  3.],
           [ 4.,  4.,  5.,  5.],
           [ 6.,  6.,  7.,  7.]])
    Apply along the 0th dimension of ``x``:
    >>> upfirdn(h, x, 2, axis=0)
    array([[ 0.,  1.],
           [ 0.,  1.],
           [ 2.,  3.],
           [ 2.,  3.],
           [ 4.,  5.],
           [ 4.,  5.],
           [ 6.,  7.],
           [ 6.,  7.]])

    References
    ----------
    .. [1] P. P. Vaidyanathan, Multirate Systems and Filter Banks,
       Prentice Hall, 1993.
    """

    ufd = _UpFIRDn(h, x.dtype, up, down)
    # This is equivalent to (but faster than) using cp.apply_along_axis
    return ufd.apply_filter(x, axis)
```

### 2.2 核心实现类（_upfirdn_cuda.py:164-261）

**学习副本路径**：`Learning/cusignal-23.08.00/python/cusignal/filtering/_upfirdn_cuda.py:164-261`

**只读基准路径**：`ZKX/cusignal-23.08.00/python/cusignal/filtering/_upfirdn_cuda.py:164-261`

```python
class _UpFIRDn(object):
    def __init__(self, h, x_dtype, up, down):
        """Helper for resampling"""

        if isinstance(h, cp.ndarray):
            pp = cp
        else:
            pp = np

        h = pp.asarray(h)
        if h.ndim != 1 or h.size == 0:
            raise ValueError("h must be 1D with non-zero length")

        self._output_type = pp.result_type(h.dtype, x_dtype, pp.float32)
        h = pp.asarray(h, self._output_type)
        self._up = int(up)
        self._down = int(down)
        if self._up < 1 or self._down < 1:
            raise ValueError("Both up and down must be >= 1")
        # This both transposes, and "flips" each phase for filtering
        self._h_trans_flip = _pad_h(h, pp, self._up)
        self._h_trans_flip = cp.asarray(self._h_trans_flip)
        self._h_trans_flip = cp.ascontiguousarray(self._h_trans_flip)
        self._h_len_orig = len(h)

    def apply_filter(
        self,
        x,
        axis,
    ):
        """Apply the prepared filter to the specified axis of a nD signal x"""

        x = cp.asarray(x, self._output_type)

        output_len = _output_len(self._h_len_orig, x.shape[axis], self._up, self._down)
        output_shape = list(x.shape)
        output_shape[axis] = output_len
        out = cp.empty(output_shape, dtype=self._output_type, order="C")
        axis = axis % x.ndim

        # Precompute variables on CPU
        x_shape_a = x.shape[axis]
        h_per_phase = len(self._h_trans_flip) // self._up
        padded_len = x.shape[axis] + (len(self._h_trans_flip) // self._up) - 1

        if out.ndim > 1:
            threadsperblock = (8, 8)
            blocks = ceil(out.shape[0] / threadsperblock[0])
            blockspergrid_x = blocks if blocks < _get_max_gdx() else _get_max_gdx()

            blocks = ceil(out.shape[1] / threadsperblock[1])
            blockspergrid_y = blocks if blocks < _get_max_gdy() else _get_max_gdy()

            blockspergrid = (blockspergrid_x, blockspergrid_y)

        else:
            threadsperblock, blockspergrid = _get_tpb_bpg()

        if out.ndim == 1:
            k_type = "upfirdn1D"

            _populate_kernel_cache(out.dtype, k_type)

            kernel = _get_backend_kernel(
                out.dtype,
                blockspergrid,
                threadsperblock,
                k_type,
            )
        elif out.ndim == 2:
            k_type = "upfirdn2D"

            _populate_kernel_cache(out.dtype, k_type)

            kernel = _get_backend_kernel(
                out.dtype,
                blockspergrid,
                threadsperblock,
                k_type,
            )
        else:
            raise NotImplementedError("upfirdn() requires ndim <= 2")

        kernel(
            x,
            self._h_trans_flip,
            self._up,
            self._down,
            axis,
            x_shape_a,
            h_per_phase,
            padded_len,
            out,
        )

        _print_atts(kernel)

        return out
```

### 2.3 必要的直接依赖

#### 2.3.1 _pad_h 函数（_upfirdn_cuda.py:31-44）

```python
def _pad_h(h, pp, up):
    """Store coefficients in a transposed, flipped arrangement.
    For example, suppose upRate is 3, and the
    input number of coefficients is 10, represented as h[0], ..., h[9].
    Then the internal buffer will look like this::
       h[9], h[6], h[3], h[0],   // flipped phase 0 coefs
       0,    h[7], h[4], h[1],   // flipped phase 1 coefs (zero-padded)
       0,    h[8], h[5], h[2],   // flipped phase 2 coefs (zero-padded)
    """
    h_padlen = len(h) + (-len(h) % up)
    h_full = pp.zeros(h_padlen, h.dtype)
    h_full[: len(h)] = h
    h_full = h_full.reshape(-1, up).T[:, ::-1].ravel()
    return h_full
```

#### 2.3.2 _output_len 函数（_upfirdn_cuda.py:47-48）

```python
def _output_len(len_h, in_len, up, down):
    return (((in_len - 1) * up + len_h) - 1) // down + 1
```

#### 2.3.3 _populate_kernel_cache 函数（_upfirdn_cuda.py:133-144）

```python
def _populate_kernel_cache(np_type, k_type):

    if np_type not in _SUPPORTED_TYPES:
        raise ValueError("Datatype {} not found for '{}'".format(np_type, k_type))

    if (str(np_type), k_type) in _cupy_kernel_cache:
        return

    _cupy_kernel_cache[(str(np_type), k_type)] = _get_function(
        "/filtering/_upfirdn.fatbin",
        "_cupy_" + k_type + "_" + str(np_type),
    )
```

#### 2.3.4 _get_backend_kernel 函数（_upfirdn_cuda.py:147-161）

```python
def _get_backend_kernel(
    dtype,
    grid,
    block,
    k_type,
):

    kernel = _cupy_kernel_cache[(str(dtype), k_type)]
    if kernel:
        if k_type == "upfirdn1D":
            return _cupy_upfirdn_wrapper(grid, block, kernel)
        elif k_type == "upfirdn2D":
            return _cupy_upfirdn2d_wrapper(grid, block, kernel)
    else:
        raise ValueError("Kernel {} not found in _cupy_kernel_cache".format(k_type))
```

#### 2.3.5 Wrapper 类（_upfirdn_cuda.py:51-131）

```python
class _cupy_upfirdn_wrapper(object):
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
        x,
        h_trans_flip,
        up,
        down,
        axis,
        x_shape_a,
        h_per_phase,
        padded_len,
        out,
    ):

        kernel_args = (
            x,
            h_trans_flip,
            up,
            down,
            axis,
            x_shape_a,
            h_per_phase,
            padded_len,
            out,
            out.shape[0],
        )

        self.kernel(self.grid, self.block, kernel_args)


class _cupy_upfirdn2d_wrapper(object):
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
        x,
        h_trans_flip,
        up,
        down,
        axis,
        x_shape_a,
        h_per_phase,
        padded_len,
        out,
    ):

        kernel_args = (
            x,
            x.shape[1],
            h_trans_flip,
            up,
            down,
            axis,
            x_shape_a,
            h_per_phase,
            padded_len,
            out,
            out.shape[0],
            out.shape[1],
        )

        self.kernel(self.grid, self.block, kernel_args)
```

---

## 三、docstring 逐行翻译与解释

### 3.1 函数用途

**原文**：
```
Upsample, FIR filter, and downsample.
```

**翻译与解释**：
- 执行三个级联操作：上采样、FIR 滤波、下采样
- 这是多速率信号处理的核心原子操作
- 对应数学物理原理中的有理数倍采样率转换（原理 3）

### 3.2 参数说明

#### `h : array_like`

**原文**：
```
1-dimensional FIR (finite-impulse response) filter coefficients.
```

**翻译与解释**：
- 一维 FIR 滤波器系数数组
- 必须是非空的 1D 数组
- 对应原理中的低通滤波器设计（原理 7）

#### `x : array_like`

**原文**：
```
Input signal array.
```

**翻译与解释**：
- 输入信号数组，可支持多维
- 滤波只沿指定的 `axis` 进行
- 其他维度通过批量处理

#### `up : int, optional`

**原文**：
```
Upsampling rate. Default is 1.
```

**翻译与解释**：
- 上采样因子 $L$，必须为正整数
- 默认值为 1（不进行上采样）
- 对应原理中的插值倍数（原理 1）

#### `down : int, optional`

**原文**：
```
Downsampling rate. Default is 1.
```

**翻译与解释**：
- 下采样因子 $M$，必须为正整数
- 默认值为 1（不进行下采样）
- 对应原理中的抽取倍数（原理 2）

#### `axis : int, optional`

**原文**：
```
The axis of the input data array along which to apply the
linear filter. The filter is applied to each subarray along
this axis. Default is -1.
```

**翻译与解释**：
- 滤波轴，默认为最后一个轴（-1）
- 支持负索引（如 -1 表示最后一维）
- 沿该轴的所有子数组独立应用同一滤波器

### 3.3 返回值说明

**原文**：
```
y : ndarray
    The output signal array. Dimensions will be the same as `x` except
    for along `axis`, which will change size according to the `h`,
    `up`,  and `down` parameters.
```

**翻译与解释**：
- 返回重采样后的数组
- 维度数与输入 `x` 相同
- 只有 `axis` 轴的长度发生变化
- 输出长度公式：$\lfloor ((N_x - 1)L + N_h - 1) / M \rfloor + 1$

### 3.4 注释要点

**原文**：
```
The algorithm is an implementation of the block diagram shown on page 129
of the Vaidyanathan text [1]_ (Figure 4.3-8d).

The direct approach of upsampling by factor of P with zero insertion,
FIR filtering of length ``N``, and downsampling by factor of Q is
O(N*Q) per output sample. The polyphase implementation used here is
O(N/P).
```

**翻译与解释**：
- 实现了 Vaidyanathan 教科书第 129 页图 4.3-8d 的多相结构（原理 6）
- 直接实现的复杂度：$O(N \cdot Q)$ 每个输出样本
- 多相实现的复杂度：$O(N / P)$ 每个输出样本（原理 4）
- 效率提升源于避免对零值的无效计算

### 3.5 示例代码解读

#### 示例 1：纯 FIR 滤波

```python
>>> upfirdn([1, 1, 1], [1, 1, 1])   # FIR filter
array([ 1.,  2.,  3.,  2.,  1.])
```

**解释**：
- `up=1, down=1`：无采样率变换
- 仅执行 FIR 卷积：$[1, 1, 1] * [1, 1, 1]$
- 结果为卷积输出：1+1+1=3（中心）、1+1=2（两侧）、1（端点）

#### 示例 2：纯上采样（插零）

```python
>>> upfirdn([1], [1, 2, 3], 3)  # upsampling with zeros insertion
array([ 1.,  0.,  0.,  2.,  0.,  0.,  3.,  0.,  0.])
```

**解释**：
- `h=[1]`：单位冲激响应（直通滤波器）
- `up=3`：每个样本后插入 2 个零
- 结果：`[1, 0, 0, 2, 0, 0, 3, 0, 0]`

#### 示例 3：上采样 + 保持

```python
>>> upfirdn([1, 1, 1], [1, 2, 3], 3)  # upsampling with sample-and-hold
array([ 1.,  1.,  1.,  2.,  2.,  2.,  3.,  3.,  3.])
```

**解释**：
- `h=[1, 1, 1]`：移动平均滤波器（平滑零值）
- 效果：每个原始样本重复 3 次（sample-and-hold）
- 原理：插值滤波器填补零值位置

#### 示例 4：线性插值

```python
>>> upfirdn([.5, 1, .5], [1, 1, 1], 2)  # linear interpolation
array([ 0.5,  1. ,  1. ,  1. ,  1. ,  1. ,  0.5,  0. ])
```

**解释**：
- `h=[0.5, 1, 0.5]`：线性插值滤波器
- `up=2`：2 倍上采样
- 原理：三角窗滤波器实现线性插值

#### 示例 5：纯下采样（抽取）

```python
>>> upfirdn([1], cp.arange(10), 1, 3)  # decimation by 3
array([ 0.,  3.,  6.,  9.])
```

**解释**：
- `h=[1]`：直通滤波器
- `up=1, down=3`：仅执行 3 倍抽取
- 结果：保留索引 0, 3, 6, 9

#### 示例 6：有理数倍重采样

```python
>>> upfirdn([.5, 1, .5], cp.arange(10), 2, 3)  # linear interp, rate 2/3
array([ 0. ,  1. ,  2.5,  4. ,  5.5,  7. ,  8.5,  0. ])
```

**解释**：
- 采样率变化：$L/M = 2/3$
- 先 2 倍插值，线性插值滤波，再 3 倍抽取
- 应用：非整数倍采样率转换

---

## 四、源代码逐行解释

### 4.1 公开 API 函数体（resample.py:537-539）

#### 行 537：创建 _UpFIRDn 对象

```python
ufd = _UpFIRDn(h, x.dtype, up, down)
```

**语法说明**：
- `_UpFIRDn`：核心实现类（定义在 `_upfirdn_cuda.py`）
- 参数：`h`（滤波器系数）、`x.dtype`（输入数据类型）、`up`、`down`
- 返回：封装了预处理滤波器的对象

**数学映射**：
- 对应原理 4（多相分解）：构造阶段
- 滤波器 `h` 在构造时被分解为多相形式（通过 `_pad_h`）

**执行逻辑**：
1. 参数校验
2. dtype 推断
3. 多相分解（`_pad_h`）
4. 滤波器系数转 CuPy 数组并转置翻转

#### 行 538：注释说明

```python
# This is equivalent to (but faster than) using cp.apply_along_axis
```

**解释**：
- 等价于 `cp.apply_along_axis`，但更快
- 原因：专用 CUDA kernel 实现多相滤波
- 避免 Python 循环开销

#### 行 539：应用滤波器并返回

```python
return ufd.apply_filter(x, axis)
```

**语法说明**：
- `ufd.apply_filter`：实例方法，执行实际滤波
- 参数：`x`（输入信号）、`axis`（滤波轴）
- 返回：重采样后的数组

**数学映射**：
- 对应原理 3、4、5、6：执行阶段
- 多相结构 + Noble 恒等式 + Vaidyanathan 框图

---

### 4.2 _UpFIRDn 类初始化方法（__init__）

#### 行 166-171：判断数组库类型

```python
if isinstance(h, cp.ndarray):
    pp = cp
else:
    pp = np
```

**语法说明**：
- `isinstance(h, cp.ndarray)`：检查 `h` 是否为 CuPy 数组
- `pp`：数组库别名，指向 `cp`（CuPy）或 `np`（NumPy）
- 目的：预处理阶段可用 CPU（NumPy）或 GPU（CuPy）

**设计考量**：
- 滤波器系数可能来自 NumPy 或 CuPy
- 选择合适的数组库避免不必要的数据搬运

#### 行 173：转换为数组

```python
h = pp.asarray(h)
```

**语法说明**：
- `pp.asarray(h)`：将 `h` 转为数组（若已是数组则不变）
- 使用 `pp`（NumPy 或 CuPy）

#### 行 174-175：参数校验

```python
if h.ndim != 1 or h.size == 0:
    raise ValueError("h must be 1D with non-zero length")
```

**语法说明**：
- `h.ndim != 1`：检查是否为 1D 数组
- `h.size == 0`：检查是否为空
- 若违反，抛出 `ValueError`

**设计考量**：
- FIR 滤波器必须是 1D
- 非空数组保证卷积有意义

#### 行 177-178：dtype 推断与转换

```python
self._output_type = pp.result_type(h.dtype, x_dtype, pp.float32)
h = pp.asarray(h, self._output_type)
```

**语法说明**：
- `pp.result_type(h.dtype, x_dtype, pp.float32)`：计算结果类型
- 参数：滤波器 dtype、输入 dtype、float32（基准类型）
- 返回：最宽的 dtype（保证精度）

**数值稳定性**：
- 强制至少 float32 精度
- 避免低精度（如 float16）导致的数值问题

#### 行 179-180：存储采样率因子

```python
self._up = int(up)
self._down = int(down)
```

**语法说明**：
- `int(up)`：强制转换为整数
- 存储为实例属性 `_up` 和 `_down`

#### 行 181-182：参数范围校验

```python
if self._up < 1 or self._down < 1:
    raise ValueError("Both up and down must be >= 1")
```

**语法说明**：
- 检查采样率因子是否为正整数
- 若违反，抛出 `ValueError`

#### 行 183-184：多相分解预处理

```python
# This both transposes, and "flips" each phase for filtering
self._h_trans_flip = _pad_h(h, pp, self._up)
```

**语法说明**：
- `_pad_h(h, pp, self._up)`：调用多相分解辅助函数
- 返回：转置并翻转的多相系数数组
- 存储为 `_h_trans_flip`

**数学映射**：
- 对应原理 4（多相分解）
- 将滤波器 $h[n]$ 重排为多相形式：
  $$H(z) = \sum_{r=0}^{L-1} z^{-r} E_r(z^L)$$

#### 行 185-186：转换为 CuPy 连续数组

```python
self._h_trans_flip = cp.asarray(self._h_trans_flip)
self._h_trans_flip = cp.ascontiguousarray(self._h_trans_flip)
```

**语法说明**：
- `cp.asarray`：确保为 CuPy 数组
- `cp.ascontiguousarray`：确保内存连续布局（C-order）

**性能优化**：
- 连续内存布局提升 GPU 访问效率
- 避免 CUDA kernel 中的非连续访存

#### 行 187：存储原始滤波器长度

```python
self._h_len_orig = len(h)
```

**语法说明**：
- 存储 `h` 的原始长度
- 用于后续计算输出长度

---

### 4.3 apply_filter 方法（核心滤波）

#### 行 194-196：输入转换

```python
"""Apply the prepared filter to the specified axis of a nD signal x"""

x = cp.asarray(x, self._output_type)
```

**语法说明**：
- 将输入 `x` 转为 CuPy 数组
- dtype 设为 `_output_type`（初始化时推断）

#### 行 198-201：计算输出形状

```python
output_len = _output_len(self._h_len_orig, x.shape[axis], self._up, self._down)
output_shape = list(x.shape)
output_shape[axis] = output_len
out = cp.empty(output_shape, dtype=self._output_type, order="C")
```

**语法说明**：
- `_output_len(...)`：计算输出长度
- `list(x.shape)`：复制输入形状
- `output_shape[axis] = output_len`：修改目标轴长度
- `cp.empty(...)`：预分配输出数组（C-order）

**数学公式**：
$$L_{out} = \lfloor ((L_{in} - 1) \cdot L + N_h - 1) / M \rfloor + 1$$

#### 行 202：规范化 axis

```python
axis = axis % x.ndim
```

**语法说明**：
- 将负索引转为正索引（如 -1 → ndim-1）
- 取模运算保证 `axis` 在有效范围

#### 行 204-208：预计算变量

```python
# Precompute variables on CPU
x_shape_a = x.shape[axis]
h_per_phase = len(self._h_trans_flip) // self._up
padded_len = x.shape[axis] + (len(self._h_trans_flip) // self._up) - 1
```

**语法说明**：
- `x_shape_a`：输入沿 axis 的长度
- `h_per_phase`：每个多相分支的系数数 = $N_h / L$
- `padded_len`：扩展后的虚拟长度（用于边界处理）

**数学映射**：
- `h_per_phase`：原理 4 中每个子滤波器 $E_r$ 的长度

#### 行 209-220：计算 CUDA grid/block 配置

```python
if out.ndim > 1:
    threadsperblock = (8, 8)
    blocks = ceil(out.shape[0] / threadsperblock[0])
    blockspergrid_x = blocks if blocks < _get_max_gdx() else _get_max_gdx()

    blocks = ceil(out.shape[1] / threadsperblock[1])
    blockspergrid_y = blocks if blocks < _get_max_gdy() else _get_max_gdy()

    blockspergrid = (blockspergrid_x, blockspergrid_y)

else:
    threadsperblock, blockspergrid = _get_tpb_bpg()
```

**语法说明**：
- 分支：1D 或 2D 情况
- 2D：固定 block 大小 `(8, 8)`，grid 大小根据输出形状和 GPU 限制计算
- 1D：调用 `_get_tpb_bpg()` 自动计算

**GPU 并行化**：
- 2D：每个线程处理一个输出元素（2D 索引）
- 1D：每个线程处理一个输出样本
- `_get_max_gdx()`、`_get_max_gdy()`：查询 GPU 最大 grid 维度

#### 行 222-246：选择 kernel 类型

```python
if out.ndim == 1:
    k_type = "upfirdn1D"

    _populate_kernel_cache(out.dtype, k_type)

    kernel = _get_backend_kernel(
        out.dtype,
        blockspergrid,
        threadsperblock,
        k_type,
    )
elif out.ndim == 2:
    k_type = "upfirdn2D"

    _populate_kernel_cache(out.dtype, k_type)

    kernel = _get_backend_kernel(
        out.dtype,
        blockspergrid,
        threadsperblock,
        k_type,
    )
else:
    raise NotImplementedError("upfirdn() requires ndim <= 2")
```

**语法说明**：
- 根据 `out.ndim` 选择 kernel 类型：`"upfirdn1D"` 或 `"upfirdn2D"`
- `_populate_kernel_cache`：从 fatbin 文件加载编译好的 CUDA kernel
- `_get_backend_kernel`：从缓存获取 kernel wrapper

**限制**：
- 仅支持 1D 和 2D 输出
- 高维需展平或分批处理

#### 行 247-258：调用 CUDA kernel

```python
kernel(
    x,
    self._h_trans_flip,
    self._up,
    self._down,
    axis,
    x_shape_a,
    h_per_phase,
    padded_len,
    out,
)
```

**语法说明**：
- `kernel(...)`：调用 CUDA kernel wrapper
- 参数列表：
  - `x`：输入数组
  - `h_trans_flip`：多相分解后的滤波器系数
  - `up`, `down`：采样率因子
  - `axis`：滤波轴
  - `x_shape_a`：输入沿 axis 的长度
  - `h_per_phase`：每个多相分支的系数数
  - `padded_len`：扩展长度
  - `out`：输出数组

**执行逻辑**：
- CUDA kernel 在 GPU 上并行执行多相滤波
- 每个线程计算一个输出样本
- 利用共享内存缓存滤波器系数

#### 行 260：打印 kernel 属性（调试）

```python
_print_atts(kernel)
```

**语法说明**：
- `_print_atts(kernel)`：打印 kernel 属性（调试用）
- 可能包括：grid 大小、block 大小、寄存器使用量等

#### 行 262：返回输出

```python
return out
```

**语法说明**：
- 返回重采样后的数组
- dtype 为 `_output_type`，shape 已在前面计算

---

### 4.4 _pad_h 函数（多相分解）

#### 行 32-39：函数 docstring

```python
"""Store coefficients in a transposed, flipped arrangement.
For example, suppose upRate is 3, and the
input number of coefficients is 10, represented as h[0], ..., h[9].
Then the internal buffer will look like this::
   h[9], h[6], h[3], h[0],   // flipped phase 0 coefs
   0,    h[7], h[4], h[1],   // flipped phase 1 coefs (zero-padded)
   0,    h[8], h[5], h[2],   // flipped phase 2 coefs (zero-padded)
"""
```

**翻译与解释**：
- 将系数存储为转置、翻转的布局
- 示例：`up=3`，10 个系数
- 重排为 3 个多相分支：
  - 分支 0：`h[9], h[6], h[3], h[0]`（翻转）
  - 分支 1：`0, h[7], h[4], h[1]`（补零）
  - 分支 2：`0, h[8], h[5], h[2]`（补零）

**数学映射**：
- 对应原理 4（多相分解）
- 将 $h[n]$ 按 $n \mod L$ 分组：
  $$E_r[k] = h[kL + r]$$
- 翻转：卷积定义要求反转

#### 行 40：计算填充长度

```python
h_padlen = len(h) + (-len(h) % up)
```

**语法说明**：
- `len(h)`：滤波器长度
- `(-len(h) % up)`：补齐到 `up` 的倍数所需补零数
- `h_padlen`：填充后的长度

**示例**：
- `len(h)=10, up=3` → `h_padlen = 10 + (-10 % 3) = 10 + 2 = 12`

#### 行 41-42：创建填充数组

```python
h_full = pp.zeros(h_padlen, h.dtype)
h_full[: len(h)] = h
```

**语法说明**：
- `pp.zeros(h_padlen, h.dtype)`：创建零数组
- `h_full[: len(h)] = h`：将原始系数复制到前端

**效果**：
- `h_full = [h[0], ..., h[9], 0, 0]`（示例）

#### 行 43：重排为多相形式

```python
h_full = h_full.reshape(-1, up).T[:, ::-1].ravel()
```

**语法说明**：
- `reshape(-1, up)`：重塑为 `(N/up, up)` 矩阵
- `.T`：转置为 `(up, N/up)`
- `[:, ::-1]`：列反转（翻转每个多相分支）
- `.ravel()`：展平为 1D

**步骤分解**（示例）：

1. `reshape(-1, 3)`：`[[h0, h1, h2], [h3, h4, h5], [h6, h7, h8], [h9, 0, 0]]`
2. `.T`：`[[h0, h3, h6, h9], [h1, h4, h7, 0], [h2, h5, h8, 0]]`
3. `[:, ::-1]`：`[[h9, h6, h3, h0], [0, h7, h4, h1], [0, h8, h5, h2]]`
4. `.ravel()`：`[h9, h6, h3, h0, 0, h7, h4, h1, 0, h8, h5, h2]`

#### 行 44：返回结果

```python
return h_full
```

**语法说明**：
- 返回多相分解后的系数数组
- 供 `_UpFIRDn.__init__` 使用

---

### 4.5 _output_len 函数（输出长度计算）

#### 行 48：公式实现

```python
return (((in_len - 1) * up + len_h) - 1) // down + 1
```

**数学公式**：
$$L_{out} = \left\lfloor \frac{(L_{in} - 1) \cdot L + N_h - 1}{M} \right\rfloor + 1$$

**推导**：
- 上采样后长度：$(L_{in} - 1) \cdot L + 1$
- 卷积后长度：$(L_{in} - 1) \cdot L + N_h$
- 下采样后长度：$\lfloor ((L_{in} - 1) \cdot L + N_h - 1) / M \rfloor + 1$

---

### 4.6 _populate_kernel_cache 函数

#### 行 135-136：类型检查

```python
if np_type not in _SUPPORTED_TYPES:
    raise ValueError("Datatype {} not found for '{}'".format(np_type, k_type))
```

**语法说明**：
- `_SUPPORTED_TYPES`：`["float32", "float64", "complex64", "complex128"]`
- 检查 dtype 是否支持
- 若不支持，抛出异常

#### 行 138-139：缓存检查

```python
if (str(np_type), k_type) in _cupy_kernel_cache:
    return
```

**语法说明**：
- `_cupy_kernel_cache`：全局字典，缓存已加载的 kernel
- 键：`(dtype_str, k_type)`
- 若已缓存，直接返回（避免重复加载）

#### 行 141-144：加载 kernel

```python
_cupy_kernel_cache[(str(np_type), k_type)] = _get_function(
    "/filtering/_upfirdn.fatbin",
    "_cupy_" + k_type + "_" + str(np_type),
)
```

**语法说明**：
- `_get_function`：从 fatbin 文件加载 CUDA kernel
- 路径：`/filtering/_upfirdn.fatbin`
- kernel 名：`"_cupy_upfirdn1D_float32"` 等

---

### 4.7 _get_backend_kernel 函数

#### 行 154：从缓存获取 kernel

```python
kernel = _cupy_kernel_cache[(str(dtype), k_type)]
```

**语法说明**：
- 从全局缓存获取 kernel 对象

#### 行 155-160：创建 wrapper

```python
if kernel:
    if k_type == "upfirdn1D":
        return _cupy_upfirdn_wrapper(grid, block, kernel)
    elif k_type == "upfirdn2D":
        return _cupy_upfirdn2d_wrapper(grid, block, kernel)
else:
    raise ValueError("Kernel {} not found in _cupy_kernel_cache".format(k_type))
```

**语法说明**：
- 根据类型创建相应的 wrapper 对象
- 1D：`_cupy_upfirdn_wrapper`
- 2D：`_cupy_upfirdn2d_wrapper`

---

### 4.8 Wrapper 类

#### 4.8.1 初始化

```python
def __init__(self, grid, block, kernel):
    if isinstance(grid, int):
        grid = (grid,)
    if isinstance(block, int):
        block = (block,)

    self.grid = grid
    self.block = block
    self.kernel = kernel
```

**语法说明**：
- 将 `grid` 和 `block` 标准化为元组
- 存储 kernel 对象

#### 4.8.2 调用（1D）

```python
def __call__(
    self,
    x,
    h_trans_flip,
    up,
    down,
    axis,
    x_shape_a,
    h_per_phase,
    padded_len,
    out,
):

    kernel_args = (
        x,
        h_trans_flip,
        up,
        down,
        axis,
        x_shape_a,
        h_per_phase,
        padded_len,
        out,
        out.shape[0],
    )

    self.kernel(self.grid, self.block, kernel_args)
```

**语法说明**：
- 打包所有参数为元组 `kernel_args`
- 调用底层 CuPy kernel：`self.kernel(grid, block, args)`
- 参数按顺序传递给 CUDA kernel

#### 4.8.3 调用（2D）

```python
kernel_args = (
    x,
    x.shape[1],
    h_trans_flip,
    up,
    down,
    axis,
    x_shape_a,
    h_per_phase,
    padded_len,
    out,
    out.shape[0],
    out.shape[1],
)
```

**语法说明**：
- 2D 情况额外传递 `x.shape[1]`、`out.shape[1]`
- 用于 kernel 内部的 2D 索引计算

---

## 五、调用链与算法总结

### 5.1 调用链

```
upfirdn(h, x, up, down, axis)
├── _UpFIRDn.__init__(h, x.dtype, up, down)
│   ├── 参数校验
│   ├── dtype 推断
│   ├── _pad_h(h, pp, up)  # 多相分解
│   └── 转为 CuPy 连续数组
└── _UpFIRDn.apply_filter(x, axis)
    ├── 输入转换
    ├── 计算输出形状
    ├── 预计算变量
    ├── 计算 grid/block 配置
    ├── _populate_kernel_cache(dtype, k_type)  # 加载 kernel
    ├── _get_backend_kernel(...)  # 获取 wrapper
    └── kernel(...)  # 执行 CUDA kernel
```

### 5.2 算法总结

#### 核心步骤

1. **初始化阶段**：
   - 参数校验
   - 多相分解（`_pad_h`）
   - 滤波器系数预处理

2. **滤波阶段**：
   - 输入转换
   - 输出形状计算
   - CUDA kernel 配置
   - 并行执行多相滤波

3. **CUDA kernel**（未展示源码，但从参数可推断）：
   - 每个线程计算一个输出样本
   - 循环遍历多相分支
   - 利用共享内存缓存系数

#### 复杂度分析

- **时间复杂度**：$O(N_h / L)$ 每个输出样本（多相实现）
- **空间复杂度**：$O(N_h + N_{out})$
- **GPU 并行度**：$N_{out}$ 个线程并行

### 5.3 边界处理

- **常数填充**（`mode='constant'`，默认）
- 等价于假设输入边界外为零
- 输出长度由公式严格计算

### 5.4 dtype 处理

- 强制至少 float32 精度
- 支持 float32、float64、complex64、complex128
- 其他类型提升到 float64

---

## 六、数学映射、边界、复杂度和阅读检查

### 6.1 数学公式与代码映射

| 数学公式 | 代码位置 | 说明 |
|----------|---------|------|
| 多相分解 $H(z) = \sum_{r=0}^{L-1} z^{-r} E_r(z^L)$ | `_pad_h` 函数（行 31-44） | 重排系数为多相形式 |
| 输出长度 $L_{out} = \lfloor ((L_{in}-1)L+N_h-1)/M \rfloor + 1$ | `_output_len` 函数（行 48） | 计算输出数组长度 |
| 多相卷积 | CUDA kernel（`_upfirdn.fatbin`） | 并行执行多相滤波 |
| Noble 恒等式 | kernel 参数设计 | 避免显式上采样/下采样 |

### 6.2 边界处理

- 默认常数填充（`mode='constant'`，`cval=0`）
- 不支持 SciPy 的其他边界模式（`symmetric`、`reflect` 等）
- 输出长度严格按公式计算，无额外边界元素

### 6.3 数值稳定性

- dtype 推断强制至少 float32
- 使用 CuPy 数组保证 GPU 精度
- 无显式数值稳定措施（依赖 CuPy 底层）

### 6.4 性能瓶颈

- 滤波器系数预处理：CPU 上执行，若 `h` 是 NumPy 数组
- kernel 加载：首次调用时有编译/加载开销（缓存后无）
- 内存带宽：输出数组写入可能成为瓶颈（依赖 GPU 架构）

---

## 七、建议阅读顺序

### 7.1 初次阅读

1. **先读公开 API**：`resample.py:448-539`
   - 理解函数签名和参数语义
   - 理解调用链入口

2. **再读核心类**：`_upfirdn_cuda.py:164-261`
   - 理解初始化流程（多相分解）
   - 理解滤波流程（kernel 调用）

3. **最后读辅助函数**：
   - `_pad_h`（多相分解）
   - `_output_len`（输出长度）
   - `_populate_kernel_cache`、`_get_backend_kernel`（kernel 管理）

### 7.2 每段代码应观察的问题

1. **upfirdn 函数**：
   - 为什么只需 2 行代码？（封装设计）
   - 为什么比 `cp.apply_along_axis` 快？（专用 kernel）

2. **_UpFIRDn.__init__**：
   - 为什么要判断 `h` 的数组库类型？（CPU/GPU 混合）
   - `_pad_h` 的作用？（多相分解）
   - 为什么要 `ascontiguousarray`？（内存布局优化）

3. **_UpFIRDn.apply_filter**：
   - 输出长度如何计算？（公式映射）
   - grid/block 如何确定？（GPU 并行化）
   - 为什么支持最多 2D？（kernel 设计限制）

4. **_pad_h 函数**：
   - 为什么需要填充？（对齐到 L 的倍数）
   - 为什么转置并翻转？（多相分解 + 卷积定义）

5. **Wrapper 类**：
   - 参数如何传递给 kernel？（kernel_args 元组）
   - 1D 和 2D 的区别？（额外传递 shape[1]）

---

## 八、参考资料

### 源码文件

1. **学习副本**：
   - `Learning/cusignal-23.08.00/python/cusignal/filtering/resample.py`
   - `Learning/cusignal-23.08.00/python/cusignal/filtering/_upfirdn_cuda.py`

2. **只读基准**：
   - `ZKX/cusignal-23.08.00/python/cusignal/filtering/resample.py`
   - `ZKX/cusignal-23.08.00/python/cusignal/filtering/_upfirdn_cuda.py`

### 相关依赖

- CuPy 文档：https://docs.cupy.dev/
- CUDA Programming Guide：https://docs.nvidia.com/cuda/

---

完成本文档后，建议用户：
1. **对照源码逐行阅读**，验证解释是否正确
2. **运行示例代码**，观察输出形状和数值
3. **尝试不同参数组合**，理解边界条件

理解后，请明确告知是否进入 **阶段三：cusignal_cpp CPU/GPU 复现逻辑**。