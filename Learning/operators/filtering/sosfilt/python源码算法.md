# sosfilt Python 源码算法

## 1. 代码定位与接口概览

### 1.1 公开导入路径

用户调用路径是：

```python
import cusignal
y = cusignal.sosfilt(sos, x)
```

公开导出链：

1. `Learning/cusignal-23.08.00/python/cusignal/filtering/__init__.py:31`（学习副本）/ `ZKX/cusignal-23.08.00/python/cusignal/filtering/__init__.py:26`（只读基准）导出 `cusignal.filtering.sosfilt`。
2. `Learning/cusignal-23.08.00/python/cusignal/__init__.py:58`（学习副本）/ `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:51`（只读基准）导出顶层 `cusignal.sosfilt`。
3. 函数定义位于 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:656`（学习副本）/ `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:591`（只读基准），符号 `sosfilt`。

### 1.2 签名、参数、返回值、dtype 与 shape

```python
sosfilt(sos, x, axis=-1, zi=None)
```

| 项目 | 语义 |
| --- | --- |
| `sos` | shape 为 `(n_sections, 6)`；每行依次是 `b0,b1,b2,a0,a1,a2`；当前实现强制 `a0 == 1` |
| `x` | 至少一维的 CuPy/可转为 CuPy 的数组；沿 `axis` 滤波，其他维组合成互相独立的信号 |
| `axis` | 默认 `-1`，允许负索引；函数把该轴移到最后交给 kernel |
| `zi` | 可选初态；shape 为 `(n_sections, ..., 2, ...)`，其中输入的滤波轴长度被 2 替换 |
| `y` | 与 `x` shape 相同；数据来自原地写回的连续 GPU 缓冲区 |
| `zf` | 只有传入 `zi` 才随 `y` 返回；但当前 kernel 未见把寄存器末状态写回 `zi`，见 8.4 节 |
| dtype | Python 层用 `cp.result_type(sos, x, zi)` 推断；字符门禁较宽，但预编译 kernel 缓存只声明 `float32`、`float64`，所以实际后端范围更窄 |

## 2. 当前算子的完整相关源码

以下摘录全部来自只读基准 `ZKX/cusignal-23.08.00`，保持原文与原始行序；学习副本只额外插入合法的 `<学习注释：...>` 独立注释行。

### 2.1 必要公开导出语句

`ZKX/cusignal-23.08.00/python/cusignal/filtering/__init__.py:14-28`：

```python
from cusignal.filtering.filtering import (
    channelize_poly,
    detrend,
    filtfilt,
    firfilter,
    firfilter2,
    firfilter_zi,
    freq_shift,
    hilbert,
    hilbert2,
    lfilter,
    lfilter_zi,
    sosfilt,
    wiener,
)
```

`ZKX/cusignal-23.08.00/python/cusignal/__init__.py:39-53` 中同样的导入块把 `sosfilt` 暴露到顶层包；与当前算子直接相关的必要语句是第 39 行导入块起点和第 51 行 `sosfilt,`。

### 2.2 `_validate_sos` 完整定义

`ZKX/cusignal-23.08.00/python/cusignal/filter_design/filter_design_utils.py:17-27`：

```python
def _validate_sos(sos):
    """Helper to validate a SOS input"""
    sos = np.atleast_2d(sos)
    if sos.ndim != 2:
        raise ValueError("sos array must be 2D")
    n_sections, m = sos.shape
    if m != 6:
        raise ValueError("sos array must be shape (n_sections, 6)")
    if not (sos[:, 3] == 1).all():
        raise ValueError("sos[:, 3] should be all ones")
    return sos, n_sections
```

### 2.3 设备上限 helper 完整定义

`ZKX/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:27-38`：

```python
def _get_max_smem():

    device_id = cp.cuda.Device()

    return device_id.attributes["MaxSharedMemoryPerBlock"]


def _get_max_tpb():

    device_id = cp.cuda.Device()

    return device_id.attributes["MaxThreadsPerBlock"]
```

### 2.4 `sosfilt` 完整定义（含完整 docstring 和示例）

`ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:591-753`：

```python
def sosfilt(
    sos,
    x,
    axis=-1,
    zi=None,
):
    """
    Filter data along one dimension using cascaded second-order sections.
    Filter a data sequence, `x`, using a digital IIR filter defined by
    `sos`.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    x : array_like
        An N-dimensional input array.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis.  Default is -1.
    zi : array_like, optional
        Initial conditions for the cascaded filter delays.  It is a (at
        least 2D) vector of shape ``(n_sections, ..., 2, ...)``, where
        ``..., 2, ...`` denotes the shape of `x`, but with ``x.shape[axis]``
        replaced by 2.  If `zi` is None or is not given then initial rest
        (i.e. all zeros) is assumed.
        Note that these initial conditions are *not* the same as the initial
        conditions given by `lfiltic` or `lfilter_zi`.

    Returns
    -------
    y : ndarray
        The output of the digital filter.
    zf : ndarray, optional
        If `zi` is None, this is not returned, otherwise, `zf` holds the
        final filter delay values.
    See Also
    --------
    zpk2sos, sos2zpk, sosfilt_zi, sosfiltfilt, sosfreqz

    Notes
    -----
    WARNING: This is an experimental API and is prone to change in future
    versions of cuSignal.

    The filter function is implemented as a series of second-order filters
    with direct-form II transposed structure. It is designed to minimize
    numerical precision errors for high-order filters.

    Limitations
    -----------
    1. The number of n_sections must be less than 513.
    2. The number of samples must be greater than the number of sections

    Examples
    --------
    sosfilt is a stable alternative to `lfilter` as using 2nd order sections
    reduces numerical error. We are working on building out sos filter output,
    so please submit GitHub feature requests as needed. You can also generate
    a filter on CPU with scipy.signal and then move that to GPU for actual
    filtering operations with `cp.asarray`.

    Plot a 13th-order filter's impulse response using both `sosfilt`:
    >>> from scipy import signal
    >>> import cusignal
    >>> import cupy as cp
    >>> # Generate filter on CPU with Scipy.Signal
    >>> sos = signal.ellip(13, 0.009, 80, 0.05, output='sos')
    >>> # Move data to GPU
    >>> sos = cp.asarray(sos)
    >>> x = cp.random.randn(100_000_000)
    >>> y = cusignal.sosfilt(sos, x)
    """

    x = cp.asarray(x)
    if x.ndim == 0:
        raise ValueError("x must be at least 1D")

    sos, n_sections = _validate_sos(sos)
    sos = cp.asarray(sos)

    x_zi_shape = list(x.shape)
    x_zi_shape[axis] = 2
    x_zi_shape = tuple([n_sections] + x_zi_shape)
    inputs = [sos, x]

    if zi is not None:
        inputs.append(np.asarray(zi))

    dtype = cp.result_type(*inputs)

    if dtype.char not in "fdgFDGO":
        raise NotImplementedError("input type '%s' not supported" % dtype)
    if zi is not None:
        zi = cp.array(zi, dtype)  # make a copy so that we can operate in place
        if zi.shape != x_zi_shape:
            raise ValueError(
                "Invalid zi shape. With axis=%r, an input with "
                "shape %r, and an sos array with %d sections, zi "
                "must have shape %r, got %r."
                % (axis, x.shape, n_sections, x_zi_shape, zi.shape)
            )
        return_zi = True
    else:
        zi = cp.zeros(x_zi_shape, dtype=dtype)
        return_zi = False

    axis = axis % x.ndim  # make positive
    x = cp.moveaxis(x, axis, -1)
    zi = cp.moveaxis(zi, [0, axis + 1], [-2, -1])
    x_shape, zi_shape = x.shape, zi.shape
    x = cp.reshape(x, (-1, x.shape[-1]))
    x = cp.array(x, dtype, order="C")  # make a copy, can modify in place
    zi = cp.ascontiguousarray(cp.reshape(zi, (-1, n_sections, 2)))
    sos = sos.astype(dtype, copy=False)

    max_smem = _get_max_smem()
    max_tpb = _get_max_tpb()

    # Determine how much shared memory is needed
    out_size = sos.shape[0]
    sos_size = sos.shape[0] * sos.shape[1]
    shared_mem = (out_size + sos_size) * x.dtype.itemsize

    if shared_mem > max_smem:
        max_sections = max_smem // (1 + zi.shape[2] + sos.shape[1]) // x.dtype.itemsize
        raise ValueError(
            "The number of sections ({}), requires too much "
            "shared memory ({}B) > ({}B). \n"
            "\n**Max sections possible ({})**".format(
                sos.shape[0], shared_mem, max_smem, max_sections
            )
        )

    if sos.shape[0] > max_tpb:
        raise ValueError(
            "The number of sections ({}), must be less "
            "than max threads per block ({})".format(sos.shape[0], max_tpb)
        )

    if sos.shape[0] > x.shape[1]:
        raise ValueError(
            "The number of samples ({}), must be greater "
            "than the number of sections ({})".format(x.shape[1], sos.shape[0])
        )

    _sosfilt(sos, x, zi)

    x.shape = x_shape
    x = cp.moveaxis(x, -1, axis)
    if return_zi:
        zi.shape = zi_shape
        zi = cp.moveaxis(zi, [-2, -1], [0, axis + 1])
        out = (x, zi)
    else:
        out = x

    return out
```

### 2.5 CUDA Python 包装完整相关源码

`ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:17-96`：

```python
_SUPPORTED_TYPES = ["float32", "float64"]


class _cupy_sosfilt_wrapper(object):
    def __init__(self, grid, block, smem, kernel):
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)

        self.grid = grid
        self.block = block
        self.smem = smem
        self.kernel = kernel

    def __call__(self, sos, x, zi):

        kernel_args = (
            x.shape[0],
            x.shape[1],
            sos.shape[0],
            zi.shape[2],
            sos,
            zi,
            x,
        )

        self.kernel(self.grid, self.block, kernel_args, shared_mem=self.smem)


def _populate_kernel_cache(np_type, k_type):

    if np_type not in _SUPPORTED_TYPES:
        raise ValueError("Datatype {} not found for '{}'".format(np_type, k_type))

    if (str(np_type), k_type) in _cupy_kernel_cache:
        return

    _cupy_kernel_cache[(str(np_type), k_type)] = _get_function(
        "/filtering/_sosfilt.fatbin",
        "_cupy_" + k_type + "_" + str(np_type),
    )


def _get_backend_kernel(dtype, grid, block, smem, k_type):
    kernel = _cupy_kernel_cache[(dtype.name, k_type)]
    if kernel:
        return _cupy_sosfilt_wrapper(grid, block, smem, kernel)
    else:
        raise ValueError("Kernel {} not found in _cupy_kernel_cache".format(k_type))

    raise NotImplementedError("No kernel found for datatype {}".format(dtype.name))


def _sosfilt(sos, x, zi):

    threadsperblock = sos.shape[0]  # Up-to (1024, 1) = 1024 max per block
    blockspergrid = x.shape[0]

    k_type = "sosfilt"

    _populate_kernel_cache(x.dtype, k_type)

    out_size = threadsperblock
    sos_size = sos.shape[0] * sos.shape[1]

    shared_mem = (out_size + sos_size) * x.dtype.itemsize

    kernel = _get_backend_kernel(
        x.dtype,
        blockspergrid,
        threadsperblock,
        shared_mem,
        k_type,
    )
    print(zi.shape)

    kernel(sos, x, zi)

    _print_atts(kernel)
```

### 2.6 CUDA kernel 完整定义

`ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:18-143`：

```cpp
constexpr int sos_width = 6;

template<typename T>
__device__ void _cupy_sosfilt( const int n_signals,
                               const int n_samples,
                               const int n_sections,
                               const int zi_width,
                               const T *__restrict__ sos,
                               const T *__restrict__ zi,
                               T *__restrict__ x_in,
                               T *s_buffer ) {

    T *s_out { s_buffer };
    T *s_sos { reinterpret_cast<T *>( &s_out[n_sections] ) };

    const int tx { static_cast<int>( threadIdx.x ) };
    const int bx { static_cast<int>( blockIdx.x ) };

    // Reset shared memory
    s_out[tx] = 0;

    // Load SOS
    // b is in s_sos[tx * sos_width + [0-2]]
    // a is in s_sos[tx * sos_width + [3-5]]
#pragma unroll sos_width
    for ( int i = 0; i < sos_width; i++ ) {
        s_sos[tx * sos_width + i] = sos[tx * sos_width + i];
    }

    // __syncthreads( );

    T zi0 = zi[bx * n_sections * zi_width + tx * zi_width + 0];
    T zi1 = zi[bx * n_sections * zi_width + tx * zi_width + 1];

    const int load_size { n_sections - 1 };
    const int unload_size { n_samples - load_size };

    T temp {};
    T x_n {};

    if ( bx < n_signals ) {
        // Loading phase
        for ( int n = 0; n < load_size; n++ ) {
            __syncthreads( );
            if ( tx == 0 ) {
                x_n = x_in[bx * n_samples + n];
            } else {
                x_n = s_out[tx - 1];
            }

            // Use direct II transposed structure
            temp = s_sos[tx * sos_width + 0] * x_n + zi0;
            zi0  = s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + zi1;
            zi1  = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;

            s_out[tx] = temp;
        }

        // Processing phase
        for ( int n = load_size; n < n_samples; n++ ) {
            __syncthreads( );
            if ( tx == 0 ) {
                x_n = x_in[bx * n_samples + n];
            } else {
                x_n = s_out[tx - 1];
            }

            // Use direct II transposed structure
            temp = s_sos[tx * sos_width + 0] * x_n + zi0;
            zi0  = s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + zi1;
            zi1  = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;

            if ( tx < load_size ) {
                s_out[tx] = temp;
            } else {
                x_in[bx * n_samples + ( n - load_size )] = temp;
            }
        }

        // Unloading phase
        for ( int n = 0; n < n_sections; n++ ) {
            __syncthreads( );
            // retire threads that are less than n
            if ( tx > n ) {
                x_n = s_out[tx - 1];

                // Use direct II transposed structure
                temp = s_sos[tx * sos_width + 0] * x_n + zi0;
                zi0  = s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + zi1;
                zi1  = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;

                if ( tx < load_size ) {
                    s_out[tx] = temp;
                } else {
                    x_in[bx * n_samples + ( n + unload_size )] = temp;
                }
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__( 1024 ) _cupy_sosfilt_float32( const int n_signals,
                                                                            const int n_samples,
                                                                            const int n_sections,
                                                                            const int zi_width,
                                                                            const float *__restrict__ sos,
                                                                            const float *__restrict__ zi,
                                                                            float *__restrict__ x_in ) {

    extern __shared__ float s_buffer_f[];

    _cupy_sosfilt<float>( n_signals, n_samples, n_sections, zi_width, sos, zi, x_in, s_buffer_f );
}

extern "C" __global__ void __launch_bounds__( 1024 ) _cupy_sosfilt_float64( const int n_signals,
                                                                            const int n_samples,
                                                                            const int n_sections,
                                                                            const int zi_width,
                                                                            const double *__restrict__ sos,
                                                                            const double *__restrict__ zi,
                                                                            double *__restrict__ x_in ) {

    extern __shared__ double s_buffer_d[];

    _cupy_sosfilt<double>( n_signals, n_samples, n_sections, zi_width, sos, zi, x_in, s_buffer_d );
}
```

## 3. docstring 逐行翻译与解释

以下按“功能小标题 → 双行号定位 → 最小完整语义块原文 → 逐行对应说明”组织；空行不单列。

#### 文档字符串

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:663` 至 `:666`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:597` 至 `:600`。

```python
    """
    Filter data along one dimension using cascaded second-order sections.
    Filter a data sequence, `x`, using a digital IIR filter defined by
    `sos`.
```

逐行对应说明：

- 第 1 行：开始函数 docstring，运行时成为 `sosfilt.__doc__`。
- 第 2 行：使用级联二阶节沿一个维度滤波。
- 第 3 行：被处理序列名为 `x`，滤波器由 `sos` 定义。
- 第 4 行：上一行英文句子的续行；反引号是 Sphinx 交叉引用样式。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:668` 至 `:679`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:602` 至 `:613`。

```python
    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    x : array_like
        An N-dimensional input array.
    axis : int, optional
        The axis of the input data array along which to apply the
```

逐行对应说明：

- 第 1 行：参数章节标题。
- 第 2 行：NumPy docstring 标题下划线。
- 第 3 行：`sos` 接受可转成数组的对象。
- 第 4 行：它保存二阶滤波节系数，必须有指定 shape。
- 第 5 行：第一维是节数，每行对应一节。
- 第 6 行：前三列为分子 $b_0,b_1,b_2$。
- 第 7 行：后三列为分母 $a_0,a_1,a_2$。
- 第 8 行：完成上一句。
- 第 9 行：输入也接受 array-like。
- 第 10 行：输入可为 N 维数组。
- 第 11 行：`axis` 是可选整数。
- 第 12 行：指定应用滤波的输入轴。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:680` 至 `:689`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:614` 至 `:623`。

```python
        linear filter. The filter is applied to each subarray along
        this axis.  Default is -1.
    zi : array_like, optional
        Initial conditions for the cascaded filter delays.  It is a (at
        least 2D) vector of shape ``(n_sections, ..., 2, ...)``, where
        ``..., 2, ...`` denotes the shape of `x`, but with ``x.shape[axis]``
        replaced by 2.  If `zi` is None or is not given then initial rest
        (i.e. all zeros) is assumed.
        Note that these initial conditions are *not* the same as the initial
        conditions given by `lfiltic` or `lfilter_zi`.
```

逐行对应说明：

- 第 1 行：该轴上的每条子数组分别经过同一线性滤波器。
- 第 2 行：默认最后一轴。
- 第 3 行：`zi` 是可选初始状态。
- 第 4 行：它为级联中每节的延迟单元给初值。
- 第 5 行：至少二维，首维是节，另含长度 2 的状态维。
- 第 6 行：除滤波轴外，状态 shape 对应输入其他各轴。
- 第 7 行：输入滤波轴长度被两个 DF-II-T 状态替代；省略时零初态。
- 第 8 行：“initial rest”就是所有状态为零。
- 第 9 行：此处状态布局与 `lfiltic` 的高阶直接型状态不同。
- 第 10 行：完成比较对象名称。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:691` 至 `:695`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:625` 至 `:629`。

```python
    Returns
    -------
    y : ndarray
        The output of the digital filter.
    zf : ndarray, optional
```

逐行对应说明：

- 第 1 行：返回值章节。
- 第 2 行：标题下划线。
- 第 3 行：主返回值是数组。
- 第 4 行：`y` 是滤波结果。
- 第 5 行：可选返回末状态。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:696` 至 `:697`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:630` 至 `:631`。

```python
        If `zi` is None, this is not returned, otherwise, `zf` holds the
        final filter delay values.
```

逐行对应说明：

- 第 1 行：不传 `zi` 时不返回 `zf`。
- 第 2 行：传 `zi` 时，契约声称 `zf` 保存最终延迟状态。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:698` 至 `:700`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:632` 至 `:634`。

```python
    See Also
    --------
    zpk2sos, sos2zpk, sosfilt_zi, sosfiltfilt, sosfreqz
```

逐行对应说明：

- 第 1 行：相关 API 章节。
- 第 2 行：标题下划线。
- 第 3 行：列出 SOS 转换、初态、前后向滤波和频响函数。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:702` 至 `:705`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:636` 至 `:639`。

```python
    Notes
    -----
    WARNING: This is an experimental API and is prone to change in future
    versions of cuSignal.
```

逐行对应说明：

- 第 1 行：说明章节。
- 第 2 行：标题下划线。
- 第 3 行：API 仍为实验性质，未来可能变更。
- 第 4 行：完成警告句。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:707` 至 `:709`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:641` 至 `:643`。

```python
    The filter function is implemented as a series of second-order filters
    with direct-form II transposed structure. It is designed to minimize
    numerical precision errors for high-order filters.
```

逐行对应说明：

- 第 1 行：算法把各二阶节串联。
- 第 2 行：每节采用 DF-II-T。
- 第 3 行：目标是降低高阶直接型的数值误差。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:711` 至 `:714`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:645` 至 `:648`。

```python
    Limitations
    -----------
    1. The number of n_sections must be less than 513.
    2. The number of samples must be greater than the number of sections
```

逐行对应说明：

- 第 1 行：当前实现限制章节。
- 第 2 行：标题下划线。
- 第 3 行：文档限制节数最多 512。
- 第 4 行：样本数必须大于节数。

#### 文档章节

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:716` 至 `:722`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:650` 至 `:656`。

```python
    Examples
    --------
    sosfilt is a stable alternative to `lfilter` as using 2nd order sections
    reduces numerical error. We are working on building out sos filter output,
    so please submit GitHub feature requests as needed. You can also generate
    a filter on CPU with scipy.signal and then move that to GPU for actual
    filtering operations with `cp.asarray`.
```

逐行对应说明：

- 第 1 行：示例章节。
- 第 2 行：标题下划线。
- 第 3 行：相比高阶 `lfilter`，SOS 通常更数值稳健。
- 第 4 行：原因是分解为低阶节；后半说明功能仍在建设。
- 第 5 行：邀请提交功能需求。
- 第 6 行：可用 SciPy 在 CPU 上设计滤波器。
- 第 7 行：再把系数迁移到 GPU 执行。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:724` 至 `:734`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:658` 至 `:668`。

```python
    Plot a 13th-order filter's impulse response using both `sosfilt`:
    >>> from scipy import signal
    >>> import cusignal
    >>> import cupy as cp
    >>> # Generate filter on CPU with Scipy.Signal
    >>> sos = signal.ellip(13, 0.009, 80, 0.05, output='sos')
    >>> # Move data to GPU
    >>> sos = cp.asarray(sos)
    >>> x = cp.random.randn(100_000_000)
    >>> y = cusignal.sosfilt(sos, x)
    """
```

逐行对应说明：

- 第 1 行：示例主题是 13 阶滤波器的冲激响应。
- 第 2 行：导入 SciPy 信号模块用于设计。
- 第 3 行：导入待调用库。
- 第 4 行：导入 CuPy 并命名为 `cp`。
- 第 5 行：下一行在 CPU 上生成系数。
- 第 6 行：设计 13 阶椭圆滤波器并直接输出 SOS。
- 第 7 行：下一步搬运数据。
- 第 8 行：将 SOS 系数转为 CuPy GPU 数组。
- 第 9 行：在 GPU 上生成一亿个正态随机样本。
- 第 10 行：调用公开入口滤波并得到 `y`。
- 第 11 行：结束 docstring。


## 4. Python 源代码逐行解释

### 4.0 两级公开导出

子包定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/__init__.py:14` 至 `:33`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/__init__.py:14` 至 `:28`。

```python
from cusignal.filtering.filtering import (
    channelize_poly,
    detrend,
    filtfilt,
    firfilter,
    firfilter2,
    firfilter_zi,
    freq_shift,
    hilbert,
    hilbert2,
    lfilter,
    lfilter_zi,
    sosfilt,
    wiener,
)
```

`from ... import (...)` 从实现模块一次导入多个过滤算子，并把它们绑定到 `cusignal.filtering` 子包。圆括号允许导入列表跨行，每个名称后的逗号分隔下一项；`sosfilt,` 是当前算子的公开绑定，其余名称不属于当前算法调用链，末行 `)` 关闭导入列表。

顶层定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/__init__.py:46` 至 `:65`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:39` 至 `:53`。

顶层包包含结构相同的 `from cusignal.filtering.filtering import (...)` 导入块，其中只需关注基准第 51 行：

```python
    sosfilt,
```

它把同一个函数对象进一步绑定为 `cusignal.sosfilt`。两级导出只改变 API 可见位置，不执行滤波、分配状态数组或启动 CUDA kernel。

### 4.1 函数签名逐行解释

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:656` 至 `:661`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:591` 至 `:596`。

```python
def sosfilt(
    sos,
    x,
    axis=-1,
    zi=None,
):
```

逐行对应说明：

- 第 1 行：用 `def` 定义公开函数；左括号开启多行形参列表。
- 第 2 行：第一个必选位置/关键字参数；逗号表示后续还有参数。
- 第 3 行：第二个必选参数，输入数组。
- 第 4 行：可选参数，默认最后一轴。
- 第 5 行：可选初态；`None` 表示未提供。
- 第 6 行：闭合形参列表，冒号开始函数 suite。


### 4.2 函数体逐行解释

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:737` 至 `:737`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:670` 至 `:670`。

```python
    x = cp.asarray(x)
```

逐行对应说明：

- 第 1 行：调用 CuPy 转换输入，并把返回数组绑定到局部名 `x`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:739` 至 `:740`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:671` 至 `:672`。

```python
    if x.ndim == 0:
        raise ValueError("x must be at least 1D")
```

逐行对应说明：

- 第 1 行：比较维数是否为 0；冒号开启条件块。
- 第 2 行：标量输入时立即抛出异常，后续代码不执行。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:743` 至 `:745`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:674` 至 `:675`。

```python
    sos, n_sections = _validate_sos(sos)
    sos = cp.asarray(sos)
```

逐行对应说明：

- 第 1 行：调用 helper；序列解包同时取得二维 SOS 和节数。
- 第 2 行：把 SOS 转成 GPU 数组。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:748` 至 `:753`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:677` 至 `:680`。

```python
    x_zi_shape = list(x.shape)
    x_zi_shape[axis] = 2
    x_zi_shape = tuple([n_sections] + x_zi_shape)
    inputs = [sos, x]
```

逐行对应说明：

- 第 1 行：将不可变 shape 元组复制成可修改列表。
- 第 2 行：用索引赋值把滤波轴长度改成两个状态。负索引由 Python 原生支持。
- 第 3 行：列表拼接在前面加入节数，再转回元组。
- 第 4 行：构造用于 dtype 推断的列表。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:755` 至 `:757`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:682` 至 `:683`。

```python
    if zi is not None:
        inputs.append(np.asarray(zi))
```

逐行对应说明：

- 第 1 行：用身份比较判断调用者是否提供初态。
- 第 2 行：将 `zi` 转为 NumPy 数组后追加；此步用于类型推断。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:760` 至 `:760`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:685` 至 `:685`。

```python
    dtype = cp.result_type(*inputs)
```

逐行对应说明：

- 第 1 行：`*` 展开列表为位置参数，按类型提升规则得到共同 dtype。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:763` 至 `:764`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:687` 至 `:688`。

```python
    if dtype.char not in "fdgFDGO":
        raise NotImplementedError("input type '%s' not supported" % dtype)
```

逐行对应说明：

- 第 1 行：成员测试拒绝不在允许字符集合中的 dtype 类别。
- 第 2 行：`%` 格式化插入 dtype，再报告该类型未实现。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:765` 至 `:767`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:689` 至 `:690`。

```python
    if zi is not None:
        zi = cp.array(zi, dtype)  # make a copy so that we can operate in place
```

逐行对应说明：

- 第 1 行：第二次分支决定初态复制和返回形式。
- 第 2 行：创建指定 dtype 的 GPU 副本；行末原注释说明为原地操作准备。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:769` 至 `:776`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:691` 至 `:698`。

```python
        if zi.shape != x_zi_shape:
            raise ValueError(
                "Invalid zi shape. With axis=%r, an input with "
                "shape %r, and an sos array with %d sections, zi "
                "must have shape %r, got %r."
                % (axis, x.shape, n_sections, x_zi_shape, zi.shape)
            )
        return_zi = True
```

逐行对应说明：

- 第 1 行：shape 元组不等则进入错误分支。
- 第 2 行：开始多行异常构造。
- 第 3 行：相邻字符串字面量会自动拼接；插槽记录 axis。
- 第 4 行：继续描述输入 shape 和节数。
- 第 5 行：继续描述期望和实际 `zi` shape。
- 第 6 行：用五元素元组填充前述 `%r/%d` 插槽。
- 第 7 行：结束 `ValueError` 与 `raise` 表达式。
- 第 8 行：记住调用者提供了初态，稍后返回二元组。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:777` 至 `:780`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:699` 至 `:701`。

```python
    else:
        zi = cp.zeros(x_zi_shape, dtype=dtype)
        return_zi = False
```

逐行对应说明：

- 第 1 行：与 689 行 `if` 配对。
- 第 2 行：按期望 shape 创建全零 GPU 状态。
- 第 3 行：记录最终只返回输出。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:783` 至 `:797`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:703` 至 `:710`。

```python
    axis = axis % x.ndim  # make positive
    x = cp.moveaxis(x, axis, -1)
    zi = cp.moveaxis(zi, [0, axis + 1], [-2, -1])
    x_shape, zi_shape = x.shape, zi.shape
    x = cp.reshape(x, (-1, x.shape[-1]))
    x = cp.array(x, dtype, order="C")  # make a copy, can modify in place
    zi = cp.ascontiguousarray(cp.reshape(zi, (-1, n_sections, 2)))
    sos = sos.astype(dtype, copy=False)
```

逐行对应说明：

- 第 1 行：模运算把负轴正规化；原注释说明目的。
- 第 2 行：返回轴重排 view/数组，把样本轴移到最后。
- 第 3 行：同时移动“节”轴和“两状态”轴到末两位。
- 第 4 行：多目标赋值保存两个移动后的 shape。
- 第 5 行：`-1` 自动推导独立信号数，末维保持样本数。
- 第 6 行：创建 C 连续可写副本；kernel 会在其中原地输出。
- 第 7 行：先变为 `[signals,sections,2]`，再确保连续。
- 第 8 行：转到共同 dtype；若已经一致则允许复用。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:800` 至 `:801`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:712` 至 `:713`。

```python
    max_smem = _get_max_smem()
    max_tpb = _get_max_tpb()
```

逐行对应说明：

- 第 1 行：查询每 block 最大共享内存字节数。
- 第 2 行：查询每 block 最大线程数。

#### 源码注释与相邻逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:803` 至 `:807`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:715` 至 `:718`。

```python
    # Determine how much shared memory is needed
    out_size = sos.shape[0]
    sos_size = sos.shape[0] * sos.shape[1]
    shared_mem = (out_size + sos_size) * x.dtype.itemsize
```

逐行对应说明：

- 第 1 行：原注释：以下计算动态共享内存需求。
- 第 2 行：节间输出槽数量等于节数。
- 第 3 行：系数元素数为节数乘 6。
- 第 4 行：元素总数乘每元素字节数，得到动态共享内存字节数。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:809` 至 `:817`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:720` 至 `:728`。

```python
    if shared_mem > max_smem:
        max_sections = max_smem // (1 + zi.shape[2] + sos.shape[1]) // x.dtype.itemsize
        raise ValueError(
            "The number of sections ({}), requires too much "
            "shared memory ({}B) > ({}B). \n"
            "\n**Max sections possible ({})**".format(
                sos.shape[0], shared_mem, max_smem, max_sections
            )
        )
```

逐行对应说明：

- 第 1 行：超过硬件上限则拒绝 launch。
- 第 2 行：用整数地板除估算可容纳节数；注意公式包含 `zi.shape[2]`。
- 第 3 行：开始构造共享内存错误。
- 第 4 行：`{}` 预留节数。
- 第 5 行：插入需求/上限字节并带换行转义。
- 第 6 行：第三段字符串后调用格式化方法。
- 第 7 行：对应节数、所需字节、上限和估算最大节数。
- 第 8 行：结束 `.format`。
- 第 9 行：结束 `ValueError` 与 `raise`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:820` 至 `:824`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:730` 至 `:734`。

```python
    if sos.shape[0] > max_tpb:
        raise ValueError(
            "The number of sections ({}), must be less "
            "than max threads per block ({})".format(sos.shape[0], max_tpb)
        )
```

逐行对应说明：

- 第 1 行：节数大于最大线程数则拒绝，因为一节一线程。
- 第 2 行：开始线程上限错误。
- 第 3 行：插入实际节数。
- 第 4 行：插入每 block 最大线程数。
- 第 5 行：结束异常。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:827` 至 `:831`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:736` 至 `:740`。

```python
    if sos.shape[0] > x.shape[1]:
        raise ValueError(
            "The number of samples ({}), must be greater "
            "than the number of sections ({})".format(x.shape[1], sos.shape[0])
        )
```

逐行对应说明：

- 第 1 行：节数多于样本数则违反该流水实现的条件。
- 第 2 行：开始样本数错误。
- 第 3 行：插入样本数。
- 第 4 行：插入节数。
- 第 5 行：结束异常。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:834` 至 `:834`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:742` 至 `:742`。

```python
    _sosfilt(sos, x, zi)
```

逐行对应说明：

- 第 1 行：调用内部 CUDA 启动包装；输出写回 `x`。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:837` 至 `:838`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:744` 至 `:745`。

```python
    x.shape = x_shape
    x = cp.moveaxis(x, -1, axis)
```

逐行对应说明：

- 第 1 行：原地恢复移动轴后的多维 shape。
- 第 2 行：把末尾样本轴移回调用者指定位置。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:839` 至 `:843`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:746` 至 `:749`。

```python
    if return_zi:
        zi.shape = zi_shape
        zi = cp.moveaxis(zi, [-2, -1], [0, axis + 1])
        out = (x, zi)
```

逐行对应说明：

- 第 1 行：根据早先布尔标志选择返回形式。
- 第 2 行：恢复状态在移动轴后的 shape。
- 第 3 行：把节轴和状态轴移回 API 约定位置。
- 第 4 行：构造 `(y,zf)` 二元组；`x` 此时就是 `y`。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:844` 至 `:845`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:750` 至 `:751`。

```python
    else:
        out = x
```

逐行对应说明：

- 第 1 行：未提供初态的返回分支。
- 第 2 行：只返回滤波输出。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/filtering.py:848` 至 `:848`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/filtering.py:753` 至 `:753`。

```python
    return out
```

逐行对应说明：

- 第 1 行：结束函数并把对象交给调用者。


### 4.3 `_validate_sos` 逐行解释

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/filter_design_utils.py:17` 至 `:21`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/filter_design_utils.py:17` 至 `:19`。

```python
def _validate_sos(sos):
    """Helper to validate a SOS input"""
    sos = np.atleast_2d(sos)
```

逐行对应说明：

- 第 1 行：定义内部验证函数。
- 第 2 行：说明它验证 SOS 输入。
- 第 3 行：将零/一维输入提升到至少二维；高维不压缩。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/filter_design_utils.py:23` 至 `:26`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/filter_design_utils.py:20` 至 `:22`。

```python
    if sos.ndim != 2:
        raise ValueError("sos array must be 2D")
    n_sections, m = sos.shape
```

逐行对应说明：

- 第 1 行：只接受恰好二维。
- 第 2 行：非二维时报错。
- 第 3 行：解包二维 shape；若维数不为 2 前面已拒绝。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/filter_design_utils.py:28` 至 `:29`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/filter_design_utils.py:23` 至 `:24`。

```python
    if m != 6:
        raise ValueError("sos array must be shape (n_sections, 6)")
```

逐行对应说明：

- 第 1 行：每节必须六个系数。
- 第 2 行：系数列数错误时报错。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/filter_design_utils.py:31` 至 `:32`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/filter_design_utils.py:25` 至 `:26`。

```python
    if not (sos[:, 3] == 1).all():
        raise ValueError("sos[:, 3] should be all ones")
```

逐行对应说明：

- 第 1 行：切出第 4 列，逐元素比较 1，`all` 汇总，再逻辑取反。
- 第 2 行：任一 $a_0\ne1$ 就拒绝；函数不自动归一化。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filter_design/filter_design_utils.py:34` 至 `:34`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filter_design/filter_design_utils.py:27` 至 `:27`。

```python
    return sos, n_sections
```

逐行对应说明：

- 第 1 行：返回验证后的数组与节数。


### 4.4 设备能力 helper 逐行解释

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:27` 至 `:27`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:27` 至 `:27`。

```python
def _get_max_smem():
```

逐行对应说明：

- 第 1 行：定义共享内存上限查询函数。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:30` 至 `:30`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:29` 至 `:29`。

```python
    device_id = cp.cuda.Device()
```

逐行对应说明：

- 第 1 行：构造当前 CUDA device 对象。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:33` 至 `:33`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:31` 至 `:31`。

```python
    return device_id.attributes["MaxSharedMemoryPerBlock"]
```

逐行对应说明：

- 第 1 行：从属性字典返回每 block 最大共享内存。

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:36` 至 `:36`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:34` 至 `:34`。

```python
def _get_max_tpb():
```

逐行对应说明：

- 第 1 行：定义线程上限查询函数。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:39` 至 `:39`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:36` 至 `:36`。

```python
    device_id = cp.cuda.Device()
```

逐行对应说明：

- 第 1 行：取得当前 CUDA device。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:42` 至 `:42`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/utils/helper_tools.py:38` 至 `:38`。

```python
    return device_id.attributes["MaxThreadsPerBlock"]
```

逐行对应说明：

- 第 1 行：返回每 block 最大线程数。


## 5. CUDA Python 包装逐行解释

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:17` 至 `:17`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:17` 至 `:17`。

```python
_SUPPORTED_TYPES = ["float32", "float64"]
```

逐行对应说明：

- 第 1 行：模块常量列出 fatbin 提供的两种 dtype 名。

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:20` 至 `:20`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:20` 至 `:20`。

```python
class _cupy_sosfilt_wrapper(object):
```

逐行对应说明：

- 第 1 行：定义继承 `object` 的可调用包装类。

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:22` 至 `:22`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:21` 至 `:21`。

```python
    def __init__(self, grid, block, smem, kernel):
```

逐行对应说明：

- 第 1 行：构造器接收 launch 配置、共享内存和 kernel。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:23` 至 `:24`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:22` 至 `:23`。

```python
        if isinstance(grid, int):
            grid = (grid,)
```

逐行对应说明：

- 第 1 行：若 grid 是单个整数，进入规范化分支。
- 第 2 行：尾逗号构造一元素元组。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:25` 至 `:26`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:24` 至 `:25`。

```python
        if isinstance(block, int):
            block = (block,)
```

逐行对应说明：

- 第 1 行：对 block 做同样检查。
- 第 2 行：规范化为一维 block 元组。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:28` 至 `:31`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:27` 至 `:30`。

```python
        self.grid = grid
        self.block = block
        self.smem = smem
        self.kernel = kernel
```

逐行对应说明：

- 第 1 行：保存 grid 到实例属性。
- 第 2 行：保存 block。
- 第 3 行：保存动态共享内存字节数。
- 第 4 行：保存 CuPy kernel 句柄。

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:33` 至 `:33`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:32` 至 `:32`。

```python
    def __call__(self, sos, x, zi):
```

逐行对应说明：

- 第 1 行：让实例可像函数一样调用。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:36` 至 `:44`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:34` 至 `:42`。

```python
        kernel_args = (
            x.shape[0],
            x.shape[1],
            sos.shape[0],
            zi.shape[2],
            sos,
            zi,
            x,
        )
```

逐行对应说明：

- 第 1 行：开始构造 kernel 位置参数元组。
- 第 2 行：参数 1：展平后的独立信号数。
- 第 3 行：参数 2：每条信号样本数。
- 第 4 行：参数 3：二阶节数。
- 第 5 行：参数 4：状态宽度，预期为 2。
- 第 6 行：参数 5：GPU 系数指针。
- 第 7 行：参数 6：GPU 初态指针。
- 第 8 行：参数 7：可原地写回的信号指针。
- 第 9 行：结束参数元组。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:47` 至 `:47`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:44` 至 `:44`。

```python
        self.kernel(self.grid, self.block, kernel_args, shared_mem=self.smem)
```

逐行对应说明：

- 第 1 行：以 grid、block、参数和动态共享内存启动 kernel。

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:50` 至 `:50`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:47` 至 `:47`。

```python
def _populate_kernel_cache(np_type, k_type):
```

逐行对应说明：

- 第 1 行：定义 fatbin 加载/缓存函数。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:53` 至 `:54`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:49` 至 `:50`。

```python
    if np_type not in _SUPPORTED_TYPES:
        raise ValueError("Datatype {} not found for '{}'".format(np_type, k_type))
```

逐行对应说明：

- 第 1 行：检查 dtype 是否在支持列表。这里 `np_type` 实际传入 `x.dtype`。
- 第 2 行：不支持时格式化 dtype 和 kernel 名并报错。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:57` 至 `:57`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:52` 至 `:52`。

```python
    if (str(np_type), k_type) in _cupy_kernel_cache:
```

逐行对应说明：

- 第 1 行：以 dtype 字符串和 kernel 类型组成缓存键。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:58` 至 `:58`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:53` 至 `:53`。

```python
        return
```

逐行对应说明：

- 第 1 行：已缓存则无值提前返回。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:61` 至 `:64`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:55` 至 `:58`。

```python
    _cupy_kernel_cache[(str(np_type), k_type)] = _get_function(
        "/filtering/_sosfilt.fatbin",
        "_cupy_" + k_type + "_" + str(np_type),
    )
```

逐行对应说明：

- 第 1 行：从二进制中取函数并写入全局缓存。
- 第 2 行：fatbin 的包内路径。
- 第 3 行：构造如 `_cupy_sosfilt_float32` 的导出符号名。
- 第 4 行：结束 `_get_function` 调用和赋值。

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:67` 至 `:69`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:61` 至 `:62`。

```python
def _get_backend_kernel(dtype, grid, block, smem, k_type):
    kernel = _cupy_kernel_cache[(dtype.name, k_type)]
```

逐行对应说明：

- 第 1 行：定义从缓存构造包装器的函数。
- 第 2 行：用 dtype 名和类型取函数句柄。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:70` 至 `:70`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:63` 至 `:63`。

```python
    if kernel:
```

逐行对应说明：

- 第 1 行：句柄 truthy 时表示找到。

#### 返回结果

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:71` 至 `:71`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:64` 至 `:64`。

```python
        return _cupy_sosfilt_wrapper(grid, block, smem, kernel)
```

逐行对应说明：

- 第 1 行：返回绑定本次 launch 参数的可调用对象。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:72` 至 `:73`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:65` 至 `:66`。

```python
    else:
        raise ValueError("Kernel {} not found in _cupy_kernel_cache".format(k_type))
```

逐行对应说明：

- 第 1 行：未找到句柄的分支。
- 第 2 行：报告缓存中没有 kernel。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:75` 至 `:75`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:68` 至 `:68`。

```python
    raise NotImplementedError("No kernel found for datatype {}".format(dtype.name))
```

逐行对应说明：

- 第 1 行：前面两分支都 return/raise，因此此行事实上不可达。

#### 定义与签名

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:78` 至 `:78`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:71` 至 `:71`。

```python
def _sosfilt(sos, x, zi):
```

逐行对应说明：

- 第 1 行：定义被公开函数调用的内部启动入口。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:81` 至 `:83`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:73` 至 `:74`。

```python
    threadsperblock = sos.shape[0]  # Up-to (1024, 1) = 1024 max per block
    blockspergrid = x.shape[0]
```

逐行对应说明：

- 第 1 行：一节映射一个 x 方向线程；原注释写明 1024 上限。
- 第 2 行：一条独立信号映射一个 block。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:85` 至 `:85`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:76` 至 `:76`。

```python
    k_type = "sosfilt"
```

逐行对应说明：

- 第 1 行：设置缓存和符号名使用的算子标识。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:88` 至 `:88`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:78` 至 `:78`。

```python
    _populate_kernel_cache(x.dtype, k_type)
```

逐行对应说明：

- 第 1 行：确保当前 dtype 的 kernel 已加载。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:90` 至 `:91`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:80` 至 `:81`。

```python
    out_size = threadsperblock
    sos_size = sos.shape[0] * sos.shape[1]
```

逐行对应说明：

- 第 1 行：每节需要一个共享节间输出槽。
- 第 2 行：系数共享区需要节数乘 6 个元素。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:94` 至 `:94`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:83` 至 `:83`。

```python
    shared_mem = (out_size + sos_size) * x.dtype.itemsize
```

逐行对应说明：

- 第 1 行：按 dtype 字节宽度计算动态共享内存。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:97` 至 `:105`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:85` 至 `:92`。

```python
    kernel = _get_backend_kernel(
        x.dtype,
        blockspergrid,
        threadsperblock,
        shared_mem,
        k_type,
    )
    print(zi.shape)
```

逐行对应说明：

- 第 1 行：开始获取带 launch 配置的包装器。
- 第 2 行：选择 kernel dtype。
- 第 3 行：传入 grid。
- 第 4 行：传入 block。
- 第 5 行：传入动态共享内存字节数。
- 第 6 行：传入算子标识。
- 第 7 行：结束调用。
- 第 8 行：无条件打印状态 shape；属于遗留诊断副作用。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:108` 至 `:108`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:94` 至 `:94`。

```python
    kernel(sos, x, zi)
```

逐行对应说明：

- 第 1 行：调用包装器并实际启动 CUDA kernel。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:111` 至 `:111`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/filtering/_sosfilt_cuda.py:96` 至 `:96`。

```python
    _print_atts(kernel)
```

逐行对应说明：

- 第 1 行：打印 kernel 属性，仍是诊断而非计算。


## 6. CUDA kernel 逐行解释

### 6.1 声明、共享内存和初态

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:18` 至 `:18`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:18` 至 `:18`。

```python
constexpr int sos_width = 6;
```

逐行对应说明：

- 第 1 行：编译期常量；每节固定六个系数。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:20` 至 `:22`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:20` 至 `:21`。

```python
template<typename T>
__device__ void _cupy_sosfilt( const int n_signals,
```

逐行对应说明：

- 第 1 行：声明类型模板，后面实例化 `float/double`。
- 第 2 行：定义只能由 GPU 代码调用且无返回值的 device 函数。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:22` 至 `:29`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:21` 至 `:28`。

```python
__device__ void _cupy_sosfilt( const int n_signals,
                               const int n_samples,
                               const int n_sections,
                               const int zi_width,
                               const T *__restrict__ sos,
                               const T *__restrict__ zi,
                               T *__restrict__ x_in,
                               T *s_buffer ) {
```

逐行对应说明：

- 第 1 行：当前 launch 的独立信号总数。
- 第 2 行：每条信号的样本数。
- 第 3 行：SOS 节数，也等于 block 线程数。
- 第 4 行：每节状态宽度，Python 准备为 2。
- 第 5 行：只读 SOS 指针；`restrict` 承诺不与其他指针别名。
- 第 6 行：只读状态指针；这个 `const` 是末状态无法写回的直接证据。
- 第 7 行：可写输入/输出缓冲区指针。
- 第 8 行：动态共享内存指针；右花括号开始函数体。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:32` 至 `:33`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:30` 至 `:31`。

```python
    T *s_out { s_buffer };
    T *s_sos { reinterpret_cast<T *>( &s_out[n_sections] ) };
```

逐行对应说明：

- 第 1 行：C++ 列表初始化；共享区开头作为节间输出。
- 第 2 行：指针偏移 `n_sections` 后作为系数区；转换保持类型 `T*`。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:36` 至 `:37`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:33` 至 `:34`。

```python
    const int tx { static_cast<int>( threadIdx.x ) };
    const int bx { static_cast<int>( blockIdx.x ) };
```

逐行对应说明：

- 第 1 行：将线程 x 索引显式转成 `int`，代表节号。
- 第 2 行：将 block x 索引转成 `int`，代表信号号。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:39` 至 `:40`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:36` 至 `:37`。

```python
    // Reset shared memory
    s_out[tx] = 0;
```

逐行对应说明：

- 第 1 行：原注释说明初始化节间输出。
- 第 2 行：每个线程清零自己的共享输出槽。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:42` 至 `:46`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:39` 至 `:42`。

```python
    // Load SOS
    // b is in s_sos[tx * sos_width + [0-2]]
    // a is in s_sos[tx * sos_width + [3-5]]
#pragma unroll sos_width
```

逐行对应说明：

- 第 1 行：原注释标记系数加载阶段。
- 第 2 行：说明共享系数每行 0–2 是分子。
- 第 3 行：说明每行 3–5 是分母。
- 第 4 行：请求编译器按固定 6 次展开下一循环。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:47` 至 `:49`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:43` 至 `:45`。

```python
    for ( int i = 0; i < sos_width; i++ ) {
        s_sos[tx * sos_width + i] = sos[tx * sos_width + i];
    }
```

逐行对应说明：

- 第 1 行：初始化、条件、递增组成 C++ `for`；每线程遍历六系数。
- 第 2 行：从全局内存复制当前节第 `i` 个系数到共享内存。
- 第 3 行：结束系数循环。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:51` 至 `:51`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:47` 至 `:47`。

```python
    // __syncthreads( );
```

逐行对应说明：

- 第 1 行：被注释掉的同步，不执行；后续循环开头才同步。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:54` 至 `:55`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:49` 至 `:50`。

```python
    T zi0 = zi[bx * n_sections * zi_width + tx * zi_width + 0];
    T zi1 = zi[bx * n_sections * zi_width + tx * zi_width + 1];
```

逐行对应说明：

- 第 1 行：按 `[signal,section,state]` 线性索引读取状态 0 到寄存器。
- 第 2 行：读取状态 1。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:58` 至 `:59`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:52` 至 `:53`。

```python
    const int load_size { n_sections - 1 };
    const int unload_size { n_samples - load_size };
```

逐行对应说明：

- 第 1 行：流水线填充延迟为节数减一。
- 第 2 行：计算排空输出的写回起点。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:61` 至 `:62`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:55` 至 `:56`。

```python
    T temp {};
    T x_n {};
```

逐行对应说明：

- 第 1 行：值初始化当前节输出为 0。
- 第 2 行：值初始化当前节输入为 0。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:64` 至 `:64`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:58` 至 `:58`。

```python
    if ( bx < n_signals ) {
```

逐行对应说明：

- 第 1 行：边界保护；当前 grid 恰等于信号数时通常恒真。


### 6.2 Loading：填充流水线

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:65` 至 `:65`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:59` 至 `:59`。

```python
        // Loading phase
```

逐行对应说明：

- 第 1 行：原注释：开始填充阶段。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:66` 至 `:67`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:60` 至 `:61`。

```python
        for ( int n = 0; n < load_size; n++ ) {
            __syncthreads( );
```

逐行对应说明：

- 第 1 行：前 `K-1` 个时刻只推进流水，不产生完整末节输出。
- 第 2 行：block 内屏障，确保上一轮所有共享输出已就绪。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:69` 至 `:73`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:62` 至 `:66`。

```python
            if ( tx == 0 ) {
                x_n = x_in[bx * n_samples + n];
            } else {
                x_n = s_out[tx - 1];
            }
```

逐行对应说明：

- 第 1 行：第 0 节选择原始输入源。
- 第 2 行：从当前信号第 `n` 个样本读取。
- 第 3 行：其余节选择前一节输出。
- 第 4 行：从共享内存读取上一节上一流水拍的结果。
- 第 5 行：结束输入来源分支。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:75` 至 `:81`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:68` 至 `:71`。

```python
            // Use direct II transposed structure
            temp = s_sos[tx * sos_width + 0] * x_n + zi0;
            zi0  = s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + zi1;
            zi1  = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;
```

逐行对应说明：

- 第 1 行：原注释标识 DF-II-T 核心。
- 第 2 行：计算本节输出 $v=b_0u+d_1$。
- 第 3 行：更新第一状态 $d_1'=b_1u-a_1v+d_2$。
- 第 4 行：更新第二状态 $d_2'=b_2u-a_2v$。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:83` 至 `:84`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:73` 至 `:74`。

```python
            s_out[tx] = temp;
        }
```

逐行对应说明：

- 第 1 行：把本节输出发布给下一节下一流水拍。
- 第 2 行：结束 Loading 循环。


### 6.3 Processing：稳定产生输出

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:86` 至 `:86`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:76` 至 `:76`。

```python
        // Processing phase
```

逐行对应说明：

- 第 1 行：原注释：流水线已填满。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:87` 至 `:88`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:77` 至 `:78`。

```python
        for ( int n = load_size; n < n_samples; n++ ) {
            __syncthreads( );
```

逐行对应说明：

- 第 1 行：从填满时刻循环到输入样本末尾。
- 第 2 行：等待所有节完成上一拍。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:89` 至 `:93`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:79` 至 `:83`。

```python
            if ( tx == 0 ) {
                x_n = x_in[bx * n_samples + n];
            } else {
                x_n = s_out[tx - 1];
            }
```

逐行对应说明：

- 第 1 行：第 0 节继续读取原输入。
- 第 2 行：读取当前输入样本。
- 第 3 行：后续节分支。
- 第 4 行：读取前一节的流水结果。
- 第 5 行：结束来源分支。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:95` 至 `:99`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:85` 至 `:88`。

```python
            // Use direct II transposed structure
            temp = s_sos[tx * sos_width + 0] * x_n + zi0;
            zi0  = s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + zi1;
            zi1  = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;
```

逐行对应说明：

- 第 1 行：说明以下三式与 Loading 相同。
- 第 2 行：当前节输出。
- 第 3 行：更新状态 0。
- 第 4 行：更新状态 1。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:102` 至 `:107`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:90` 至 `:95`。

```python
            if ( tx < load_size ) {
                s_out[tx] = temp;
            } else {
                x_in[bx * n_samples + ( n - load_size )] = temp;
            }
        }
```

逐行对应说明：

- 第 1 行：非末节仍负责传递中间结果。
- 第 2 行：写共享内存。
- 第 3 行：唯一的末节线程进入输出分支。
- 第 4 行：把成熟输出按补偿后的时间索引写回输入缓冲区。
- 第 5 行：结束末节分支。
- 第 6 行：结束 Processing 循环。


### 6.4 Unloading：排空流水线

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:109` 至 `:109`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:97` 至 `:97`。

```python
        // Unloading phase
```

逐行对应说明：

- 第 1 行：原注释：输入耗尽后排出尚在级联中的结果。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:111` 至 `:113`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:98` 至 `:100`。

```python
        for ( int n = 0; n < n_sections; n++ ) {
            __syncthreads( );
            // retire threads that are less than n
```

逐行对应说明：

- 第 1 行：最多推进 `K` 个排空拍。
- 第 2 行：同步节间共享结果。
- 第 3 行：原注释称逐步停用已完成的前端线程。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:114` 至 `:115`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:101` 至 `:102`。

```python
            if ( tx > n ) {
                x_n = s_out[tx - 1];
```

逐行对应说明：

- 第 1 行：只有节号大于当前排空拍的线程继续工作。
- 第 2 行：从前一节取得尚未排出的中间值。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:117` 至 `:121`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:104` 至 `:107`。

```python
                // Use direct II transposed structure
                temp = s_sos[tx * sos_width + 0] * x_n + zi0;
                zi0  = s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + zi1;
                zi1  = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;
```

逐行对应说明：

- 第 1 行：以下仍是同一递推。
- 第 2 行：计算排空拍的本节输出。
- 第 3 行：更新第一状态。
- 第 4 行：更新第二状态。

#### 控制流

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:123` 至 `:131`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:109` 至 `:117`。

```python
                if ( tx < load_size ) {
                    s_out[tx] = temp;
                } else {
                    x_in[bx * n_samples + ( n + unload_size )] = temp;
                }
            }
        }
    }
}
```

逐行对应说明：

- 第 1 行：非末节继续传递。
- 第 2 行：写共享输出。
- 第 3 行：末节写最终结果。
- 第 4 行：写入输出尾部对应位置。
- 第 5 行：结束末节分支。
- 第 6 行：结束活跃线程条件。
- 第 7 行：结束排空循环。
- 第 8 行：结束信号边界条件。
- 第 9 行：结束 device 模板函数。


### 6.5 float32/float64 全局入口

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:133` 至 `:133`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:119` 至 `:119`。

```python
extern "C" __global__ void __launch_bounds__( 1024 ) _cupy_sosfilt_float32( const int n_signals,
```

逐行对应说明：

- 第 1 行：用 C linkage 导出 CUDA global kernel；`__launch_bounds__(1024)` 声明最大 block 线程假设。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:133` 至 `:139`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:119` 至 `:125`。

```python
extern "C" __global__ void __launch_bounds__( 1024 ) _cupy_sosfilt_float32( const int n_signals,
                                                                            const int n_samples,
                                                                            const int n_sections,
                                                                            const int zi_width,
                                                                            const float *__restrict__ sos,
                                                                            const float *__restrict__ zi,
                                                                            float *__restrict__ x_in ) {
```

逐行对应说明：

- 第 1 行：float32 入口参数 1。
- 第 2 行：参数 2。
- 第 3 行：参数 3。
- 第 4 行：参数 4。
- 第 5 行：float32 SOS 指针。
- 第 6 行：只读 float32 状态指针。
- 第 7 行：可写 float32 信号指针并开始函数体。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:142` 至 `:142`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:127` 至 `:127`。

```python
    extern __shared__ float s_buffer_f[];
```

逐行对应说明：

- 第 1 行：声明由 launch 的 `shared_mem` 决定大小的动态共享数组。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:144` 至 `:145`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:129` 至 `:130`。

```python
    _cupy_sosfilt<float>( n_signals, n_samples, n_sections, zi_width, sos, zi, x_in, s_buffer_f );
}
```

逐行对应说明：

- 第 1 行：显式实例化/调用 `T=float` 的 device 模板。
- 第 2 行：结束 float32 global kernel。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:147` 至 `:147`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:132` 至 `:132`。

```python
extern "C" __global__ void __launch_bounds__( 1024 ) _cupy_sosfilt_float64( const int n_signals,
```

逐行对应说明：

- 第 1 行：导出 double 版本入口。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:147` 至 `:153`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:132` 至 `:138`。

```python
extern "C" __global__ void __launch_bounds__( 1024 ) _cupy_sosfilt_float64( const int n_signals,
                                                                            const int n_samples,
                                                                            const int n_sections,
                                                                            const int zi_width,
                                                                            const double *__restrict__ sos,
                                                                            const double *__restrict__ zi,
                                                                            double *__restrict__ x_in ) {
```

逐行对应说明：

- 第 1 行：double 入口参数 1。
- 第 2 行：参数 2。
- 第 3 行：参数 3。
- 第 4 行：参数 4。
- 第 5 行：double SOS 指针。
- 第 6 行：只读 double 状态指针。
- 第 7 行：可写 double 信号指针并开始函数体。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:156` 至 `:156`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:140` 至 `:140`。

```python
    extern __shared__ double s_buffer_d[];
```

逐行对应说明：

- 第 1 行：声明 double 动态共享数组。

#### 连续执行逻辑

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:158` 至 `:159`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/filtering/_sosfilt.cu:142` 至 `:143`。

```python
    _cupy_sosfilt<double>( n_signals, n_samples, n_sections, zi_width, sos, zi, x_in, s_buffer_d );
}
```

逐行对应说明：

- 第 1 行：调用 `T=double` 的同一 device 模板。
- 第 2 行：结束 float64 global kernel。


## 7. 调用链与算法总结

```text
cusignal.sosfilt
  → filtering.sosfilt
    → _validate_sos
    → CuPy dtype/axis/shape/连续内存准备
    → _get_max_smem + _get_max_tpb
    → filtering._sosfilt_cuda._sosfilt
      → _populate_kernel_cache
      → _get_backend_kernel
      → _cupy_sosfilt_wrapper.__call__
        → fatbin 全局入口 _cupy_sosfilt_float32/float64
          → device 模板 _cupy_sosfilt<T>
            → Loading → Processing → Unloading
```

执行顺序可以压缩为五步：

1. 验证 `(K,6)` SOS，要求每节 $a_0=1$。
2. 把指定滤波轴移到末尾，展平其他轴为 $B$ 条独立信号；状态变为 `[B,K,2]`。
3. 一个 CUDA block 处理一条信号，一个线程处理一个二阶节。
4. 各线程反复执行 DF-II-T 三式，通过共享内存把节输出传给下一节。
5. 末节线程按时间顺序把结果原地写回 `x`，Python 恢复原 shape 和轴。

## 8. 数学映射、边界、复杂度和风险

### 8.1 数学公式到代码

| 数学量 | 代码位置 | 对应关系 |
| --- | --- | --- |
| $u_k[n]$ | `_sosfilt.cu:63/65,80/82,102` | `x_n`：第 0 节来自输入，后续节来自 `s_out[tx-1]` |
| $v_k[n]$ | `_sosfilt.cu:69,86,105` | `temp = b0*x_n + zi0` |
| $d_{1,k}$ | `_sosfilt.cu:49,70,87,106` | `zi0` |
| $d_{2,k}$ | `_sosfilt.cu:50,71,88,107` | `zi1` |
| $H(z)=\prod_kH_k(z)$ | 共享内存 `s_out` 传递 | 每节输出作为下一节输入 |
| $a_{0,k}=1$ | `_validate_sos:25-26` | 实现没有除以 $a_0$，所以强制归一化 |

### 8.2 dtype、内存布局和边界

- Python 字符门禁允许的类别比 fatbin 实际支持范围宽；最终 `_populate_kernel_cache` 只接受 `float32/float64`。
- `x` 被强制为指定 dtype、C 连续且可写；结果原地覆盖这个新副本，不覆盖调用者原对象。
- `zi` 被复制并整理为 `[B,K,2]` 连续数组。
- `sos` 只在 dtype 不同时复制；kernel 将其搬进共享内存。
- `K` 同时受文档的 `<513`、设备每 block 最大线程数、动态共享内存上限限制。
- `N_samples > K` 是流水实现的限制，不是 SOS/DF-II-T 理论限制。
- 输入标量、SOS 非二维/非六列、$a_0\ne1$、`zi` shape 错误或不支持 dtype 都会提前失败。

### 8.3 时间与空间复杂度

设独立信号数为 $B$，每条样本数为 $N$，二阶节数为 $K$。

- 算术工作量：$O(BNK)$；每个“样本×节”执行固定数量乘加。
- 理想流水后的跨度：单 block 内约 $O(N+K)$ 个同步拍，但每拍由 $K$ 个线程并行；这不改变总工作量。
- 全局输入/输出：$O(BN)$。
- 全局状态：$O(BK)$。
- SOS 系数：$O(K)$。
- 每 block 动态共享内存：`(K + 6K) * sizeof(T) = 7K*sizeof(T)`；Python 端 `max_sections` 估算式却写成 `1 + zi.shape[2] + sos.shape[1] = 9` 个元素/节，与实际 launch 的 7 个元素/节不一致，属于保守/疑似遗留公式。

主要性能瓶颈是每个流水拍一次 `__syncthreads()`、节间串行依赖、每 block 线程利用率随填充/排空变化，以及很短信号时 $K$ 相对 $N$ 的流水开销。

### 8.4 `zf` 状态写回缺口

这是源码级确定事实，不是运行推测：

1. Python 把 `zi` 传入 kernel，并在调用后把同一数组作为 `zf` 返回。
2. 两个 global kernel 的 `zi` 形参均为 `const float*`/`const double*`。
3. device 模板把初态读入局部寄存器 `zi0/zi1`，递推会更新局部变量。
4. `_cupy_sosfilt.cu:49-115` 没有任何 `zi[...] = ...` 写回。

因此当前代码返回的 `zf` 仍是输入初态副本，而不是文档声称的最终状态。阶段二未运行测试；若以后修复，应增加分块连续滤波测试，验证“一次完整滤波”等于“前半块得到 `zf`，再以其作为后半块 `zi`”。

### 8.5 其他可见问题

- `_sosfilt_cuda.py:92` 无条件 `print(zi.shape)`，会污染普通 API 标准输出。
- `_print_atts(kernel)` 每次调用执行诊断输出，可能引入额外同步或日志成本，需查看 helper 才能确认具体副作用；它不属于核心算法，阶段二没有继续扩读无关 helper。
- `_get_backend_kernel:68` 的 `NotImplementedError` 位于穷尽的 `if/else` 之后，是不可达代码。
- Python dtype 预门禁和后端 dtype 列表不一致，错误会延迟到 kernel 缓存阶段才暴露。

## 9. 阅读顺序与检查问题

建议按以下顺序亲自阅读：

1. `filter_design_utils.py::_validate_sos`：为什么强制六列和 $a_0=1$？
2. `filtering.py::sosfilt` 670–710 行：多维 `x`、`axis` 和 `zi` 如何变成 `[B,N]`、`[B,K,2]`？
3. `_sosfilt_cuda.py::_sosfilt`：为什么 grid 是 $B$、block 是 $K$？
4. `_sosfilt.cu` 68–73 行：逐项对照阶段一 DF-II-T 三式。
5. kernel 的 Loading/Processing/Unloading：用 $K=3,N=5$ 在纸上画出每拍哪个线程处理哪个样本。
6. 回到 `filtering.py` 742–753 行：输出与状态怎样恢复 shape？
7. 搜索 kernel 中对 `zi` 的写操作，解释为什么当前 `zf` 契约未实现。

阶段二自检问题：

1. 为什么 `axis` 在构造 `x_zi_shape` 时尚未正规化，负索引仍然有效？
2. 为什么 `zi` 的状态维在 `axis+1`，而不是 `axis`？
3. `x = cp.array(..., order="C")` 为什么既是 dtype 转换也是可写副本门禁？
4. 每条信号一个 block 有什么独立性前提？
5. `__syncthreads()` 为什么不能只让部分线程执行？当前三个阶段是否满足这一规则？
6. 为什么排空条件使用 `tx > n`？
7. `n_sections=1` 时 `load_size=0`，三个阶段分别执行多少次？
8. 当前实现如何证明 `y` 的输出顺序仍与输入样本顺序一致？
9. `zf` 问题会影响一次性 `zi=None` 的 `y` 吗？会影响哪类流式调用？

阶段二参考答案：

1. Python list 原生接受 `-1` 等负索引，所以 `x_zi_shape[axis]=2` 在 axis 转正前仍能正确定位；703 行稍后用 `% x.ndim` 统一转正，供 `moveaxis` 的配对维度计算。
2. `zi` 比 `x` 多了最前面的 `n_sections` 维，因此 `x` 的原 axis 在状态数组中右移一位，成为 `axis+1`；705 行把节维 0 和状态维 `axis+1` 一起移到末两位。
3. `cp.array(x,dtype,order="C")` 同时执行共同 dtype 转换、保证 C 连续布局并新建可写副本；742 行 kernel 会把结果原地写回这个 `x`，因而不能只保留任意 stride 的只读 view。
4. 不同 block 处理展平后的不同信号行，各行拥有互不重叠的输入区间与 `[line,section,2]` 状态；只有这种数据和状态独立性才允许 block 间无同步并行。依据是 `_sosfilt_cuda.py:73-74` 的 grid/block 配置和 kernel 的 `bx` 索引。
5. `__syncthreads()` 是 block 屏障；若仍活跃的同一 block 线程在分歧分支中并非全部到达，会死锁或产生未定义行为。当前三个主循环把屏障放在分支外层，所有 block 线程先同步，再按 `tx` 分支，因此同步位置一致。
6. 排空第 `n` 拍时，前端若干节已经没有待传递样本；`tx>n` 让仍位于更后节的线程继续推进，逐拍把尾部结果送到末节。证据是 `_sosfilt.cu:98-115`。
7. 当 `K=1` 时 `load_size=0`：Loading 执行 0 次，Processing 执行 `N` 次且唯一线程每次直接写输出，Unloading 循环虽有 1 次但 `tx=0` 不满足 `tx>n=0`，所以不做递推。
8. Processing 的末节把第 `n` 拍写到 `n-load_size`，产生前 `N-K+1` 个输出；Unloading 再写到 `n+unload_size`，补齐尾部索引。两段索引连续且不重叠，因此恢复后的 `y` 保持样本顺序。
9. `zi=None` 时 Python 只返回 `y`，而局部 `zi0/zi1` 在单次 kernel 内仍正确维持递推，所以 `zf` 缺口不直接破坏该次 `y`。它会破坏需要取上一块末状态作为下一块初态的流式/分块调用，以及任何显式依赖返回 `zf` 的场景；证据是 global `zi` 为 const 且无写回。

## 10. 完整性复核

阶段二完成后，对以下学习副本逐文件去除所有符合 `^\s*(#|//) <学习注释：.*>\s*$` 的新增行，再与 `ZKX/cusignal-23.08.00` 对应只读文件逐行比较，结果全部一致：

- `python/cusignal/filtering/filtering.py`
- `python/cusignal/filtering/_sosfilt_cuda.py`
- `python/cusignal/filter_design/filter_design_utils.py`
- `python/cusignal/utils/helper_tools.py`
- `python/cusignal/filtering/__init__.py`
- `python/cusignal/__init__.py`
- `cpp/src/filtering/_sosfilt.cu`

本阶段没有修改 `ZKX/cusignal-23.08.00`，没有运行 formatter，没有构建或执行本地/远程测试，也没有进入 `ZKX/cusignal_cpp` 阶段。
