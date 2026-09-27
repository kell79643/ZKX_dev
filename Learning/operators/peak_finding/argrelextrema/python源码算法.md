# argrelextrema：cuSignal 23.08.00 Python 源码算法

## 1. 代码定位与接口概览

### 1.1 公开导入路径

```python
import cusignal
cusignal.argrelextrema(...)

from cusignal.peak_finding import argrelextrema
argrelextrema(...)
```

| 层次 | 学习副本位置 | 只读基准位置 | 符号 |
| --- | --- | --- | --- |
| 顶层导出 | `Learning/cusignal-23.08.00/python/cusignal/__init__.py:70` | `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:57` | `argrelextrema` 导入语句 |
| 子包导出 | `Learning/cusignal-23.08.00/python/cusignal/peak_finding/__init__.py:15` | `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/__init__.py:14` | `argrelextrema` 导入语句 |
| 公开函数 | `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:201` | `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:184` | `argrelextrema` |
| 布尔 helper | 学习副本同文件 `:21` | 只读基准同文件 `:21` | `_boolrelextrema` |
| GPU 调度 | `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:134` | `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:124` | `_peak_finding` |
| CUDA 一维核心 | `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:102` | `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:91` | `_cupy_boolrelextrema_1D` |
| CUDA 二维核心 | 学习副本同文件 `:182` | 只读基准同文件 `:161` | `_cupy_boolrelextrema_2D` |

### 1.2 函数签名

```python
argrelextrema(data, comparator, axis=0, order=1, mode="clip")
```

| 参数 | 默认值 | 语义 |
| --- | --- | --- |
| `data` | 无 | 输入数组；公开函数用 `cp.asarray` 转成 `cupy.ndarray` |
| `comparator` | 无 | 接收中心数组和邻居数组、返回布尔数组的二元 callable；一、二维 kernel 实际只支持六个登记的 CuPy 比较器 |
| `axis` | `0` | 沿此轴比较，其他坐标保持不变 |
| `order` | `1` | 每侧比较的样本数，必须是整数且 `>=1` |
| `mode` | `"clip"` | `clip` 为端点复制；其他非 `raise` 值在实现中走回绕路径；`raise` 最终被拒绝 |

返回值是长度等于 `data.ndim` 的 tuple。tuple 中每个元素都是整数坐标数组，所有数组长度均为候选位置数。对一维输入也返回 `(indices,)`，而不是单独数组。输出 dtype 由 `cp.nonzero` 决定，语义上是整数索引；输出不包含极值幅值。

### 1.3 shape 与 dtype 语义

- 中间布尔掩码 `results.shape == data.shape`，dtype 为 `bool`。
- 一维/二维预编译 kernel 只接受 `int32`、`int64`、`float32`、`float64`。
- 三维及以上走 CuPy 数组路径，源码没有同样的四 dtype 白名单，但 `comparator` 必须支持输入 dtype。
- 算法不改变输入数组，不分配浮点临时计算结果；高维路径会为索引、邻居和布尔结果分配 GPU 数组。

## 2. 当前算子的完整相关源码

以下摘录全部来自未加学习注释的只读基准 `ZKX/cusignal-23.08.00`。学习副本定位会因新增注释而后移。

### 2.1 必要的公开导出语句

```python
from cusignal.peak_finding.peak_finding import argrelextrema, argrelmax, argrelmin
```

该原始语句分别出现在顶层 `python/cusignal/__init__.py:57` 和子包 `python/cusignal/peak_finding/__init__.py:14`。

### 2.2 `_boolrelextrema` 完整定义

只读基准：`ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:21-77`。  
学习副本：`Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:21-94`。

```python
def _boolrelextrema(data, comparator, axis=0, order=1, mode="clip"):
    """
    Calculate the relative extrema of `data`.

    Relative extrema are calculated by finding locations where
    ``comparator(data[n], data[n+1:n+order+1])`` is True.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative extrema.
    comparator : callable
        Function to use to compare two data points.
        Should take two arrays as arguments.
    axis : int, optional
        Axis over which to select from `data`.  Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n,n+x)`` to be True.

    Returns
    -------
    extrema : ndarray
        Boolean array of the same shape as `data` that is True at an extrema,
        False otherwise.

    See also
    --------
    argrelmax, argrelmin
    """
    if (int(order) != order) or (order < 1):
        raise ValueError("Order must be an int >= 1")

    if data.ndim < 3:
        results = cp.empty(data.shape, dtype=bool)
        _peak_finding(data, comparator, axis, order, mode, results)
    else:
        datalen = data.shape[axis]
        locs = cp.arange(0, datalen)
        results = cp.ones(data.shape, dtype=bool)
        main = cp.take(data, locs, axis=axis)
        for shift in cp.arange(1, order + 1):
            if mode == "clip":
                p_locs = cp.clip(locs + shift, a_min=None, a_max=(datalen - 1))
                m_locs = cp.clip(locs - shift, a_min=0, a_max=None)
            else:
                p_locs = locs + shift
                m_locs = locs - shift
            plus = cp.take(data, p_locs, axis=axis)
            minus = cp.take(data, m_locs, axis=axis)
            results &= comparator(main, plus)
            results &= comparator(main, minus)

            if ~results.any():
                return results

    return results
```

### 2.3 `argrelextrema` 完整定义、docstring 与示例

只读基准：同文件 `:184-233`。  
学习副本：同文件 `:201-254`。

```python
def argrelextrema(data, comparator, axis=0, order=1, mode="clip"):
    """
    Calculate the relative extrema of `data`.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative extrema.
    comparator : callable
        Function to use to compare two data points.
        Should take two arrays as arguments.
    axis : int, optional
        Axis over which to select from `data`.  Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n, n+x)`` to be True.

    Returns
    -------
    extrema : tuple of ndarrays
        Indices of the maxima in arrays of integers.  ``extrema[k]`` is
        the array of indices of axis `k` of `data`.  Note that the
        return value is a tuple even when `data` is one-dimensional.

    See Also
    --------
    argrelmin, argrelmax

    Examples
    --------
    >>> from cusignal import argrelextrema
    >>> import cupy as cp
    >>> x = cp.array([2, 1, 2, 3, 2, 0, 1, 0])
    >>> argrelextrema(x, cp.greater)
    (array([3, 6]),)
    >>> y = cp.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelextrema(y, cp.less, axis=1)
    (array([0, 2]), array([2, 1]))

    """
    data = cp.asarray(data)
    results = _boolrelextrema(data, comparator, axis, order, mode)

    if mode == "raise":
        raise NotImplementedError("CuPy `take` doesn't support `mode='raise'`.")

    return cp.nonzero(results)
```

### 2.4 GPU 调度模块完整相关源码

只读基准：`ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:14-166`。  
学习副本：`Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:14-185`。

```python
import cupy as cp

from ..convolution.convolution_utils import _iDivUp
from ..utils._caches import _cupy_kernel_cache
from ..utils.helper_tools import _get_function, _get_tpb_bpg, _print_atts

_modedict = {
    cp.less: 0,
    cp.greater: 1,
    cp.less_equal: 2,
    cp.greater_equal: 3,
    cp.equal: 4,
    cp.not_equal: 5,
}

_SUPPORTED_TYPES = [
    "int32",
    "int64",
    "float32",
    "float64",
]


class _cupy_boolrelextrema_1d_wrapper(object):
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
        data,
        comp,
        axis,
        order,
        clip,
        out,
    ):

        kernel_args = (data.shape[axis], order, clip, comp, data, out)

        self.kernel(self.grid, self.block, kernel_args)


class _cupy_boolrelextrema_2d_wrapper(object):
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
        data,
        comp,
        axis,
        order,
        clip,
        out,
    ):

        kernel_args = (
            data.shape[1],
            data.shape[0],
            order,
            clip,
            comp,
            axis,
            data,
            out,
        )

        self.kernel(self.grid, self.block, kernel_args)


def _populate_kernel_cache(np_type, k_type):

    if np_type not in _SUPPORTED_TYPES:
        raise ValueError("Datatype {} not found for '{}'".format(np_type, k_type))

    if (str(np_type), k_type) in _cupy_kernel_cache:
        return

    _cupy_kernel_cache[(str(np_type), k_type)] = _get_function(
        "/peak_finding/_peak_finding.fatbin",
        "_cupy_" + k_type + "_" + str(np_type),
    )


def _get_backend_kernel(dtype, grid, block, k_type):

    kernel = _cupy_kernel_cache[(str(dtype), k_type)]
    if kernel:
        if k_type == "boolrelextrema_1D":
            return _cupy_boolrelextrema_1d_wrapper(grid, block, kernel)
        else:
            return _cupy_boolrelextrema_2d_wrapper(grid, block, kernel)
    else:
        raise ValueError("Kernel {} not found in _cupy_kernel_cache".format(k_type))


def _peak_finding(data, comparator, axis, order, mode, results):

    comp = _modedict[comparator]

    if mode == "clip":
        clip = True
    else:
        clip = False

    if data.ndim == 1:
        k_type = "boolrelextrema_1D"

        threadsperblock, blockspergrid = _get_tpb_bpg()

        _populate_kernel_cache(data.dtype, k_type)

        kernel = _get_backend_kernel(
            data.dtype,
            blockspergrid,
            threadsperblock,
            k_type,
        )
    else:
        k_type = "boolrelextrema_2D"

        threadsperblock = (16, 16)
        blockspergrid = (
            _iDivUp(data.shape[1], threadsperblock[0]),
            _iDivUp(data.shape[0], threadsperblock[1]),
        )

        _populate_kernel_cache(data.dtype, k_type)

        kernel = _get_backend_kernel(
            data.dtype,
            blockspergrid,
            threadsperblock,
            k_type,
        )

    kernel(data, comp, axis, order, clip, results)

    _print_atts(kernel)
```

### 2.5 CUDA kernel 完整相关源码

只读基准：`ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:14-253`。  
学习副本：`Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:14-281`。

```cpp
#include <nvfunctional>

///////////////////////////////////////////////////////////////////////////////
//                            FUNCTION POINTERS                              //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ __forceinline__ bool less( const T &a, const T &b ) {
    return ( a < b );
}

template<typename T>
__device__ __forceinline__ bool greater( const T &a, const T &b ) {
    return ( a > b );
}

template<typename T>
__device__ __forceinline__ bool less_equal( const T &a, const T &b ) {
    return ( a <= b );
}

template<typename T>
__device__ __forceinline__ bool greater_equal( const T &a, const T &b ) {
    return ( a >= b );
}

template<typename T>
__device__ __forceinline__ bool equal( const T &a, const T &b ) {
    return ( a == b );
}

template<typename T>
__device__ __forceinline__ bool not_equal( const T &a, const T &b ) {
    return ( a != b );
}

template<typename T>
using op_func = bool ( * )( const T &, const T & );

__device__ op_func<int> const func_i[6]      = { less, greater, less_equal, greater_equal, equal, not_equal };
__device__ op_func<long int> const func_l[6] = { less, greater, less_equal, greater_equal, equal, not_equal };
__device__ op_func<float> const func_f[6]    = { less, greater, less_equal, greater_equal, equal, not_equal };
__device__ op_func<double> const func_d[6]   = { less, greater, less_equal, greater_equal, equal, not_equal };

///////////////////////////////////////////////////////////////////////////////
//                              HELPER FUNCTIONS                             //
///////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ void clip_plus( const bool &clip, const int &n, int &plus ) {
    if ( clip ) {
        if ( plus >= n ) {
            plus = n - 1;
        }
    } else {
        if ( plus >= n ) {
            plus -= n;
        }
    }
}

__device__ __forceinline__ void clip_minus( const bool &clip, const int &n, int &minus ) {
    if ( clip ) {
        if ( minus < 0 ) {
            minus = 0;
        }
    } else {
        if ( minus < 0 ) {
            minus += n;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
//                          BOOLRELEXTREMA 1D                                //
///////////////////////////////////////////////////////////////////////////////

template<typename T, class U>
__device__ void _cupy_boolrelextrema_1D( const int  n,
                                         const int  order,
                                         const bool clip,
                                         const T *__restrict__ inp,
                                         bool *__restrict__ results,
                                         U func ) {

    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

    for ( int tid = tx; tid < n; tid += stride ) {

        const T data { inp[tid] };
        bool    temp { true };

        for ( int o = 1; o < ( order + 1 ); o++ ) {
            int plus { tid + o };
            int minus { tid - o };

            clip_plus( clip, n, plus );
            clip_minus( clip, n, minus );

            temp &= func( data, inp[plus] );
            temp &= func( data, inp[minus] );
        }
        results[tid] = temp;
    }
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_1D_int32( const int  n,
                                                                                   const int  order,
                                                                                   const bool clip,
                                                                                   const int  comp,
                                                                                   const int *__restrict__ inp,
                                                                                   bool *__restrict__ results ) {
    _cupy_boolrelextrema_1D<int, op_func<int>>( n, order, clip, inp, results, func_i[comp] );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_1D_int64( const int  n,
                                                                                   const int  order,
                                                                                   const bool clip,
                                                                                   const int  comp,
                                                                                   const long int *__restrict__ inp,
                                                                                   bool *__restrict__ results ) {
    _cupy_boolrelextrema_1D<long int, op_func<long int>>( n, order, clip, inp, results, func_l[comp] );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_1D_float32( const int  n,
                                                                                     const int  order,
                                                                                     const bool clip,
                                                                                     const int  comp,
                                                                                     const float *__restrict__ inp,
                                                                                     bool *__restrict__ results ) {
    _cupy_boolrelextrema_1D<float, op_func<float>>( n, order, clip, inp, results, func_f[comp] );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_1D_float64( const int  n,
                                                                                     const int  order,
                                                                                     const bool clip,
                                                                                     const int  comp,
                                                                                     const double *__restrict__ inp,
                                                                                     bool *__restrict__ results ) {
    _cupy_boolrelextrema_1D<double, op_func<double>>( n, order, clip, inp, results, func_d[comp] );
}

///////////////////////////////////////////////////////////////////////////////
//                          BOOLRELEXTREMA 2D                                //
///////////////////////////////////////////////////////////////////////////////

template<typename T, class U>
__device__ void _cupy_boolrelextrema_2D( const int  in_x,
                                         const int  in_y,
                                         const int  order,
                                         const bool clip,
                                         const int  axis,
                                         const T *__restrict__ inp,
                                         bool *__restrict__ results,
                                         U func ) {

    const int ty { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int tx { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };

    if ( ( tx < in_y ) && ( ty < in_x ) ) {
        int tid { tx * in_x + ty };

        const T data { inp[tid] };
        bool    temp { true };

        for ( int o = 1; o < ( order + 1 ); o++ ) {

            int plus {};
            int minus {};

            if ( axis == 0 ) {
                plus  = tx + o;
                minus = tx - o;

                clip_plus( clip, in_y, plus );
                clip_minus( clip, in_y, minus );

                plus  = plus * in_x + ty;
                minus = minus * in_x + ty;
            } else {
                plus  = ty + o;
                minus = ty - o;

                clip_plus( clip, in_x, plus );
                clip_minus( clip, in_x, minus );

                plus  = tx * in_x + plus;
                minus = tx * in_x + minus;
            }

            temp &= func( data, inp[plus] );
            temp &= func( data, inp[minus] );
        }
        results[tid] = temp;
    }
}

extern "C" __global__ void __launch_bounds__( 256 ) _cupy_boolrelextrema_2D_int32( const int  in_x,
                                                                                   const int  in_y,
                                                                                   const int  order,
                                                                                   const bool clip,
                                                                                   const int  comp,
                                                                                   const int  axis,
                                                                                   const int *__restrict__ inp,
                                                                                   bool *__restrict__ results ) {
    _cupy_boolrelextrema_2D<int, op_func<int>>( in_x, in_y, order, clip, axis, inp, results, func_i[comp] );
}

extern "C" __global__ void __launch_bounds__( 256 ) _cupy_boolrelextrema_2D_int64( const int  in_x,
                                                                                   const int  in_y,
                                                                                   const int  order,
                                                                                   const bool clip,
                                                                                   const int  comp,
                                                                                   const int  axis,
                                                                                   const long int *__restrict__ inp,
                                                                                   bool *__restrict__ results ) {
    _cupy_boolrelextrema_2D<long int, op_func<long int>>( in_x, in_y, order, clip, axis, inp, results, func_l[comp] );
}

extern "C" __global__ void __launch_bounds__( 256 ) _cupy_boolrelextrema_2D_float32( const int  in_x,
                                                                                     const int  in_y,
                                                                                     const int  order,
                                                                                     const bool clip,
                                                                                     const int  comp,
                                                                                     const int  axis,
                                                                                     const float *__restrict__ inp,
                                                                                     bool *__restrict__ results ) {
    _cupy_boolrelextrema_2D<float, op_func<float>>( in_x, in_y, order, clip, axis, inp, results, func_f[comp] );
}

extern "C" __global__ void __launch_bounds__( 256 ) _cupy_boolrelextrema_2D_float64( const int  in_x,
                                                                                     const int  in_y,
                                                                                     const int  order,
                                                                                     const bool clip,
                                                                                     const int  comp,
                                                                                     const int  axis,
                                                                                     const double *__restrict__ inp,
                                                                                     bool *__restrict__ results ) {
    _cupy_boolrelextrema_2D<double, op_func<double>>( in_x, in_y, order, clip, axis, inp, results, func_d[comp] );
}
```

## 3. docstring 逐行翻译与解释

### 3.1 `_boolrelextrema` docstring

#### docstring 概要与起始界限

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:22` 至 `:23`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:22` 至 `:23`。

```python
    """
    Calculate the relative extrema of `data`.
```

逐行对应说明：

- 第 1 行：开始函数 docstring；这是运行时可通过 `__doc__` 读取的字符串。
- 第 2 行：“计算 `data` 的相对极值。”这里 helper 实际返回布尔掩码。

#### docstring 中的邻域比较判据

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:25` 至 `:26`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:25` 至 `:26`。

```python
    Relative extrema are calculated by finding locations where
    ``comparator(data[n], data[n+1:n+order+1])`` is True.
```

逐行对应说明：

- 第 1 行：“通过查找满足下述条件的位置来计算相对极值。”
- 第 2 行：表达中心与正向邻域比较的意图；真实实现还逐个比较负向邻居，并要求全部结果为真。

#### 参数文档与接口约束

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:28` 至 `:39`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:28` 至 `:39`。

```python
    Parameters
    ----------
    data : ndarray
        Array in which to find the relative extrema.
    comparator : callable
        Function to use to compare two data points.
        Should take two arrays as arguments.
    axis : int, optional
        Axis over which to select from `data`.  Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n,n+x)`` to be True.
```

逐行对应说明：

- 第 1 行：参数章节标题。
- 第 2 行：NumPy docstring 风格的标题下划线。
- 第 3 行：`data` 应为数组；公开函数会先转换为 CuPy ndarray。
- 第 4 行：“要在其中寻找相对极值的数组。”
- 第 5 行：`comparator` 是可调用对象。
- 第 6 行：“用于比较两个数据点的函数。”
- 第 7 行：它应接收两个数组；实际还需返回可与 `results` 逐元素逻辑与的布尔数组。
- 第 8 行：`axis` 是可选整数轴号。
- 第 9 行：沿该轴筛选；默认轴 0。
- 第 10 行：`order` 是可选整数。
- 第 11 行：每侧参与比较的样本数。
- 第 12 行：所有相应偏移比较均为真时，中心才保留。

#### 返回值文档与 shape 语义

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:41` 至 `:45`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:41` 至 `:45`。

```python
    Returns
    -------
    extrema : ndarray
        Boolean array of the same shape as `data` that is True at an extrema,
        False otherwise.
```

逐行对应说明：

- 第 1 行：返回值章节标题。
- 第 2 行：返回值标题下划线。
- 第 3 行：返回对象在 docstring 中命名为 `extrema`，类型为 ndarray。
- 第 4 行：返回与输入 shape 相同的布尔数组，极值候选位置为真。
- 第 5 行：其他位置为假。

#### 相关 API 文档与 docstring 结束

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:47` 至 `:50`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:47` 至 `:50`。

```python
    See also
    --------
    argrelmax, argrelmin
    """
```

逐行对应说明：

- 第 1 行：相关 API 章节。
- 第 2 行：相关 API 标题下划线。
- 第 3 行：指向固定使用 `greater`/`less` 的便捷函数。
- 第 4 行：docstring 结束。


学习副本对应 `peak_finding.py:22-50`，该 docstring 前没有插入学习注释，因此这一段与基准行号一致。

### 3.2 `argrelextrema` docstring

#### docstring 概要与起始界限

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:202` 至 `:203`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:185` 至 `:186`。

```python
    """
    Calculate the relative extrema of `data`.
```

逐行对应说明：

- 第 1 行：开始公开函数 docstring；学习副本对应行 202。
- 第 2 行：“计算 `data` 的相对极值。”

#### 参数文档与接口约束

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:205` 至 `:216`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:188` 至 `:199`。

```python
    Parameters
    ----------
    data : ndarray
        Array in which to find the relative extrema.
    comparator : callable
        Function to use to compare two data points.
        Should take two arrays as arguments.
    axis : int, optional
        Axis over which to select from `data`.  Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n, n+x)`` to be True.
```

逐行对应说明：

- 第 1 行：参数章节标题。
- 第 2 行：参数标题下划线。
- 第 3 行：输入数组。
- 第 4 行：“要在其中寻找相对极值的数组。”
- 第 5 行：二元比较函数。
- 第 6 行：比较两个数据点或两个同 shape 数组。
- 第 7 行：它应接受中心数组与邻居数组。
- 第 8 行：可选检测轴。
- 第 9 行：默认沿轴 0。
- 第 10 行：可选邻域阶数。
- 第 11 行：每侧比较 `order` 个点。
- 第 12 行：中心对这些邻居的比较都必须通过。

#### 返回值文档与 shape 语义

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:218` 至 `:222`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:201` 至 `:205`。

```python
    Returns
    -------
    extrema : tuple of ndarrays
        Indices of the maxima in arrays of integers.  ``extrema[k]`` is
        the array of indices of axis `k` of `data`.  Note that the
```

逐行对应说明：

- 第 1 行：返回值章节。
- 第 2 行：返回值标题下划线。
- 第 3 行：返回 ndarray 组成的 tuple。
- 第 4 行：原文写 maxima 并不严谨：具体是最大、最小还是其他局部模式取决于 comparator。
- 第 5 行：`extrema[k]` 保存每个命中位置在第 `k` 轴上的坐标。

#### 一维输入仍返回坐标 tuple

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:223` 至 `:223`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:206` 至 `:206`。

```python
        return value is a tuple even when `data` is one-dimensional.
```

逐行对应说明：

- 第 1 行：即使输入是一维，仍返回单元素 tuple。

#### 相关 API 文档与 docstring 结束

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:225` 至 `:227`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:208` 至 `:210`。

```python
    See Also
    --------
    argrelmin, argrelmax
```

逐行对应说明：

- 第 1 行：相关 API 标题。
- 第 2 行：标题下划线。
- 第 3 行：相关的固定最小/最大比较器入口。

#### docstring 示例输入与调用

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:229` 至 `:240`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:212` 至 `:223`。

```python
    Examples
    --------
    >>> from cusignal import argrelextrema
    >>> import cupy as cp
    >>> x = cp.array([2, 1, 2, 3, 2, 0, 1, 0])
    >>> argrelextrema(x, cp.greater)
    (array([3, 6]),)
    >>> y = cp.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelextrema(y, cp.less, axis=1)
```

逐行对应说明：

- 第 1 行：示例章节标题。
- 第 2 行：示例标题下划线。
- 第 3 行：从顶层包导入公开函数，验证顶层导出链。
- 第 4 行：导入 CuPy，并使用 `cp` 别名。
- 第 5 行：创建长度 8 的一维 GPU 数组。
- 第 6 行：用严格大于比较器和默认 `order=1, mode="clip"` 查局部最大。
- 第 7 行：索引 3、6 命中；末尾逗号表示一维坐标仍包装为 tuple。
- 第 8 行：开始构造三行四列的二维数组。
- 第 9 行：doctest 续行提示符，不是 Python 源文件里的省略实现。
- 第 10 行：完成二维数组构造。
- 第 11 行：docstring 中独立的 doctest 续行提示；不参与计算。
- 第 12 行：沿每一行检测严格局部最小。

#### 二维示例的坐标 tuple 输出

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:241` 至 `:241`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:224` 至 `:224`。

```python
    (array([0, 2]), array([2, 1]))
```

逐行对应说明：

- 第 1 行：两个命中坐标分别是 `(0,2)` 与 `(2,1)`。

#### 结束 docstring

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:243` 至 `:243`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:226` 至 `:226`。

```python
    """
```

逐行对应说明：

- 第 1 行：docstring 结束；学习副本对应行 243。


## 4. 源代码逐行解释

### 4.1 公开导出语句

#### 顶层 `cusignal` 公开导出

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/__init__.py:70`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/__init__.py:57`。

```python
from cusignal.peak_finding.peak_finding import argrelextrema, argrelmax, argrelmin
```

逐行对应说明：

- 第 1 行：绝对导入把实现模块中的三个名字绑定到 `cusignal` 顶层模块，因而可调用 `cusignal.argrelextrema`；同一条原始语句还导出两个便捷入口。

#### `peak_finding` 子包公开导出

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/__init__.py:15`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/__init__.py:14`。

```python
from cusignal.peak_finding.peak_finding import argrelextrema, argrelmax, argrelmin
```

逐行对应说明：

- 第 1 行：把实现模块中的三个函数绑定到 `cusignal.peak_finding` 子包，因而也可写 `from cusignal.peak_finding import argrelextrema`。


### 4.2 `_boolrelextrema` 函数体逐行解释

docstring 行已在 3.1 节逐行覆盖，下表继续覆盖签名与每一行非空可执行源码。

#### 定义：`def _boolrelextrema(data, comparator, axis=0, order=1, mode="clip"):`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:21` 至 `:21`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:21` 至 `:21`。

```python
def _boolrelextrema(data, comparator, axis=0, order=1, mode="clip"):
```

逐行对应说明：

- 第 1 行：用 `def` 定义私有 helper；默认轴 0、阶数 1、边界模式 `clip`。前导下划线表示内部 API 约定。

#### 条件分支：`if (int(order) != order) or (order < 1):`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:52` 至 `:54`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:51` 至 `:52`。

```python
    if (int(order) != order) or (order < 1):
        raise ValueError("Order must be an int >= 1")
```

逐行对应说明：

- 第 1 行：`int(order) != order` 检查整数性，`or` 再检查下界；任一为真进入异常分支。
- 第 2 行：构造并抛出 `ValueError`，立即终止函数。

#### 条件分支：`if data.ndim < 3:`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:57` 至 `:61`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:54` 至 `:56`。

```python
    if data.ndim < 3:
        results = cp.empty(data.shape, dtype=bool)
        _peak_finding(data, comparator, axis, order, mode, results)
```

逐行对应说明：

- 第 1 行：按维数分派；一维、二维走预编译 kernel。
- 第 2 行：分配同 shape 布尔 GPU 数组；`empty` 不初始化内容，必须由 kernel 全量写入。
- 第 3 行：调用直接 GPU 调度 helper，并通过可变输出参数 `results` 接收结果。

#### else 路径：`datalen = data.shape[axis]`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:62` 至 `:70`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:57` 至 `:61`。

```python
    else:
        datalen = data.shape[axis]
        locs = cp.arange(0, datalen)
        results = cp.ones(data.shape, dtype=bool)
        main = cp.take(data, locs, axis=axis)
```

逐行对应说明：

- 第 1 行：三维及以上进入 CuPy 数组实现。
- 第 2 行：取目标轴长度；非法轴会由索引操作触发异常。
- 第 3 行：在 GPU 上生成合法轴索引 `[0,...,datalen-1]`。
- 第 4 行：以全真布尔掩码作为逻辑与累积的单位元。
- 第 5 行：沿指定轴取全部合法位置；等价于得到中心值数组。

#### 循环：`for shift in cp.arange(1, order + 1):`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:72` 至 `:72`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:62` 至 `:62`。

```python
        for shift in cp.arange(1, order + 1):
```

逐行对应说明：

- 第 1 行：遍历闭区间 `1..order`；Python 上界需写 `order+1`。这里迭代 CuPy 标量可能带来逐轮同步/调度成本。

#### 条件分支：`if mode == "clip":`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:74` 至 `:76`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:63` 至 `:65`。

```python
            if mode == "clip":
                p_locs = cp.clip(locs + shift, a_min=None, a_max=(datalen - 1))
                m_locs = cp.clip(locs - shift, a_min=0, a_max=None)
```

逐行对应说明：

- 第 1 行：仅精确字符串 `clip` 进入裁剪分支。
- 第 2 行：正向索引上限裁到 `datalen-1`，下限不设。
- 第 3 行：负向索引下限裁到 0，上限不设。

#### else 路径：`p_locs = locs + shift`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:77` 至 `:87`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:66` 至 `:72`。

```python
            else:
                p_locs = locs + shift
                m_locs = locs - shift
            plus = cp.take(data, p_locs, axis=axis)
            minus = cp.take(data, m_locs, axis=axis)
            results &= comparator(main, plus)
            results &= comparator(main, minus)
```

逐行对应说明：

- 第 1 行：其他模式保留越界索引，依赖 CuPy 回绕。
- 第 2 行：生成正向邻居索引。
- 第 3 行：生成负向邻居索引。
- 第 4 行：沿 axis 收集正向邻居，shape 与 data 相同。
- 第 5 行：沿 axis 收集负向邻居。
- 第 6 行：调用比较器并原位逻辑与；此前失败的位置不会恢复。
- 第 7 行：再累积负向比较，落实两侧对称邻域。

#### 条件分支：`if ~results.any():`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:90` 至 `:90`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:74` 至 `:74`。

```python
            if ~results.any():
```

逐行对应说明：

- 第 1 行：`results.any()` 判断是否仍有真值；`~` 对 CuPy 布尔标量取反。此条件的真值求值可能触发设备到主机同步。

#### 返回：`return results`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:91` 至 `:91`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:75` 至 `:75`。

```python
                return results
```

逐行对应说明：

- 第 1 行：全假时提前返回，避免更远偏移的无效工作。

#### 返回：`return results`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:94` 至 `:94`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:77` 至 `:77`。

```python
    return results
```

逐行对应说明：

- 第 1 行：一/二维 kernel 完成或高维循环结束后返回布尔掩码。


### 4.3 `argrelextrema` 函数体逐行解释

#### 定义：`def argrelextrema(data, comparator, axis=0, order=1, mode="clip"):`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:201` 至 `:201`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:184` 至 `:184`。

```python
def argrelextrema(data, comparator, axis=0, order=1, mode="clip"):
```

逐行对应说明：

- 第 1 行：定义公开 API；参数原样传给 helper。

#### 计算或更新：`data`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:245` 至 `:247`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:227` 至 `:228`。

```python
    data = cp.asarray(data)
    results = _boolrelextrema(data, comparator, axis, order, mode)
```

逐行对应说明：

- 第 1 行：转成 CuPy ndarray；已有兼容 CuPy 数组通常避免复制，CPU 输入需要传到 GPU。
- 第 2 行：获得同 shape 布尔候选掩码。

#### 条件分支：`if mode == "raise":`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:250` 至 `:251`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:230` 至 `:231`。

```python
    if mode == "raise":
        raise NotImplementedError("CuPy `take` doesn't support `mode='raise'`.")
```

逐行对应说明：

- 第 1 行：单独识别不支持的 `raise` 模式。注意检查发生在 helper 调用之后。
- 第 2 行：明确报告 CuPy `take` 不支持该模式。

#### 返回：`return cp.nonzero(results)`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:254` 至 `:254`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/peak_finding.py:233` 至 `:233`。

```python
    return cp.nonzero(results)
```

逐行对应说明：

- 第 1 行：返回所有真值的坐标 tuple；会分配整数索引数组。


### 4.4 GPU 调度模块逐行解释

学习副本因注释发生偏移，五个符号起点依次为 `:39`、`:68`、`:104`、`:121`、`:134`；只读基准起点依次为 `:37`、`:63`、`:98`、`:112`、`:124`。下表以只读基准行号逐行对应，完整学习副本范围是 `:14-185`。

#### 导入依赖：`import cupy as cp`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:14` 至 `:14`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:14` 至 `:14`。

```python
import cupy as cp
```

逐行对应说明：

- 第 1 行：导入 CuPy 并绑定别名 `cp`。

#### 导入依赖：`from ..convolution.convolution_utils import _iDivUp`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:16` 至 `:18`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:16` 至 `:18`。

```python
from ..convolution.convolution_utils import _iDivUp
from ..utils._caches import _cupy_kernel_cache
from ..utils.helper_tools import _get_function, _get_tpb_bpg, _print_atts
```

逐行对应说明：

- 第 1 行：相对导入整数向上除法 helper，用于二维 grid 尺寸。
- 第 2 行：导入共享 kernel cache 字典。
- 第 3 行：一次导入 fatbin 函数加载、线程配置和调试输出三个 helper。

#### 计算或更新：`_modedict`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:21` 至 `:28`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:20` 至 `:27`。

```python
_modedict = {
    cp.less: 0,
    cp.greater: 1,
    cp.less_equal: 2,
    cp.greater_equal: 3,
    cp.equal: 4,
    cp.not_equal: 5,
}
```

逐行对应说明：

- 第 1 行：开始比较器对象到整数操作码的字典。
- 第 2 行：严格小于映射到操作码 0。
- 第 3 行：严格大于映射到 1。
- 第 4 行：小于等于映射到 2。
- 第 5 行：大于等于映射到 3。
- 第 6 行：等于映射到 4。
- 第 7 行：不等于映射到 5。
- 第 8 行：结束字典；key 是函数对象，不是字符串名称。

#### 计算或更新：`_SUPPORTED_TYPES`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:31` 至 `:36`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:29` 至 `:34`。

```python
_SUPPORTED_TYPES = [
    "int32",
    "int64",
    "float32",
    "float64",
]
```

逐行对应说明：

- 第 1 行：开始预编译 kernel dtype 白名单。
- 第 2 行：32 位有符号整数。
- 第 3 行：64 位有符号整数。
- 第 4 行：单精度浮点。
- 第 5 行：双精度浮点。
- 第 6 行：结束白名单。

#### 定义：`class _cupy_boolrelextrema_1d_wrapper(object):`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:39` 至 `:39`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:37` 至 `:37`。

```python
class _cupy_boolrelextrema_1d_wrapper(object):
```

逐行对应说明：

- 第 1 行：定义一维 kernel 的 Python 可调用包装类；显式继承 `object`。

#### 定义：`def __init__(self, grid, block, kernel):`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:41` 至 `:41`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:38` 至 `:38`。

```python
    def __init__(self, grid, block, kernel):
```

逐行对应说明：

- 第 1 行：构造器接收启动配置和底层 kernel。

#### 条件分支：`if isinstance(grid, int):`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:42` 至 `:43`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:39` 至 `:40`。

```python
        if isinstance(grid, int):
            grid = (grid,)
```

逐行对应说明：

- 第 1 行：若 grid 是单个整数，则需规范化为 tuple。
- 第 2 行：尾随逗号创建单元素 tuple。

#### 条件分支：`if isinstance(block, int):`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:44` 至 `:45`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:41` 至 `:42`。

```python
        if isinstance(block, int):
            block = (block,)
```

逐行对应说明：

- 第 1 行：同样检查 block。
- 第 2 行：规范化一维 block。

#### 计算或更新：`self.grid`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:47` 至 `:49`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:44` 至 `:46`。

```python
        self.grid = grid
        self.block = block
        self.kernel = kernel
```

逐行对应说明：

- 第 1 行：保存 grid 为实例属性。
- 第 2 行：保存 block。
- 第 3 行：保存可调用底层 kernel 句柄。

#### 定义：`def __call__(`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:51` 至 `:59`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:48` 至 `:56`。

```python
    def __call__(
        self,
        data,
        comp,
        axis,
        order,
        clip,
        out,
    ):
```

逐行对应说明：

- 第 1 行：开始定义实例调用协议，使包装器可写成 `kernel(...)`。
- 第 2 行：当前实例参数。
- 第 3 行：输入 CuPy 数组。
- 第 4 行：比较器操作码。
- 第 5 行：检测轴；一维时通常为 0，也用于取得轴长。
- 第 6 行：邻域阶数。
- 第 7 行：布尔边界标志。
- 第 8 行：预分配布尔输出。
- 第 9 行：结束多行函数签名并开始函数体。

#### 组装 CUDA kernel 参数

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:62` 至 `:62`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:58` 至 `:58`。

```python
        kernel_args = (data.shape[axis], order, clip, comp, data, out)
```

逐行对应说明：

- 第 1 行：按 CUDA 入口声明顺序组装六个实参；第一项是 `data.shape[axis]`。

#### 启动底层 CUDA kernel

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:65` 至 `:65`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:60` 至 `:60`。

```python
        self.kernel(self.grid, self.block, kernel_args)
```

逐行对应说明：

- 第 1 行：使用 CuPy RawKernel 调用约定启动 kernel。

#### 定义：`class _cupy_boolrelextrema_2d_wrapper(object):`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:68` 至 `:68`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:63` 至 `:63`。

```python
class _cupy_boolrelextrema_2d_wrapper(object):
```

逐行对应说明：

- 第 1 行：定义二维包装类。

#### 定义：`def __init__(self, grid, block, kernel):`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:69` 至 `:69`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:64` 至 `:64`。

```python
    def __init__(self, grid, block, kernel):
```

逐行对应说明：

- 第 1 行：二维包装器构造器，参数与一维相同。

#### 条件分支：`if isinstance(grid, int):`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:70` 至 `:71`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:65` 至 `:66`。

```python
        if isinstance(grid, int):
            grid = (grid,)
```

逐行对应说明：

- 第 1 行：兼容整数 grid 输入。
- 第 2 行：转为 tuple。

#### 条件分支：`if isinstance(block, int):`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:72` 至 `:73`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:67` 至 `:68`。

```python
        if isinstance(block, int):
            block = (block,)
```

逐行对应说明：

- 第 1 行：兼容整数 block。
- 第 2 行：转为 tuple。

#### 计算或更新：`self.grid`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:75` 至 `:77`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:70` 至 `:72`。

```python
        self.grid = grid
        self.block = block
        self.kernel = kernel
```

逐行对应说明：

- 第 1 行：保存二维 grid。
- 第 2 行：保存二维 block。
- 第 3 行：保存底层 kernel。

#### 定义：`def __call__(`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:79` 至 `:87`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:74` 至 `:82`。

```python
    def __call__(
        self,
        data,
        comp,
        axis,
        order,
        clip,
        out,
    ):
```

逐行对应说明：

- 第 1 行：开始二维包装器调用签名。
- 第 2 行：实例本身。
- 第 3 行：二维输入。
- 第 4 行：比较器操作码。
- 第 5 行：要比较的轴。
- 第 6 行：邻域阶数。
- 第 7 行：边界布尔标志。
- 第 8 行：布尔输出。
- 第 9 行：结束签名。

#### 组装 CUDA kernel 参数

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:90` 至 `:99`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:84` 至 `:93`。

```python
        kernel_args = (
            data.shape[1],
            data.shape[0],
            order,
            clip,
            comp,
            axis,
            data,
            out,
        )
```

逐行对应说明：

- 第 1 行：开始构造二维 CUDA 参数 tuple。
- 第 2 行：`in_x`：列数。
- 第 3 行：`in_y`：行数。
- 第 4 行：传入邻域阶数。
- 第 5 行：传入边界标志。
- 第 6 行：传入比较器操作码。
- 第 7 行：传入检测轴。
- 第 8 行：传入输入设备指针。
- 第 9 行：传入输出设备指针。
- 第 10 行：结束参数 tuple。

#### 启动底层 CUDA kernel

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:101` 至 `:101`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:95` 至 `:95`。

```python
        self.kernel(self.grid, self.block, kernel_args)
```

逐行对应说明：

- 第 1 行：按保存的二维启动配置执行 kernel。

#### 定义：`def _populate_kernel_cache(np_type, k_type):`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:104` 至 `:104`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:98` 至 `:98`。

```python
def _populate_kernel_cache(np_type, k_type):
```

逐行对应说明：

- 第 1 行：定义“确保 kernel 已缓存”的 helper。

#### 条件分支：`if np_type not in _SUPPORTED_TYPES:`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:107` 至 `:108`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:100` 至 `:101`。

```python
    if np_type not in _SUPPORTED_TYPES:
        raise ValueError("Datatype {} not found for '{}'".format(np_type, k_type))
```

逐行对应说明：

- 第 1 行：用 dtype 与字符串列表比较；NumPy/CuPy dtype 可与相应字符串相等。
- 第 2 行：不支持的 dtype 抛错；`format` 填入 dtype 和 kernel 类型。

#### 条件分支：`if (str(np_type), k_type) in _cupy_kernel_cache:`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:111` 至 `:111`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:103` 至 `:103`。

```python
    if (str(np_type), k_type) in _cupy_kernel_cache:
```

逐行对应说明：

- 第 1 行：以 `(dtype字符串, kernel类型)` 二元 tuple 查询 cache。

#### kernel 已缓存时提前返回

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:112` 至 `:112`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:104` 至 `:104`。

```python
        return
```

逐行对应说明：

- 第 1 行：已缓存时直接返回 `None`。

#### 计算或更新：`_cupy_kernel_cache[(str(np_type), k_type)]`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:115` 至 `:118`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:106` 至 `:109`。

```python
    _cupy_kernel_cache[(str(np_type), k_type)] = _get_function(
        "/peak_finding/_peak_finding.fatbin",
        "_cupy_" + k_type + "_" + str(np_type),
    )
```

逐行对应说明：

- 第 1 行：未缓存时加载函数并存入共享字典。
- 第 2 行：fatbin 资源路径。
- 第 3 行：拼出符号名，如 `_cupy_boolrelextrema_1D_float32`。
- 第 4 行：结束 `_get_function` 调用和赋值。

#### 定义：`def _get_backend_kernel(dtype, grid, block, k_type):`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:121` 至 `:121`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:112` 至 `:112`。

```python
def _get_backend_kernel(dtype, grid, block, k_type):
```

逐行对应说明：

- 第 1 行：定义从 cache 构造包装器的 helper。

#### 计算或更新：`kernel`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:124` 至 `:124`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:114` 至 `:114`。

```python
    kernel = _cupy_kernel_cache[(str(dtype), k_type)]
```

逐行对应说明：

- 第 1 行：直接索引 cache；key 缺失会产生 `KeyError`。

#### 条件分支：`if kernel:`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:125` 至 `:125`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:115` 至 `:115`。

```python
    if kernel:
```

逐行对应说明：

- 第 1 行：检查取得的句柄是否为真。

#### 条件分支：`if k_type == "boolrelextrema_1D":`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:126` 至 `:126`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:116` 至 `:116`。

```python
        if k_type == "boolrelextrema_1D":
```

逐行对应说明：

- 第 1 行：区分一维符号类型。

#### 返回：`return _cupy_boolrelextrema_1d_wrapper(grid, block, kernel)`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:127` 至 `:127`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:117` 至 `:117`。

```python
            return _cupy_boolrelextrema_1d_wrapper(grid, block, kernel)
```

逐行对应说明：

- 第 1 行：返回一维包装器实例。

#### 非一维类型选择二维 wrapper

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:128` 至 `:128`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:118` 至 `:118`。

```python
        else:
```

逐行对应说明：

- 第 1 行：非一维时进入二维包装。

#### 返回：`return _cupy_boolrelextrema_2d_wrapper(grid, block, kernel)`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:129` 至 `:129`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:119` 至 `:119`。

```python
            return _cupy_boolrelextrema_2d_wrapper(grid, block, kernel)
```

逐行对应说明：

- 第 1 行：返回二维包装器。

#### else 路径：`raise ValueError("Kernel {} not found in _cupy_kernel_cache".format(k_ty…`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:130` 至 `:131`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:120` 至 `:121`。

```python
    else:
        raise ValueError("Kernel {} not found in _cupy_kernel_cache".format(k_type))
```

逐行对应说明：

- 第 1 行：kernel 句柄为空时进入异常分支。
- 第 2 行：报告 cache 中没有有效 kernel。

#### 定义：`def _peak_finding(data, comparator, axis, order, mode, results):`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:134` 至 `:134`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:124` 至 `:124`。

```python
def _peak_finding(data, comparator, axis, order, mode, results):
```

逐行对应说明：

- 第 1 行：定义一/二维后端总调度函数；通过 `results` 原位输出。

#### 计算或更新：`comp`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:137` 至 `:137`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:126` 至 `:126`。

```python
    comp = _modedict[comparator]
```

逐行对应说明：

- 第 1 行：将比较器对象查表转成整数；自定义 callable 或未登记对象会 `KeyError`。

#### 条件分支：`if mode == "clip":`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:140` 至 `:141`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:128` 至 `:129`。

```python
    if mode == "clip":
        clip = True
```

逐行对应说明：

- 第 1 行：识别端点裁剪模式。
- 第 2 行：传给 CUDA 的边界标志设真。

#### else 路径：`clip = False`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:142` 至 `:143`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:130` 至 `:131`。

```python
    else:
        clip = False
```

逐行对应说明：

- 第 1 行：其他模式分支。
- 第 2 行：CUDA helper 将采用回绕分支。

#### 条件分支：`if data.ndim == 1:`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:146` 至 `:147`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:133` 至 `:134`。

```python
    if data.ndim == 1:
        k_type = "boolrelextrema_1D"
```

逐行对应说明：

- 第 1 行：一维输入分支。
- 第 2 行：选择一维 kernel 名称主体。

#### 计算或更新：`threadsperblock, blockspergrid`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:149` 至 `:149`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:136` 至 `:136`。

```python
        threadsperblock, blockspergrid = _get_tpb_bpg()
```

逐行对应说明：

- 第 1 行：获取项目统一的一维线程数与 block 数。

#### 确保 kernel 已进入缓存

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:152` 至 `:152`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:138` 至 `:138`。

```python
        _populate_kernel_cache(data.dtype, k_type)
```

逐行对应说明：

- 第 1 行：确保对应 dtype 的一维符号已加载。

#### 计算或更新：`kernel`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:154` 至 `:159`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:140` 至 `:145`。

```python
        kernel = _get_backend_kernel(
            data.dtype,
            blockspergrid,
            threadsperblock,
            k_type,
        )
```

逐行对应说明：

- 第 1 行：开始取得一维包装器。
- 第 2 行：传 dtype。
- 第 3 行：传 grid。
- 第 4 行：传 block。
- 第 5 行：传一维类型名。
- 第 6 行：结束调用并把包装器赋给 `kernel`。

#### else 路径：`k_type = "boolrelextrema_2D"`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:160` 至 `:162`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:146` 至 `:147`。

```python
    else:
        k_type = "boolrelextrema_2D"
```

逐行对应说明：

- 第 1 行：调用方仅对一、二维进入此模块，所以这里是二维分支。
- 第 2 行：选择二维名称主体。

#### 计算或更新：`threadsperblock`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:165` 至 `:170`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:149` 至 `:153`。

```python
        threadsperblock = (16, 16)
        blockspergrid = (
            _iDivUp(data.shape[1], threadsperblock[0]),
            _iDivUp(data.shape[0], threadsperblock[1]),
        )
```

逐行对应说明：

- 第 1 行：每 block 为 16×16=256 个线程。
- 第 2 行：开始计算二维 grid。
- 第 3 行：列方向 block 数向上取整。
- 第 4 行：行方向 block 数向上取整。
- 第 5 行：结束 grid tuple。

#### 确保 kernel 已进入缓存

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:172` 至 `:172`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:155` 至 `:155`。

```python
        _populate_kernel_cache(data.dtype, k_type)
```

逐行对应说明：

- 第 1 行：加载/复用二维 dtype 符号。

#### 计算或更新：`kernel`

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:174` 至 `:179`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:157` 至 `:162`。

```python
        kernel = _get_backend_kernel(
            data.dtype,
            blockspergrid,
            threadsperblock,
            k_type,
        )
```

逐行对应说明：

- 第 1 行：开始构造二维包装器。
- 第 2 行：传 dtype。
- 第 3 行：传二维 grid。
- 第 4 行：传 `(16,16)` block。
- 第 5 行：传二维类型名。
- 第 6 行：结束调用。

#### 启动已选择的一维或二维后端

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:182` 至 `:182`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:164` 至 `:164`。

```python
    kernel(data, comp, axis, order, clip, results)
```

逐行对应说明：

- 第 1 行：调用包装器，组装底层参数并异步启动 CUDA kernel。

#### 输出可选 kernel 调试属性

定位：学习副本 `Learning/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:185` 至 `:185`；只读基准 `ZKX/cusignal-23.08.00/python/cusignal/peak_finding/_peak_finding_cuda.py:166` 至 `:166`。

```python
    _print_atts(kernel)
```

逐行对应说明：

- 第 1 行：调试辅助；不改变结果。


### 4.5 CUDA 比较器、边界 helper 与一维 kernel 逐行解释

学习副本关键起点：`clip_plus:71`、`clip_minus:84`、一维核心 `:102`、四个入口 `:140/:149/:158/:167`；基准对应 `:62/:74/:91/:120/:129/:138/:147`。下表逐行覆盖基准 `:14-154` 的全部非空内容。

#### 引入 CUDA 函数指针依赖

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:14` 至 `:14`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:14` 至 `:14`。

```python
#include <nvfunctional>
```

逐行对应说明：

- 第 1 行：预处理器包含 NVIDIA 的函数对象/函数指针支持头文件。

#### 源码区域：FUNCTION POINTERS

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:16` 至 `:18`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:16` 至 `:18`。

```python
///////////////////////////////////////////////////////////////////////////////
//                            FUNCTION POINTERS                              //
///////////////////////////////////////////////////////////////////////////////
```

逐行对应说明：

- 第 1 行：注释分隔线，无执行效果。
- 第 2 行：标记比较函数指针区域。
- 第 3 行：结束区域标题。

#### 模板定义：`__device__ __forceinline__ bool less( const T &a, const T &b ) {`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:20` 至 `:22`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:20` 至 `:21`。

```python
template<typename T>
__device__ __forceinline__ bool less( const T &a, const T &b ) {
```

逐行对应说明：

- 第 1 行：声明下一函数以数值类型 `T` 为模板参数。
- 第 2 行：定义仅在 device 调用、强制内联的严格小于函数；参数为 const 引用。

#### 返回：`return ( a < b );`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:23` 至 `:24`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:22` 至 `:23`。

```python
    return ( a < b );
}
```

逐行对应说明：

- 第 1 行：返回 `a<b` 的布尔结果。
- 第 2 行：结束 `less`。

#### 模板定义：`__device__ __forceinline__ bool greater( const T &a, const T &b ) {`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:26` 至 `:28`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:25` 至 `:26`。

```python
template<typename T>
__device__ __forceinline__ bool greater( const T &a, const T &b ) {
```

逐行对应说明：

- 第 1 行：为严格大于函数声明模板。
- 第 2 行：定义 device 严格大于函数。

#### 返回：`return ( a > b );`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:29` 至 `:30`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:27` 至 `:28`。

```python
    return ( a > b );
}
```

逐行对应说明：

- 第 1 行：返回 `a>b`。
- 第 2 行：结束 `greater`。

#### 模板定义：`__device__ __forceinline__ bool less_equal( const T &a, const T &b ) {`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:32` 至 `:34`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:30` 至 `:31`。

```python
template<typename T>
__device__ __forceinline__ bool less_equal( const T &a, const T &b ) {
```

逐行对应说明：

- 第 1 行：为小于等于函数声明模板。
- 第 2 行：定义 device 小于等于函数。

#### 返回：`return ( a <= b );`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:35` 至 `:36`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:32` 至 `:33`。

```python
    return ( a <= b );
}
```

逐行对应说明：

- 第 1 行：返回 `a<=b`。
- 第 2 行：结束函数。

#### 模板定义：`__device__ __forceinline__ bool greater_equal( const T &a, const T &b ) …`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:38` 至 `:40`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:35` 至 `:36`。

```python
template<typename T>
__device__ __forceinline__ bool greater_equal( const T &a, const T &b ) {
```

逐行对应说明：

- 第 1 行：为大于等于函数声明模板。
- 第 2 行：定义 device 大于等于函数。

#### 返回：`return ( a >= b );`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:41` 至 `:42`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:37` 至 `:38`。

```python
    return ( a >= b );
}
```

逐行对应说明：

- 第 1 行：返回 `a>=b`。
- 第 2 行：结束函数。

#### 模板定义：`__device__ __forceinline__ bool equal( const T &a, const T &b ) {`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:44` 至 `:46`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:40` 至 `:41`。

```python
template<typename T>
__device__ __forceinline__ bool equal( const T &a, const T &b ) {
```

逐行对应说明：

- 第 1 行：为相等函数声明模板。
- 第 2 行：定义 device 相等函数。

#### 返回：`return ( a == b );`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:47` 至 `:48`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:42` 至 `:43`。

```python
    return ( a == b );
}
```

逐行对应说明：

- 第 1 行：返回 `a==b`。
- 第 2 行：结束函数。

#### 模板定义：`__device__ __forceinline__ bool not_equal( const T &a, const T &b ) {`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:50` 至 `:52`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:45` 至 `:46`。

```python
template<typename T>
__device__ __forceinline__ bool not_equal( const T &a, const T &b ) {
```

逐行对应说明：

- 第 1 行：为不等函数声明模板。
- 第 2 行：定义 device 不等函数。

#### 返回：`return ( a != b );`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:53` 至 `:54`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:47` 至 `:48`。

```python
    return ( a != b );
}
```

逐行对应说明：

- 第 1 行：返回 `a!=b`。
- 第 2 行：结束函数。

#### 模板定义：`using op_func = bool ( * )( const T &, const T & );`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:56` 至 `:58`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:50` 至 `:51`。

```python
template<typename T>
using op_func = bool ( * )( const T &, const T & );
```

逐行对应说明：

- 第 1 行：为函数指针别名声明类型模板。
- 第 2 行：`op_func<T>` 表示接收两个 `const T&`、返回 `bool` 的函数指针。

#### 计算或更新：`__device__ op_func<int> const func_i[6]`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:61` 至 `:64`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:53` 至 `:56`。

```python
__device__ op_func<int> const func_i[6]      = { less, greater, less_equal, greater_equal, equal, not_equal };
__device__ op_func<long int> const func_l[6] = { less, greater, less_equal, greater_equal, equal, not_equal };
__device__ op_func<float> const func_f[6]    = { less, greater, less_equal, greater_equal, equal, not_equal };
__device__ op_func<double> const func_d[6]   = { less, greater, less_equal, greater_equal, equal, not_equal };
```

逐行对应说明：

- 第 1 行：device 常量函数指针表，供 int32 入口按操作码选择比较器。
- 第 2 行：`long int`/int64 对应表，顺序同 Python 字典。
- 第 3 行：float32 对应表。
- 第 4 行：float64 对应表。

#### 源码区域：HELPER FUNCTIONS

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:66` 至 `:68`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:58` 至 `:60`。

```python
///////////////////////////////////////////////////////////////////////////////
//                              HELPER FUNCTIONS                             //
///////////////////////////////////////////////////////////////////////////////
```

逐行对应说明：

- 第 1 行：helper 区域分隔线。
- 第 2 行：标记边界索引 helper。
- 第 3 行：结束标题。

#### 定义正向边界索引修正 helper

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:71` 至 `:71`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:62` 至 `:62`。

```python
__device__ __forceinline__ void clip_plus( const bool &clip, const int &n, int &plus ) {
```

逐行对应说明：

- 第 1 行：定义正向索引修正函数；`plus` 为非常量引用，函数原位修改它。

#### 条件分支：`if ( clip ) {`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:72` 至 `:72`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:63` 至 `:63`。

```python
    if ( clip ) {
```

逐行对应说明：

- 第 1 行：选择端点裁剪语义。

#### 条件分支：`if ( plus >= n ) {`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:73` 至 `:76`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:64` 至 `:67`。

```python
        if ( plus >= n ) {
            plus = n - 1;
        }
    } else {
```

逐行对应说明：

- 第 1 行：检测正向索引是否越过末端。
- 第 2 行：越界时固定到最后一个合法索引。
- 第 3 行：结束越界判断。
- 第 4 行：`clip=false` 时选择回绕。

#### 条件分支：`if ( plus >= n ) {`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:77` 至 `:81`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:68` 至 `:72`。

```python
        if ( plus >= n ) {
            plus -= n;
        }
    }
}
```

逐行对应说明：

- 第 1 行：检查正向越界。
- 第 2 行：只减一次轴长；当偏移跨多周期时可能仍越界。
- 第 3 行：结束判断。
- 第 4 行：结束边界模式分支。
- 第 5 行：结束 `clip_plus`。

#### 定义负向边界索引修正 helper

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:84` 至 `:84`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:74` 至 `:74`。

```python
__device__ __forceinline__ void clip_minus( const bool &clip, const int &n, int &minus ) {
```

逐行对应说明：

- 第 1 行：定义负向索引修正函数。

#### 条件分支：`if ( clip ) {`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:85` 至 `:85`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:75` 至 `:75`。

```python
    if ( clip ) {
```

逐行对应说明：

- 第 1 行：选择裁剪。

#### 条件分支：`if ( minus < 0 ) {`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:86` 至 `:89`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:76` 至 `:79`。

```python
        if ( minus < 0 ) {
            minus = 0;
        }
    } else {
```

逐行对应说明：

- 第 1 行：检测负向越过首端。
- 第 2 行：固定到首索引。
- 第 3 行：结束越界判断。
- 第 4 行：选择回绕。

#### 条件分支：`if ( minus < 0 ) {`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:90` 至 `:94`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:80` 至 `:84`。

```python
        if ( minus < 0 ) {
            minus += n;
        }
    }
}
```

逐行对应说明：

- 第 1 行：检查负索引。
- 第 2 行：只加一次轴长。
- 第 3 行：结束判断。
- 第 4 行：结束模式分支。
- 第 5 行：结束 `clip_minus`。

#### 源码区域：BOOLRELEXTREMA 1D

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:96` 至 `:98`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:86` 至 `:88`。

```python
///////////////////////////////////////////////////////////////////////////////
//                          BOOLRELEXTREMA 1D                                //
///////////////////////////////////////////////////////////////////////////////
```

逐行对应说明：

- 第 1 行：一维区域分隔线。
- 第 2 行：标记一维核心。
- 第 3 行：结束标题。

#### 模板定义：`__device__ void _cupy_boolrelextrema_1D( const int  n,`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:100` 至 `:107`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:90` 至 `:96`。

```python
template<typename T, class U>
__device__ void _cupy_boolrelextrema_1D( const int  n,
                                         const int  order,
                                         const bool clip,
                                         const T *__restrict__ inp,
                                         bool *__restrict__ results,
                                         U func ) {
```

逐行对应说明：

- 第 1 行：一维核心有数据类型 `T` 和比较器类型 `U` 两个模板参数。
- 第 2 行：开始 device 核心签名，`n` 是轴长。
- 第 3 行：每侧邻居数。
- 第 4 行：边界模式标志。
- 第 5 行：只读输入设备指针；`__restrict__` 告诉编译器其别名受限。
- 第 6 行：可写布尔输出指针。
- 第 7 行：比较函数参数并结束签名。

#### 一维线程全局索引与 grid stride

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:110` 至 `:112`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:98` 至 `:99`。

```python
    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };
```

逐行对应说明：

- 第 1 行：计算线程一维全局索引 `blockIdx.x*blockDim.x+threadIdx.x`。
- 第 2 行：计算 grid 总线程数作为跨步。

#### 循环：`for ( int tid = tx; tid < n; tid += stride ) {`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:115` 至 `:115`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:101` 至 `:101`。

```python
    for ( int tid = tx; tid < n; tid += stride ) {
```

逐行对应说明：

- 第 1 行：grid-stride loop；一个线程可处理多个中心。

#### 加载一维中心并初始化全称比较累积器

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:118` 至 `:119`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:103` 至 `:104`。

```python
        const T data { inp[tid] };
        bool    temp { true };
```

逐行对应说明：

- 第 1 行：把中心样本读入局部常量。
- 第 2 行：全称逻辑与累积器初始化为真。

#### 循环：`for ( int o = 1; o < ( order + 1 ); o++ ) {`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:122` 至 `:124`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:106` 至 `:108`。

```python
        for ( int o = 1; o < ( order + 1 ); o++ ) {
            int plus { tid + o };
            int minus { tid - o };
```

逐行对应说明：

- 第 1 行：偏移从 1 遍历到 `order`。
- 第 2 行：计算正向邻居索引。
- 第 3 行：计算负向邻居索引。

#### 修正一维正负邻居边界

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:126` 至 `:127`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:110` 至 `:111`。

```python
            clip_plus( clip, n, plus );
            clip_minus( clip, n, minus );
```

逐行对应说明：

- 第 1 行：原位修正正向边界。
- 第 2 行：原位修正负向边界。

#### 累积一维两侧比较并写回 mask

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:130` 至 `:137`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:113` 至 `:118`。

```python
            temp &= func( data, inp[plus] );
            temp &= func( data, inp[minus] );
        }
        results[tid] = temp;
    }
}
```

逐行对应说明：

- 第 1 行：中心与正向邻居比较，并逻辑与到累积器。
- 第 2 行：累积负向比较。注意 `&=` 不提供短路，所有偏移都会执行。
- 第 3 行：结束偏移循环。
- 第 4 行：写回当前中心最终布尔值。
- 第 5 行：结束 grid-stride loop。
- 第 6 行：结束一维 device 核心。

#### 导出 CUDA 入口：`extern "C" __global__ void __launch_bounds__`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:140` 至 `:147`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:120` 至 `:127`。

```python
extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_1D_int32( const int  n,
                                                                                   const int  order,
                                                                                   const bool clip,
                                                                                   const int  comp,
                                                                                   const int *__restrict__ inp,
                                                                                   bool *__restrict__ results ) {
    _cupy_boolrelextrema_1D<int, op_func<int>>( n, order, clip, inp, results, func_i[comp] );
}
```

逐行对应说明：

- 第 1 行：定义可从 host 启动的 int32 kernel；C linkage 避免符号名改编，launch bounds 上限 512 线程。
- 第 2 行：int32 入口的阶数参数。
- 第 3 行：边界标志。
- 第 4 行：比较器操作码。
- 第 5 行：int32 输入指针。
- 第 6 行：输出指针并开始函数体。
- 第 7 行：实例化 int 核心，并用 `func_i[comp]` 选比较函数。
- 第 8 行：结束 int32 入口。

#### 导出 CUDA 入口：`extern "C" __global__ void __launch_bounds__`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:149` 至 `:156`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:129` 至 `:136`。

```python
extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_1D_int64( const int  n,
                                                                                   const int  order,
                                                                                   const bool clip,
                                                                                   const int  comp,
                                                                                   const long int *__restrict__ inp,
                                                                                   bool *__restrict__ results ) {
    _cupy_boolrelextrema_1D<long int, op_func<long int>>( n, order, clip, inp, results, func_l[comp] );
}
```

逐行对应说明：

- 第 1 行：定义 int64 一维入口。
- 第 2 行：阶数。
- 第 3 行：边界标志。
- 第 4 行：操作码。
- 第 5 行：int64 输入指针。
- 第 6 行：输出指针并开始函数体。
- 第 7 行：调用 long int 核心和 `func_l` 表。
- 第 8 行：结束 int64 入口。

#### 导出 CUDA 入口：`extern "C" __global__ void __launch_bounds__`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:158` 至 `:165`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:138` 至 `:145`。

```python
extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_1D_float32( const int  n,
                                                                                     const int  order,
                                                                                     const bool clip,
                                                                                     const int  comp,
                                                                                     const float *__restrict__ inp,
                                                                                     bool *__restrict__ results ) {
    _cupy_boolrelextrema_1D<float, op_func<float>>( n, order, clip, inp, results, func_f[comp] );
}
```

逐行对应说明：

- 第 1 行：定义 float32 一维入口。
- 第 2 行：阶数。
- 第 3 行：边界标志。
- 第 4 行：操作码。
- 第 5 行：float32 输入。
- 第 6 行：输出并开始函数体。
- 第 7 行：调用 float 核心和 `func_f`。
- 第 8 行：结束 float32 入口。

#### 导出 CUDA 入口：`extern "C" __global__ void __launch_bounds__`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:167` 至 `:174`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:147` 至 `:154`。

```python
extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_1D_float64( const int  n,
                                                                                     const int  order,
                                                                                     const bool clip,
                                                                                     const int  comp,
                                                                                     const double *__restrict__ inp,
                                                                                     bool *__restrict__ results ) {
    _cupy_boolrelextrema_1D<double, op_func<double>>( n, order, clip, inp, results, func_d[comp] );
}
```

逐行对应说明：

- 第 1 行：定义 float64 一维入口。
- 第 2 行：阶数。
- 第 3 行：边界标志。
- 第 4 行：操作码。
- 第 5 行：float64 输入。
- 第 6 行：输出并开始函数体。
- 第 7 行：调用 double 核心和 `func_d`。
- 第 8 行：结束 float64 入口。


### 4.6 二维 CUDA kernel 与类型入口逐行解释

学习副本二维核心起点 `:182`，四个入口为 `:239/:250/:261/:272`；只读基准对应 `:161` 与 `:211/:222/:233/:244`。

#### 源码区域：BOOLRELEXTREMA 2D

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:176` 至 `:178`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:156` 至 `:158`。

```python
///////////////////////////////////////////////////////////////////////////////
//                          BOOLRELEXTREMA 2D                                //
///////////////////////////////////////////////////////////////////////////////
```

逐行对应说明：

- 第 1 行：二维区域分隔线。
- 第 2 行：标记二维核心。
- 第 3 行：结束标题。

#### 模板定义：`__device__ void _cupy_boolrelextrema_2D( const int  in_x,`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:180` 至 `:189`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:160` 至 `:168`。

```python
template<typename T, class U>
__device__ void _cupy_boolrelextrema_2D( const int  in_x,
                                         const int  in_y,
                                         const int  order,
                                         const bool clip,
                                         const int  axis,
                                         const T *__restrict__ inp,
                                         bool *__restrict__ results,
                                         U func ) {
```

逐行对应说明：

- 第 1 行：二维核心以数据类型和比较器类型参数化。
- 第 2 行：开始二维核心；`in_x` 实际是列数。
- 第 3 行：`in_y` 实际是行数。
- 第 4 行：邻域阶数。
- 第 5 行：边界标志。
- 第 6 行：检测轴。
- 第 7 行：二维输入的连续设备指针。
- 第 8 行：同 shape 布尔输出指针。
- 第 9 行：比较函数并结束签名。

#### 计算二维线程的行列坐标

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:192` 至 `:193`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:170` 至 `:171`。

```python
    const int ty { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int tx { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };
```

逐行对应说明：

- 第 1 行：由 CUDA x 维计算列坐标，变量名 `ty` 与直觉相反。
- 第 2 行：由 CUDA y 维计算行坐标。

#### 条件分支：`if ( ( tx < in_y ) && ( ty < in_x ) ) {`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:196` 至 `:198`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:173` 至 `:174`。

```python
    if ( ( tx < in_y ) && ( ty < in_x ) ) {
        int tid { tx * in_x + ty };
```

逐行对应说明：

- 第 1 行：排除因 grid 向上取整产生的越界线程。
- 第 2 行：按 C 行优先布局把 `(row=tx,column=ty)` 转成线性索引。

#### 加载二维中心并初始化全称比较累积器

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:200` 至 `:201`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:176` 至 `:177`。

```python
        const T data { inp[tid] };
        bool    temp { true };
```

逐行对应说明：

- 第 1 行：读取中心样本。
- 第 2 行：初始化全称比较累积器。

#### 循环：`for ( int o = 1; o < ( order + 1 ); o++ ) {`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:203` 至 `:203`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:179` 至 `:179`。

```python
        for ( int o = 1; o < ( order + 1 ); o++ ) {
```

逐行对应说明：

- 第 1 行：遍历每侧偏移。

#### 初始化二维正负邻居索引

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:205` 至 `:206`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:181` 至 `:182`。

```python
            int plus {};
            int minus {};
```

逐行对应说明：

- 第 1 行：值初始化正向索引为 0，稍后覆盖。
- 第 2 行：值初始化负向索引。

#### 条件分支：`if ( axis == 0 ) {`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:209` 至 `:211`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:184` 至 `:186`。

```python
            if ( axis == 0 ) {
                plus  = tx + o;
                minus = tx - o;
```

逐行对应说明：

- 第 1 行：轴 0 分支，沿行坐标移动。
- 第 2 行：正向行坐标。
- 第 3 行：负向行坐标。

#### axis 0 的行边界修正

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:213` 至 `:214`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:188` 至 `:189`。

```python
                clip_plus( clip, in_y, plus );
                clip_minus( clip, in_y, minus );
```

逐行对应说明：

- 第 1 行：以行数修正正向行坐标。
- 第 2 行：修正负向行坐标。

#### 计算或更新：`plus`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:216` 至 `:221`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:191` 至 `:195`。

```python
                plus  = plus * in_x + ty;
                minus = minus * in_x + ty;
            } else {
                plus  = ty + o;
                minus = ty - o;
```

逐行对应说明：

- 第 1 行：把正向 `(row,column)` 转成线性索引。
- 第 2 行：转换负向索引。
- 第 3 行：非轴 0 都按轴 1 处理，源码没有单独验证 `axis==1`。
- 第 4 行：正向列坐标。
- 第 5 行：负向列坐标。

#### axis 1 的列边界修正

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:223` 至 `:224`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:197` 至 `:198`。

```python
                clip_plus( clip, in_x, plus );
                clip_minus( clip, in_x, minus );
```

逐行对应说明：

- 第 1 行：以列数修正正向列。
- 第 2 行：修正负向列。

#### 计算或更新：`plus`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:226` 至 `:228`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:200` 至 `:202`。

```python
                plus  = tx * in_x + plus;
                minus = tx * in_x + minus;
            }
```

逐行对应说明：

- 第 1 行：正向列转换为线性地址。
- 第 2 行：负向列转换为线性地址。
- 第 3 行：结束 axis 分支。

#### 累积二维两侧比较并写回 mask

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:231` 至 `:236`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:204` 至 `:209`。

```python
            temp &= func( data, inp[plus] );
            temp &= func( data, inp[minus] );
        }
        results[tid] = temp;
    }
}
```

逐行对应说明：

- 第 1 行：累积正向邻居比较。
- 第 2 行：累积负向邻居比较。
- 第 3 行：结束偏移循环。
- 第 4 行：写回二维中心结果。
- 第 5 行：结束有效线程判断。
- 第 6 行：结束二维核心。

#### 导出 CUDA 入口：`extern "C" __global__ void __launch_bounds__`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:239` 至 `:248`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:211` 至 `:220`。

```python
extern "C" __global__ void __launch_bounds__( 256 ) _cupy_boolrelextrema_2D_int32( const int  in_x,
                                                                                   const int  in_y,
                                                                                   const int  order,
                                                                                   const bool clip,
                                                                                   const int  comp,
                                                                                   const int  axis,
                                                                                   const int *__restrict__ inp,
                                                                                   bool *__restrict__ results ) {
    _cupy_boolrelextrema_2D<int, op_func<int>>( in_x, in_y, order, clip, axis, inp, results, func_i[comp] );
}
```

逐行对应说明：

- 第 1 行：定义 int32 二维 host 可启动入口；launch bounds 为 256。
- 第 2 行：行数。
- 第 3 行：阶数。
- 第 4 行：边界标志。
- 第 5 行：比较器操作码。
- 第 6 行：检测轴。
- 第 7 行：int32 输入。
- 第 8 行：输出并开始函数体。
- 第 9 行：用 `func_i[comp]` 调用 int 二维核心。
- 第 10 行：结束 int32 入口。

#### 导出 CUDA 入口：`extern "C" __global__ void __launch_bounds__`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:250` 至 `:259`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:222` 至 `:231`。

```python
extern "C" __global__ void __launch_bounds__( 256 ) _cupy_boolrelextrema_2D_int64( const int  in_x,
                                                                                   const int  in_y,
                                                                                   const int  order,
                                                                                   const bool clip,
                                                                                   const int  comp,
                                                                                   const int  axis,
                                                                                   const long int *__restrict__ inp,
                                                                                   bool *__restrict__ results ) {
    _cupy_boolrelextrema_2D<long int, op_func<long int>>( in_x, in_y, order, clip, axis, inp, results, func_l[comp] );
}
```

逐行对应说明：

- 第 1 行：定义 int64 二维入口。
- 第 2 行：行数。
- 第 3 行：阶数。
- 第 4 行：边界标志。
- 第 5 行：操作码。
- 第 6 行：检测轴。
- 第 7 行：int64 输入。
- 第 8 行：输出并开始函数体。
- 第 9 行：用 `func_l` 调用 long int 核心。
- 第 10 行：结束 int64 入口。

#### 导出 CUDA 入口：`extern "C" __global__ void __launch_bounds__`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:261` 至 `:270`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:233` 至 `:242`。

```python
extern "C" __global__ void __launch_bounds__( 256 ) _cupy_boolrelextrema_2D_float32( const int  in_x,
                                                                                     const int  in_y,
                                                                                     const int  order,
                                                                                     const bool clip,
                                                                                     const int  comp,
                                                                                     const int  axis,
                                                                                     const float *__restrict__ inp,
                                                                                     bool *__restrict__ results ) {
    _cupy_boolrelextrema_2D<float, op_func<float>>( in_x, in_y, order, clip, axis, inp, results, func_f[comp] );
}
```

逐行对应说明：

- 第 1 行：定义 float32 二维入口。
- 第 2 行：行数。
- 第 3 行：阶数。
- 第 4 行：边界标志。
- 第 5 行：操作码。
- 第 6 行：检测轴。
- 第 7 行：float 输入。
- 第 8 行：输出并开始函数体。
- 第 9 行：用 `func_f` 调用 float 核心。
- 第 10 行：结束 float32 入口。

#### 导出 CUDA 入口：`extern "C" __global__ void __launch_bounds__`

定位：学习副本 `Learning/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:272` 至 `:281`；只读基准 `ZKX/cusignal-23.08.00/cpp/src/peak_finding/_peak_finding.cu:244` 至 `:253`。

```python
extern "C" __global__ void __launch_bounds__( 256 ) _cupy_boolrelextrema_2D_float64( const int  in_x,
                                                                                     const int  in_y,
                                                                                     const int  order,
                                                                                     const bool clip,
                                                                                     const int  comp,
                                                                                     const int  axis,
                                                                                     const double *__restrict__ inp,
                                                                                     bool *__restrict__ results ) {
    _cupy_boolrelextrema_2D<double, op_func<double>>( in_x, in_y, order, clip, axis, inp, results, func_d[comp] );
}
```

逐行对应说明：

- 第 1 行：定义 float64 二维入口。
- 第 2 行：行数。
- 第 3 行：阶数。
- 第 4 行：边界标志。
- 第 5 行：操作码。
- 第 6 行：检测轴。
- 第 7 行：double 输入。
- 第 8 行：输出并开始函数体。
- 第 9 行：用 `func_d` 调用 double 核心。
- 第 10 行：结束 float64 入口和文件。


## 5. 调用链与算法总结

### 5.1 总调用链

```text
cusignal.argrelextrema
  → peak_finding.argrelextrema
      → cp.asarray(data)
      → _boolrelextrema
          ├─ data.ndim < 3
          │   → _peak_finding
          │       → 比较器操作码 + dtype/kernel 类型选择
          │       → fatbin cache / Python wrapper
          │       → 1D 或 2D CUDA kernel
          │           → 每个中心循环 o=1..order
          │           → 边界修正
          │           → 正负邻居比较全部逻辑与
          └─ data.ndim >= 3
              → CuPy 索引数组循环 shift=1..order
              → cp.take 取得正负邻居
              → comparator 数组比较全部逻辑与
      → cp.nonzero(results)
      → 坐标数组 tuple
```

### 5.2 一维执行例

对 `x=[2,1,2,3,2,0,1,0]`、`comparator=cp.greater`、`order=1`：

1. `cp.asarray` 得到一维 CuPy 数组。
2. `_boolrelextrema` 因 `ndim=1` 分配 8 个未初始化布尔值并调用 `_peak_finding`。
3. `cp.greater` 查表得到操作码 1，dtype 决定具体 fatbin 符号。
4. CUDA 每个中心比较 `x[n]>x[n+1]` 与 `x[n]>x[n-1]`。
5. 结果掩码为 `[False,False,False,True,False,False,True,False]`。
6. `cp.nonzero` 返回 `(array([3,6]),)`。

### 5.3 三维及以上执行差别

数学判据相同，但不会进入 fatbin：中心数组和每个偏移的邻居数组由 `cp.take` 生成，调用者的 `comparator` 直接作用于整个数组。每轮产生/使用两组邻居索引和两组邻居值，再原位更新布尔掩码。

## 6. 数学公式与源码映射

令边界映射为 $b$，中心索引为 $n$，每侧阶数为 $r$，比较器为 $C$：

$$
M[n]=\bigwedge_{k=1}^{r}\left(C(x[n],x[b(n+k)])\land C(x[n],x[b(n-k)])\right).
$$

| 数学对象 | Python 高维路径 | CUDA 一/二维路径 |
| --- | --- | --- |
| 中心 $x[n]$ | `main = cp.take(...)`，基准 `peak_finding.py:61` | `data = inp[tid]`，基准 `_peak_finding.cu:103/176` |
| 偏移 $k=1..r$ | `for shift...`，`:62` | `for (int o=1;...)`，`:106/179` |
| 正邻居 $b(n+k)$ | `p_locs`、`plus`，`:64/67/69` | `plus` 与 `clip_plus`，`:107/110` 或 `:185/188` |
| 负邻居 $b(n-k)$ | `m_locs`、`minus`，`:65/68/70` | `minus` 与 `clip_minus`，`:108/111` 或 `:186/189` |
| 比较器 $C$ | 直接调用 `comparator`，`:71-72` | `func_i/l/f/d[comp]`，CUDA `:53-56` 与 `:113-114/:204-205` |
| 全称逻辑与 $\bigwedge$ | `results &= ...` | `temp &= ...` |
| 命中集合坐标 | `cp.nonzero(results)`，`:233` | kernel 只生成掩码，不生成坐标 |

对 `cp.greater` 且 `order=1`：

$$
M[n]=(x[n]>x[b(n+1)])\land(x[n]>x[b(n-1)]),
$$

等价于相邻差分从正变负。代码没有显式求差分，因此不引入减法舍入误差；判断完全由输入 dtype 的比较规则决定。

## 7. 边界、异常、数值稳定性与实现限制

### 7.1 参数和异常

- `order`：`int(order) != order` 或 `<1` 时抛 `ValueError`。`True` 因 Python 中等于整数 1，可能通过检查；这是 Python 类型语义。
- `axis`：公开函数没有主动范围检查。高维路径由 `shape[axis]`/`cp.take` 报错；二维 kernel 对任何非 0 的轴都按轴 1 执行，属于需要调用者避免的风险。
- `comparator`：一、二维必须是 `_modedict` 的精确 key，否则在 kernel 启动前 `KeyError`；高维只要求能返回兼容 shape 的布尔数组。
- `mode`：`clip` 被支持；非 `clip` 在 helper/kernel 中走回绕；公开函数只对精确 `raise` 最终抛 `NotImplementedError`。未知字符串不会被拒绝，而会被当成回绕，属于接口校验缺口。
- dtype：一、二维不在四种白名单中时抛 `ValueError`。

### 7.2 边界细节

`clip` 的索引为

$$
b_{\mathrm{clip}}(j)=\min(\max(j,0),N-1).
$$

一、二维 CUDA 回绕 helper 只执行一次 `j-=N` 或 `j+=N`。因此当 `order` 使某偏移跨越多于一个轴长时，不能视为完整的模运算实现，可能访问非法地址。高维 CuPy 整数数组索引则按 CuPy 自身回绕语义工作。

### 7.3 平台、NaN 与 dtype

- `greater/less` 是严格比较，相等平台不会通过全部比较。
- `greater_equal/less_equal` 可能返回平台内多个样本，不会把平台压缩成一个代表索引。
- 对浮点 NaN，六种有序比较通常为假，`not_equal` 通常为真；实际语义完全由 IEEE 浮点比较和所选 comparator 决定。
- 整数比较无舍入；浮点比较本身不进行算术，所以没有迭代累计误差，但极接近数值仍受输入表示精度影响。

## 8. 复杂度、内存与性能瓶颈

设输入总元素数为 $P$，邻域阶数为 $r$。

- 时间复杂度：两条路径都需至多 $2Pr$ 次比较，渐进复杂度为 $O(Pr)$。
- 输出掩码：$O(P)$ 布尔存储；最终坐标最坏为 $O(dP)$，其中 $d=\text{data.ndim}$。
- 一/二维 CUDA 路径：每个元素只保留中心和一个布尔累积器，除输出外没有完整邻居数组；主要成本是非合并边界访问、函数指针调用和随 `order` 增长的串行线程内循环。
- 高维路径：除 `results` 外，还会反复建立 `p_locs/m_locs`、`plus/minus` 和比较结果；峰值临时内存为 $O(P+N_a)$，但每个 shift 都有多个 kernel launch，启动开销约随 $r$ 线性增加。
- `results.any()` 可提前结束全假情形，但 Python 对设备标量做真值判断可能引入每轮设备同步；候选长期存在时，这个检查可能抵消部分收益。
- `cp.nonzero` 需要扫描完整掩码，并为坐标分配空间。

文档源码第 1.3 节中“输出 tuple 的长度”等于维数；因此高维且命中很多位置时，坐标输出可能比单个布尔掩码占用更多内存。

## 9. 与 SciPy 语义的对应和有意差异

- 数学判据、`axis`、`order`、`clip/wrap` 和 tuple 坐标输出意图与 SciPy `argrelextrema` 对齐。
- cuSignal 把计算搬到 CuPy 和自有 CUDA fatbin。
- 一、二维比较器与 dtype 因预编译分发受到额外限制。
- `mode='raise'` 不支持。
- 源码 docstring 的“Indices of the maxima”表述过窄；函数也能检测最小值或任意比较器定义的局部模式。

## 10. 建议阅读顺序与检查问题

1. 先读公开函数基准 `peak_finding.py:184-233`：观察它只做转换、helper 调用、模式拒绝和坐标化。
2. 再读 `_boolrelextrema` 基准 `:21-77`：画出 `ndim<3` 的分支图。
3. 手算高维路径 `:58-75` 中一次 `shift` 后 `results` 如何变化。
4. 读 `_peak_finding_cuda.py:124-166`：追踪 `cp.greater → 1 → func_*[1] → greater`。
5. 读 CUDA 一维核心 `:90-118`：确认每个线程负责哪些 `tid`。
6. 读二维核心 `:160-209`：分别代入 `axis=0` 与 `axis=1`，检查线性地址。
7. 最后读八个导出 kernel，认识它们只是类型实例化包装，不是八套不同算法。

### 10.1 自检问题

1. 为什么一/二维 `cp.empty` 是安全的，而高维初始化必须使用 `cp.ones`？
2. 为什么 `results &= comparator(...)` 能表达数学全称量词？
3. `cp.greater` 在 Python、操作码和 CUDA 函数表三层如何对应？
4. 二维 `axis=0` 时，哪些坐标改变，哪些坐标不变？
5. 为什么 `cp.nonzero` 必须放在公开函数而不是 CUDA 核心中？
6. 自定义 lambda 为什么可在三维路径工作，却不能在一维 kernel 路径工作？
7. `mode="unknown"` 实际会发生什么？这与理想参数校验有何差别？
8. `order>N` 时一维 CUDA 回绕 helper 为什么存在风险？
9. `results.any()` 的提前退出为什么可能带来同步成本？
10. 平台信号应使用严格比较、非严格比较还是区域极值算法？三者输出有何不同？

### 10.2 逐题参考答案

1. 一维和二维专用 kernel 会为每个有效输出位置写入布尔结果（基准 `_peak_finding.cu:101-116,173-207`），所以 `peak_finding.py:55` 的 `cp.empty` 安全；高维在 `:60-72` 多轮执行 `results &= ...`，初值必须全真，因此使用 `cp.ones`。
2. 基准 `peak_finding.py:62-72` 的每次 `&=` 都把此前条件与当前正/负邻居条件逐元素合取；遍历 `1..order` 后为真当且仅当全部邻居比较成立，正好实现全称量词。
3. `_peak_finding_cuda.py:20-27` 把 `cp.greater` 编码为 1；CUDA `_peak_finding.cu:53-56` 的函数表在索引 1 放置 `greater`，入口再以 `func_*[comp]` 传入核心。
4. `_peak_finding.cu:184-202` 显示 `axis=0` 改变行坐标并保持列坐标；线性地址按 `row*in_x+column` 换算。变量名 `tx/ty` 与直觉可能相反，应以地址公式为准。
5. `_peak_finding.cu:91-116,161-207` 只生成同 shape 掩码；公开函数 `peak_finding.py:233` 的 `cp.nonzero` 再把动态真值位置压缩成坐标 tuple。
6. 三维及以上在 `peak_finding.py:58-72` 直接调用 comparator 数组函数；一二维在 `_peak_finding_cuda.py:20-27,126` 只能查六个预编译 key，因此任意 Python lambda 会在操作码查表处失败。
7. 当前逻辑不会拒绝未知字符串：只有精确 `clip` 进入裁剪，精确 `raise` 在公开函数最终抛错，其余字符串都按回绕处理。一、二维在 `_peak_finding_cuda.py:128-131` 把它们转成 `clip=False`；三维以上在 `peak_finding.py:66-70` 保留越界索引并依赖 CuPy 回绕。这忠实反映固定版实现，但比只接受 `clip/wrap` 的严格接口更宽松。
8. 一、二维 CUDA helper `_peak_finding.cu:62-84` 对正负越界只减或加一次轴长；当 `order>N` 使偏移跨越多个周期时，索引修正后仍可能越界。三维以上 CuPy 整数数组索引使用模回绕，不存在同一缺陷。
9. 基准 `peak_finding.py:74-75` 用 `results.any()` 驱动 Python `if`；设备归约标量必须反馈给 host 控制流，通常引入同步。它可能减少后续比较，也可能增加每轮同步开销。
10. `_peak_finding.cu:20-47` 的严格比较遇到相等邻居即失败，非严格比较可能让平台多个点通过；该源码没有连通区域或平台中心选择逻辑，因此区域极值需要不同算法。

## 11. 本阶段验证说明

- 已逐个比较五个相关学习副本和 `ZKX/cusignal-23.08.00` 只读基准。
- 加注释前，未注释文件 hash 一致；顶层 `__init__.py` 去除已有合法学习注释后逐行一致。
- 本阶段只插入独立的 `<学习注释：...>` 行，没有改写、移动、删除或格式化任何原始行。
- 按学习规则未在本地或 ZQ500 上构建、运行 Python、CUDA 或硬件测试；文中执行结果仅解释原始 docstring 示例，没有声称重新运行。
