# ca_cfar Python 源码算法

---

## 1. 代码定位与接口概览

### 公开导入路径

```python
from cusignal.radartools import ca_cfar, cfar_alpha
```

导出链：
- 定义文件：`cusignal/radartools/radartools.py`
- 模块导出：`cusignal/radartools/__init__.py:15-21`（将 `ca_cfar` 和 `cfar_alpha` 从 `radartools` 子模块导入到 `cusignal.radartools` 包）

### 符号名与行号

| 符号 | 类型 | 只读基准行号 | 学习副本行号 |
|------|------|-------------|-------------|
| `cfar_alpha` | 函数定义 | `radartools.py:243-261` | `radartools.py:266-286` |
| `ca_cfar` | 函数定义 | `radartools.py:264-366` | `radartools.py:289-391` |
| `_ca_cfar_2d_kernel` | `cp.RawKernel` 定义 | `radartools.py:369-436` | `radartools.py:394-461` |
| `_ca_cfar_1d_kernel` | `cp.RawKernel` 定义 | `radartools.py:439-473` | `radartools.py:464-499` |

文件相对路径（从 `ZKX_dev/` 起算）：`cusignal-23.08.00/python/cusignal/radartools/radartools.py`

---

## 2. 当前算子的完整相关源码

### 2.1 cfar_alpha 函数（ca_cfar 的直接依赖）

只读基准原文（行 243–261）：

```python
def cfar_alpha(pfa, N):
    """
    Computes the value of alpha corresponding to a given probability
    of false alarm and number of reference cells N.

    Parameters
    ----------
    pfa : float
        Probability of false alarm.

    N : int
        Number of reference cells.

    Returns
    -------
    alpha : float
        Alpha value.
    """
    return N * (pfa ** (-1.0 / N) - 1)
```

### 2.2 ca_cfar 函数

只读基准原文（行 264–366）：

```python
def ca_cfar(array, guard_cells, reference_cells, pfa=1e-3):
    """
    Computes the cell-averaged constant false alarm rate (CA CFAR) detector
    threshold and returns for a given array.

    Parameters
    ----------
    array : ndarray
        Array containing data to be processed.

    guard_cells_x : int
        One-sided guard cell count in the first dimension.

    guard_cells_y : int
        One-sided guard cell count in the second dimension.

    reference_cells_x : int
        one-sided reference cell count in the first dimension.

    reference_cells_y : int
        one-sided referenc cell count in the second dimension.

    pfa : float
        Probability of false alarm.

    Returns
    -------
    threshold : ndarray
        CFAR threshold
    return : ndarray
        CFAR detections
    """
    shape = array.shape
    if len(shape) > 2:
        raise TypeError("Only 1D and 2D arrays are currently supported.")
    mask = cp.zeros(shape, dtype=cp.float32)

    if len(shape) == 1:
        if len(array) <= 2 * guard_cells + 2 * reference_cells:
            raise ValueError("Array too small for given parameters")
        intermediate = cp.cumsum(array, axis=0, dtype=cp.float32)
        N = 2 * reference_cells
        alpha = cfar_alpha(pfa, N)
        tpb = (32,)
        bpg = (
            (len(array) - 2 * reference_cells - 2 * guard_cells + tpb[0] - 1) // tpb[0],
        )
        _ca_cfar_1d_kernel(
            bpg,
            tpb,
            (
                array,
                intermediate,
                mask,
                len(array),
                N,
                cp.float32(alpha),
                guard_cells,
                reference_cells,
            ),
        )
    elif len(shape) == 2:
        if len(guard_cells) != 2 or len(reference_cells) != 2:
            raise TypeError("Guard and reference cells must be two " "dimensional.")
        guard_cells_x, guard_cells_y = guard_cells
        reference_cells_x, reference_cells_y = reference_cells
        if shape[0] - 2 * guard_cells_x - 2 * reference_cells_x <= 0:
            raise ValueError("Array first dimension too small for given " "parameters.")
        if shape[1] - 2 * guard_cells_y - 2 * reference_cells_y <= 0:
            raise ValueError(
                "Array second dimension too small for given " "parameters."
            )
        intermediate = cp.cumsum(array, axis=0, dtype=cp.float32)
        intermediate = cp.cumsum(intermediate, axis=1, dtype=cp.float32)
        N = 2 * reference_cells_x * (2 * reference_cells_y + 2 * guard_cells_y + 1)
        N += 2 * (2 * guard_cells_x + 1) * reference_cells_y
        alpha = cfar_alpha(pfa, N)
        tpb = (8, 8)
        bpg_x = (
            shape[0] - 2 * (reference_cells_x + guard_cells_x) + tpb[0] - 1
        ) // tpb[0]
        bpg_y = (
            shape[1] - 2 * (reference_cells_y + guard_cells_y) + tpb[1] - 1
        ) // tpb[1]
        bpg = (bpg_x, bpg_y)
        _ca_cfar_2d_kernel(
            bpg,
            tpb,
            (
                array,
                intermediate,
                mask,
                shape[0],
                shape[1],
                N,
                cp.float32(alpha),
                guard_cells_x,
                guard_cells_y,
                reference_cells_x,
                reference_cells_y,
            ),
        )
    return (mask, array - mask > 0)
```

### 2.3 _ca_cfar_2d_kernel（cp.RawKernel）

只读基准原文（行 369–436）：

```python
_ca_cfar_2d_kernel = cp.RawKernel(
    r"""
extern "C" __global__ void
_ca_cfar_2d_kernel(float * array, float * intermediate, float * mask,
                   int width, int height, int N, float alpha,
                   int guard_cells_x, int guard_cells_y,
                   int reference_cells_x, int reference_cells_y)
{
    int i_init = threadIdx.x+blockIdx.x*blockDim.x;
    int j_init = threadIdx.y+blockIdx.y*blockDim.y;
    int i, j, x, y, offset;
    int tro, tlo, blo, bro, tri, tli, bli, bri;
    float outer_area, inner_area, T;

    for (i=i_init; i<width-2*(guard_cells_x+reference_cells_x);
         i += blockDim.x*gridDim.x){
        for (j=j_init; j<height-2*(guard_cells_y+reference_cells_y);
             j += blockDim.y*gridDim.y){
            /* 'tri' is Top Right Inner (square), 'blo' is Bottom Left
             * Outer (square), etc. These are the corners at which
             * the intermediate array must be evaluated.
             */
            x = i+guard_cells_x+reference_cells_x;
            y = j+guard_cells_y+reference_cells_y;
            offset = x*height+y;

            tro = (x+guard_cells_x+reference_cells_x)*height+y+
                guard_cells_y+reference_cells_y;
            tlo = (x-guard_cells_x-reference_cells_x-1)*height+y+
                guard_cells_y+reference_cells_y;
            blo = (x-guard_cells_x-reference_cells_x-1)*height+y-
                guard_cells_y-reference_cells_y-1;
            bro = (x+guard_cells_x+reference_cells_x)*height+y-
                guard_cells_y-reference_cells_y-1;

            tri = (x+guard_cells_x)*height+y+guard_cells_y;
            tli = (x-guard_cells_x-1)*height+y+guard_cells_y;
            bli = (x-guard_cells_x-1)*height+y-guard_cells_y-1;
            bri = (x+guard_cells_x)*height+y-guard_cells_y-1;

            /* It would be nice to eliminate the triple
             * branching here, but it only occurs on the boundaries
             * of the array (i==0 or j==0). So it shouldn't hurt
             * overall performance much.
             */
            if (i>0 && j>0){
                outer_area = intermediate[tro]-intermediate[tlo]-
                    intermediate[bro]+intermediate[blo];
            } else if (i == 0 && j > 0){
                outer_area = intermediate[tro]-intermediate[bro];
            } else if (i > 0 && j == 0){
                outer_area = intermediate[tro]-intermediate[tlo];
            } else if (i == 0 && j == 0){
                outer_area = intermediate[tro];
            }

            inner_area = intermediate[tri]-intermediate[tli]-
                intermediate[bri]+intermediate[bli];

            T = outer_area-inner_area;
            T = alpha/N*T;
            mask[offset] = T;
        }
    }
}
""",
    "_ca_cfar_2d_kernel",
)
```

### 2.4 _ca_cfar_1d_kernel（cp.RawKernel）

只读基准原文（行 439–473）：

```python
_ca_cfar_1d_kernel = cp.RawKernel(
    r"""
extern "C" __global__ void
_ca_cfar_1d_kernel(float * array, float * intermediate, float * mask,
                   int width, int N, float alpha,
                   int guard_cells, int reference_cells)
{
    int i_init = threadIdx.x+blockIdx.x*blockDim.x;
    int i, x;
    int br, bl, sr, sl;
    float big_area, small_area, T;

    for (i=i_init; i<width-2*(guard_cells+reference_cells);
         i += blockDim.x*gridDim.x){
        x = i+guard_cells+reference_cells;

        br = x+guard_cells+reference_cells;
        bl = x-guard_cells-reference_cells-1;
        sr = x+guard_cells;
        sl = x-guard_cells-1;

        if (i>0){
            big_area = intermediate[br]-intermediate[bl];
        } else{
            big_area = intermediate[br];
        }
        small_area = intermediate[sr]-intermediate[sl];

        T = big_area-small_area;
        T = alpha/N*T;
        mask[x] = T;
    }
}
""",
    "_ca_cfar_1d_kernel",
)
```

---

## 3. docstring 逐行翻译与解释

### 3.1 cfar_alpha 的 docstring

| 原文 | 翻译/解释 |
|------|----------|
| `Computes the value of alpha corresponding to a given probability of false alarm and number of reference cells N.` | 计算与给定虚警概率和参考单元数 N 对应的 alpha 值。 |
| `pfa : float` — `Probability of false alarm.` | 虚警概率 $P_{fa}$，标量浮点数，典型值 $10^{-3}$ 到 $10^{-6}$。 |
| `N : int` — `Number of reference cells.` | 参考单元总数 $N$。1D 时 $N = 2R$，2D 时由矩形窗公式计算。 |
| `alpha : float` — `Alpha value.` | 返回的阈值乘子 $\alpha = N(P_{fa}^{-1/N} - 1)$。 |

### 3.2 ca_cfar 的 docstring

| 原文 | 翻译/解释 |
|------|----------|
| `Computes the cell-averaged constant false alarm rate (CA CFAR) detector threshold and returns for a given array.` | 对给定数组计算单元平均恒虚警率（CA-CFAR）检测器的阈值并返回。 |
| `array : ndarray` — `Array containing data to be processed.` | 输入功率数组。1D（距离像）或 2D（距离-多普勒图）。应为平方律检波后的功率值（指数分布噪声）。 |
| `guard_cells_x : int` — `One-sided guard cell count in the first dimension.` | **docstring 有误**：参数名实际为 `guard_cells`，1D 时为单侧保护单元数（int），2D 时为二元组 `(guard_cells_x, guard_cells_y)`。 |
| `guard_cells_y : int` — `One-sided guard cell count in the second dimension.` | 同上，仅 2D 时使用。 |
| `reference_cells_x : int` — `one-sided reference cell count in the first dimension.` | **docstring 有误**：参数名实际为 `reference_cells`，1D 时为单侧参考单元数（int），2D 时为二元组 `(reference_cells_x, reference_cells_y)`。 |
| `reference_cells_y : int` — `one-sided referenc cell count in the second dimension.` | 同上，仅 2D 时使用。注意原文 "referenc" 少了一个 "e"，是拼写错误。 |
| `pfa : float` — `Probability of false alarm.` | 虚警概率 $P_{fa}$，默认值 $10^{-3}$。 |
| `threshold : ndarray` — `CFAR threshold` | 返回值 1：CFAR 门限数组 `mask`，shape 与输入相同，float32。不可检测位置值为 0。 |
| `return : ndarray` — `CFAR detections` | 返回值 2：布尔检测数组 `array - mask > 0`，True 表示该单元检测到目标。注意返回名 "return" 是 Python 关键字，实际使用时通过元组解包获取。 |

**docstring 缺陷总结**：
1. 参数名与实际签名不一致（docstring 写 `guard_cells_x`/`guard_cells_y`，实际为 `guard_cells`）
2. "referenc" 拼写错误
3. 未说明 1D 与 2D 参数形式的差异
4. 返回值名 "return" 是 Python 关键字

---

## 4. 源代码逐行解释

### 4.1 cfar_alpha 函数（只读基准行 243–261）

| 行号 | 代码 | 解释 |
|------|------|------|
| 243 | `def cfar_alpha(pfa, N):` | 函数签名。`pfa`：虚警概率；`N`：参考单元总数。 |
| 244–260 | docstring | 见 3.1 节。 |
| 261 | `    return N * (pfa ** (-1.0 / N) - 1)` | 核心公式 $\alpha = N(P_{fa}^{-1/N} - 1)$。逐步展开：`-1.0 / N` 计算指数 $-1/N$；`pfa ** (-1.0 / N)` 计算 $P_{fa}^{-1/N}$；减 1 后乘 $N$ 得 $\alpha$。 |

### 4.2 ca_cfar 函数（只读基准行 264–366）

| 行号 | 代码 | 解释 |
|------|------|------|
| 264 | `def ca_cfar(array, guard_cells, reference_cells, pfa=1e-3):` | 函数签名。`array`：CuPy ndarray，1D 或 2D 功率数组；`guard_cells`：1D 时为 int（单侧保护单元数），2D 时为 `(int, int)`；`reference_cells`：同理；`pfa`：虚警概率，默认 $10^{-3}$。 |
| 265–295 | docstring | 见 3.2 节。 |
| 296 | `    shape = array.shape` | 获取输入数组形状。1D 时 shape 为 `(L,)`，2D 时为 `(M, Nd)`。 |
| 297–298 | `    if len(shape) > 2:\n        raise TypeError(...)` | 维度检查：只支持 1D 和 2D，3D 及以上抛出 TypeError。 |
| 299 | `    mask = cp.zeros(shape, dtype=cp.float32)` | 初始化输出门限数组 `mask`，与输入同 shape，全零 float32。不可检测位置保持为 0。 |

#### 4.2.1 一维分支（行 301–324）

| 行号 | 代码 | 解释 |
|------|------|------|
| 301 | `    if len(shape) == 1:` | 进入一维处理分支。 |
| 302–303 | `        if len(array) <= 2 * guard_cells + 2 * reference_cells:\n            raise ValueError(...)` | 数组长度检查：数组必须能容纳两侧保护+参考单元，即 $L > 2G + 2R$，否则至少有一个方向没有参考单元。 |
| 304 | `        intermediate = cp.cumsum(array, axis=0, dtype=cp.float32)` | 计算一维前缀和（累积和）数组 `intermediate`，dtype 强制 float32 防止精度溢出。`intermediate[k] = sum(array[0:k+1])`。这是前缀和加速的关键预计算。 |
| 305 | `        N = 2 * reference_cells` | 计算参考单元总数 $N = 2R$（两侧各 $R$ 个）。 |
| 306 | `        alpha = cfar_alpha(pfa, N)` | 调用 `cfar_alpha` 计算阈值乘子 $\alpha = N(P_{fa}^{-1/N} - 1)$。 |
| 307 | `        tpb = (32,)` | 设置每块线程数（threads per block）= 32。1D 配置，单维度。 |
| 308–310 | `        bpg = (\n            (len(array) - 2 * reference_cells - 2 * guard_cells + tpb[0] - 1) // tpb[0],\n        )` | 计算网格块数（blocks per grid）。可检测单元数 = $L - 2R - 2G$，向上取整除以 tpb[0]=32。公式 `(n + tpb - 1) // tpb` 等价于 `ceil(n / tpb)`，确保所有可检测 CUT 都有线程覆盖。 |
| 311–324 | `        _ca_cfar_1d_kernel(\n            bpg,\n            tpb,\n            (\n                array,\n                intermediate,\n                mask,\n                len(array),\n                N,\n                cp.float32(alpha),\n                guard_cells,\n                reference_cells,\n            ),\n        )` | 启动 1D CUDA kernel。参数依次为：grid 大小 `bpg`、block 大小 `tpb`、kernel 参数元组。kernel 参数：`array`（原始功率数组）、`intermediate`（前缀和数组）、`mask`（输出门限数组）、`len(array)`（数组长度 width）、`N`（参考单元总数）、`cp.float32(alpha)`（阈值乘子，强制 float32）、`guard_cells`（单侧保护单元数）、`reference_cells`（单侧参考单元数）。 |

#### 4.2.2 二维分支（行 325–365）

| 行号 | 代码 | 解释 |
|------|------|------|
| 325 | `    elif len(shape) == 2:` | 进入二维处理分支。 |
| 326–327 | `        if len(guard_cells) != 2 or len(reference_cells) != 2:\n            raise TypeError(...)` | 2D 时 `guard_cells` 和 `reference_cells` 必须是长度为 2 的序列（元组或列表），否则抛出 TypeError。 |
| 328 | `        guard_cells_x, guard_cells_y = guard_cells` | 解包保护单元数：`guard_cells_x` 为第一维（行/距离）单侧数，`guard_cells_y` 为第二维（列/多普勒）单侧数。 |
| 329 | `        reference_cells_x, reference_cells_y = reference_cells` | 解包参考单元数，同上。 |
| 330–331 | `        if shape[0] - 2 * guard_cells_x - 2 * reference_cells_x <= 0:\n            raise ValueError(...)` | 第一维长度检查：$M > 2G_x + 2R_x$。 |
| 332–335 | `        if shape[1] - 2 * guard_cells_y - 2 * reference_cells_y <= 0:\n            raise ValueError(...)` | 第二维长度检查：$N_d > 2G_y + 2R_y$。 |
| 336 | `        intermediate = cp.cumsum(array, axis=0, dtype=cp.float32)` | 第一步累积：沿行方向（axis=0）做 cumsum。`intermediate[m, n] = sum(array[0:m+1, n])`。 |
| 337 | `        intermediate = cp.cumsum(intermediate, axis=1, dtype=cp.float32)` | 第二步累积：沿列方向（axis=1）做 cumsum。结果 `intermediate[m, n] = sum(array[0:m+1, 0:n+1])`，即二维前缀和矩阵（积分图）。 |
| 338–339 | `        N = 2 * reference_cells_x * (2 * reference_cells_y + 2 * guard_cells_y + 1)\n        N += 2 * (2 * guard_cells_x + 1) * reference_cells_y` | 计算二维参考单元总数 $N = 2R_x(2R_y + 2G_y + 1) + 2(2G_x + 1)R_y$。这是矩形窗总面积减去保护区域面积的结果。 |
| 340 | `        alpha = cfar_alpha(pfa, N)` | 计算阈值乘子 $\alpha$。 |
| 341 | `        tpb = (8, 8)` | 2D 线程块配置：每块 $8 \times 8 = 64$ 个线程。比 1D 的 32 更大，因为 2D 有更多 CUT 要处理。 |
| 342–344 | `        bpg_x = (\n            shape[0] - 2 * (reference_cells_x + guard_cells_x) + tpb[0] - 1\n        ) // tpb[0]` | x 方向（行）的网格块数，向上取整。可检测行数 = $M - 2(R_x + G_x)$。 |
| 345–347 | `        bpg_y = (\n            shape[1] - 2 * (reference_cells_y + guard_cells_y) + tpb[1] - 1\n        ) // tpb[1]` | y 方向（列）的网格块数，向上取整。可检测列数 = $N_d - 2(R_y + G_y)$。 |
| 348 | `        bpg = (bpg_x, bpg_y)` | 组合为 2D 网格大小。 |
| 349–365 | `        _ca_cfar_2d_kernel(\n            bpg,\n            tpb,\n            (\n                array,\n                intermediate,\n                mask,\n                shape[0],\n                shape[1],\n                N,\n                cp.float32(alpha),\n                guard_cells_x,\n                guard_cells_y,\n                reference_cells_x,\n                reference_cells_y,\n            ),\n        )` | 启动 2D CUDA kernel。参数：`array`、`intermediate`、`mask`、`shape[0]`（width=行数）、`shape[1]`（height=列数）、`N`、`cp.float32(alpha)`、四个单元数参数。注意 kernel 参数名 `width` 对应 `shape[0]`（行数），`height` 对应 `shape[1]`（列数），采用行优先存储。 |

#### 4.2.3 返回值（行 366）

| 行号 | 代码 | 解释 |
|------|------|------|
| 366 | `    return (mask, array - mask > 0)` | 返回元组。`mask`：float32 门限数组，每个可检测位置存放 $T = \alpha/N \cdot \text{ref\_sum}$，不可检测位置为 0。`array - mask > 0`：布尔检测数组，等价于 `array > mask`，True 表示 CUT 功率超过门限，检测到目标。 |

### 4.3 _ca_cfar_1d_kernel 逐行解释（只读基准行 439–473 内的 CUDA 代码）

| CUDA 行 | 代码 | 解释 |
|---------|------|------|
| 1 | `extern "C" __global__ void` | CUDA kernel 声明，`extern "C"` 确保 C 链接（cp.RawKernel 要求），`__global__` 表示 device 函数由 host 调用。 |
| 2 | `_ca_cfar_1d_kernel(float * array, float * intermediate, float * mask, int width, int N, float alpha, int guard_cells, int reference_cells)` | kernel 参数：`array`（原始功率，只读）、`intermediate`（前缀和数组，只读）、`mask`（输出门限，写）、`width`（数组长度 $L$）、`N`（参考单元总数）、`alpha`（阈值乘子 float32）、`guard_cells`（单侧保护数 $G$）、`reference_cells`（单侧参考数 $R$）。 |
| 3 | `{` | 函数体开始。 |
| 4 | `    int i_init = threadIdx.x+blockIdx.x*blockDim.x;` | 计算本线程的全局线程索引（1D）。每个线程初始负责一个 CUT。 |
| 5 | `    int i, x;` | `i`：循环变量（CUT 在可检测范围内的偏移索引）；`x`：CUT 在数组中的实际位置。 |
| 6 | `    int br, bl, sr, sl;` | 前缀和查表索引：`br`（big right = 整个窗口右端）、`bl`（big left = 整个窗口左端-1）、`sr`（small right = 保护区域右端）、`sl`（small left = 保护区域左端-1）。 |
| 7 | `    float big_area, small_area, T;` | `big_area`：整个窗口（参考+保护+CUT）的功率和；`small_area`：保护区域+CUT 的功率和；`T`：最终门限。 |
| 8 | | 空行。 |
| 9 | `    for (i=i_init; i<width-2*(guard_cells+reference_cells);` | 循环遍历所有可检测 CUT。`i` 从线程初始索引开始，步长为总线程数 `blockDim.x*gridDim.x`（grid-stride loop 模式，确保线程数少于 CUT 数时仍能覆盖全部）。上界 `width-2*(G+R)` 即可检测 CUT 数量。 |
| 10 | `         i += blockDim.x*gridDim.x){` | 循环步长。 |
| 11 | `        x = i+guard_cells+reference_cells;` | 将可检测索引 `i` 转换为数组绝对位置 `x`。偏移量 $G+R$ 是因为前 $G+R$ 个单元无法作为 CUT。 |
| 12 | | 空行。 |
| 13 | `        br = x+guard_cells+reference_cells;` | `big_area` 右端索引：$x + G + R$，即整个窗口最右侧单元。 |
| 14 | `        bl = x-guard_cells-reference_cells-1;` | `big_area` 左端索引-1：$x - G - R - 1$。前缀和差值 `intermediate[br] - intermediate[bl]` 等于 $\sum_{k=x-G-R}^{x+G+R} x[k]$（整个窗口和）。 |
| 15 | `        sr = x+guard_cells;` | `small_area` 右端索引：$x + G$，即保护区域右端。 |
| 16 | `        sl = x-guard_cells-1;` | `small_area` 左端索引-1：$x - G - 1$。差值 `intermediate[sr] - intermediate[sl]` 等于 $\sum_{k=x-G}^{x+G} x[k]$（保护区域+CUT 和）。 |
| 17 | | 空行。 |
| 18–19 | `        if (i>0){\n            big_area = intermediate[br]-intermediate[bl];` | 非边界情况（$i > 0$，即不是最左侧 CUT）：`big_area = S[br] - S[bl]`，标准前缀和差值。 |
| 20–21 | `        } else{\n            big_area = intermediate[br];` | 边界情况（$i = 0$，最左侧 CUT）：`bl = -1`，前缀和 $S[-1]$ 不存在，等价于 0，所以 `big_area = S[br] - 0 = S[br]`。 |
| 22 | `        small_area = intermediate[sr]-intermediate[sl];` | `small_area` 无边界问题：最左 CUT 的 `sl = G+R-G-1 = R-1 >= 0`（因为 $R \geq 1$），所以 `sl` 始终有效。 |
| 23 | | 空行。 |
| 24 | `        T = big_area-small_area;` | 参考区域求和 = 整个窗口和 - 保护区域和 = `big_area - small_area`。 |
| 25 | `        T = alpha/N*T;` | 门限 $T = \frac{\alpha}{N} \times \text{ref\_sum}$。等价于 $T = \alpha \times \bar{Z}$（$\bar{Z} = \text{ref\_sum}/N$），但合并为一次乘法避免先除后乘。 |
| 26 | `        mask[x] = T;` | 将门限写入 `mask` 数组的 CUT 位置 `x`。 |
| 27 | `    }` | 循环结束。 |
| 28 | `}` | kernel 结束。 |

### 4.4 _ca_cfar_2d_kernel 逐行解释（只读基准行 369–436 内的 CUDA 代码）

| CUDA 行 | 代码 | 解释 |
|---------|------|------|
| 1–2 | `extern "C" __global__ void\n_ca_cfar_2d_kernel(...)` | CUDA kernel 声明。参数：`array`（原始功率）、`intermediate`（二维前缀和矩阵）、`mask`（输出）、`width`（行数 $M$）、`height`（列数 $N_d$）、`N`（参考单元总数）、`alpha`（阈值乘子）、四个保护/参考单元数。 |
| 3 | `{` | 函数体。 |
| 4 | `    int i_init = threadIdx.x+blockIdx.x*blockDim.x;` | x 方向线程索引。 |
| 5 | `    int j_init = threadIdx.y+blockIdx.y*blockDim.y;` | y 方向线程索引。2D kernel 使用 2D 线程块和网格。 |
| 6 | `    int i, j, x, y, offset;` | `i,j`：可检测范围内的偏移索引；`x,y`：CUT 在数组中的实际位置；`offset`：CUT 在行优先一维化数组中的线性偏移。 |
| 7 | `    int tro, tlo, blo, bro, tri, tli, bli, bri;` | 外矩形四角（tro=Top Right Outer, tlo=Top Left Outer, blo=Bottom Left Outer, bro=Bottom Right Outer）和内矩形四角（tri=Top Right Inner, ...）在前缀和矩阵中的线性索引。 |
| 8 | `    float outer_area, inner_area, T;` | 外矩形面积和、内矩形面积和、门限。 |
| 9 | | 空行。 |
| 10–11 | `    for (i=i_init; i<width-2*(guard_cells_x+reference_cells_x);\n         i += blockDim.x*gridDim.x){` | x 方向 grid-stride 循环，遍历可检测行。上界 `width - 2*(G_x + R_x)`。 |
| 12–13 | `        for (j=j_init; j<height-2*(guard_cells_y+reference_cells_y);\n             j += blockDim.y*gridDim.y){` | y 方向 grid-stride 循环，遍历可检测列。 |
| 14–17 | `            /* 'tri' is Top Right Inner ... */` | C 注释：解释四角命名规则。tri = Top Right Inner（内矩形右上角），blo = Bottom Left Outer（外矩形左下角）等。 |
| 18 | `            x = i+guard_cells_x+reference_cells_x;` | 将可检测偏移 `i` 转为数组行位置 `x`。 |
| 19 | `            y = j+guard_cells_y+reference_cells_y;` | 将可检测偏移 `j` 转为数组列位置 `y`。 |
| 20 | `            offset = x*height+y;` | 行优先线性偏移：`offset = x * height + y`。这是 2D 数组一维化后在 GPU 内存中的地址。 |
| 21 | | 空行。 |
| 22–23 | `            tro = (x+guard_cells_x+reference_cells_x)*height+y+\n                guard_cells_y+reference_cells_y;` | 外矩形右上角（Top Right Outer）的线性索引。行坐标 = $x + G_x + R_x$，列坐标 = $y + G_y + R_y$。这是容斥公式中 $S[r_2, c_2]$ 的位置。 |
| 24–25 | `            tlo = (x-guard_cells_x-reference_cells_x-1)*height+y+\n                guard_cells_y+reference_cells_y;` | 外矩形左上角偏移（Top Left Outer，行坐标-1）。容斥公式中 $-S[r_1-1, c_2]$ 的位置。行坐标 = $x - G_x - R_x - 1$。 |
| 26–27 | `            blo = (x-guard_cells_x-reference_cells_x-1)*height+y-\n                guard_cells_y-reference_cells_y-1;` | 外矩形左下角偏移（Bottom Left Outer）。容斥公式中 $+S[r_1-1, c_1-1]$ 的位置。 |
| 28–29 | `            bro = (x+guard_cells_x+reference_cells_x)*height+y-\n                guard_cells_y-reference_cells_y-1;` | 外矩形右下角偏移（Bottom Right Outer）。容斥公式中 $-S[r_2, c_1-1]$ 的位置。 |
| 30 | | 空行。 |
| 31 | `            tri = (x+guard_cells_x)*height+y+guard_cells_y;` | 内矩形右上角（Top Right Inner）：行 = $x + G_x$，列 = $y + G_y$。 |
| 32 | `            tli = (x-guard_cells_x-1)*height+y+guard_cells_y;` | 内矩形左上角偏移：行 = $x - G_x - 1$。 |
| 33 | `            bli = (x-guard_cells_x-1)*height+y-guard_cells_y-1;` | 内矩形左下角偏移。 |
| 34 | `            bri = (x+guard_cells_x)*height+y-guard_cells_y-1;` | 内矩形右下角偏移。 |
| 35 | | 空行。 |
| 36–39 | `            /* It would be nice to eliminate ... */` | C 注释：作者希望消除边界分支（$i=0$ 或 $j=0$ 时的特殊处理），但因为只影响数组最边缘的 CUT，对整体性能影响小。 |
| 40–42 | `            if (i>0 && j>0){\n                outer_area = intermediate[tro]-intermediate[tlo]-\n                    intermediate[bro]+intermediate[blo];` | 一般情况（$i > 0$ 且 $j > 0$）：标准容斥公式 `outer_area = S[tro] - S[tlo] - S[bro] + S[blo]`，四个角查表加减交替。 |
| 43–44 | `            } else if (i == 0 && j > 0){\n                outer_area = intermediate[tro]-intermediate[bro];` | $i = 0$（最左列 CUT）：`tlo` 和 `blo` 的行坐标为 -1，前缀和为 0。容斥简化为 `S[tro] - S[bro]`。 |
| 45–46 | `            } else if (i > 0 && j == 0){\n                outer_area = intermediate[tro]-intermediate[tlo];` | $j = 0$（最上行 CUT）：`bro` 和 `blo` 的列坐标为 -1，前缀和为 0。容斥简化为 `S[tro] - S[tlo]`。 |
| 47–48 | `            } else if (i == 0 && j == 0){\n                outer_area = intermediate[tro];` | $i = 0$ 且 $j = 0$（左上角 CUT）：只有 `tro` 项有效，其余三项前缀和均为 0。 |
| 49 | | 空行。 |
| 50–52 | `            inner_area = intermediate[tri]-intermediate[tli]-\n                intermediate[bri]+intermediate[bli];` | 内矩形面积。**无边界分支**：因为内矩形比外矩形小一圈（保护单元至少 1），即使外矩形触及边界，内矩形也不会。所以始终使用完整的四角容斥公式。 |
| 53 | | 空行。 |
| 54 | `            T = outer_area-inner_area;` | 参考区域求和 = 外矩形和 - 内矩形和。 |
| 55 | `            T = alpha/N*T;` | 门限 $T = \frac{\alpha}{N} \times \text{ref\_sum}$。 |
| 56 | `            mask[offset] = T;` | 写入 `mask` 数组的线性偏移位置。 |
| 57–58 | `        }\n    }` | 双层循环结束。 |
| 59 | `}` | kernel 结束。 |

---

## 5. 调用链与算法总结

### 5.1 一维调用链

```
ca_cfar(array, guard_cells, reference_cells, pfa)
  ├─ cp.cumsum(array, axis=0, dtype=float32)     → intermediate（前缀和）
  ├─ N = 2 * reference_cells                      → 参考单元总数
  ├─ cfar_alpha(pfa, N)                           → α = N(P_fa^{-1/N} - 1)
  ├─ _ca_cfar_1d_kernel(bpg, tpb, ...)            → GPU 并行计算每个 CUT 的门限
  │   ├─ big_area = S[br] - S[bl]                 → 整个窗口和（前缀和差值）
  │   ├─ small_area = S[sr] - S[sl]               → 保护区域+CUT 和
  │   ├─ ref_sum = big_area - small_area           → 参考区域和
  │   └─ T = α/N * ref_sum                        → 门限
  └─ return (mask, array - mask > 0)               → 门限数组 + 检测结果
```

### 5.2 二维调用链

```
ca_cfar(array, (Gx,Gy), (Rx,Ry), pfa)
  ├─ intermediate = cp.cumsum(cp.cumsum(array, axis=0), axis=1)  → 二维前缀和
  ├─ N = 2Rx(2Ry+2Gy+1) + 2(2Gx+1)Ry                          → 参考单元总数
  ├─ cfar_alpha(pfa, N)                                         → α
  ├─ _ca_cfar_2d_kernel(bpg, tpb, ...)                          → GPU 并行
  │   ├─ outer_area = 容斥(四角)                                → 外矩形和
  │   ├─ inner_area = 容斥(四角)                                → 内矩形和
  │   ├─ ref_sum = outer_area - inner_area                      → 参考区域和
  │   └─ T = α/N * ref_sum                                     → 门限
  └─ return (mask, array - mask > 0)
```

---

## 6. 数学映射、边界、复杂度与阅读检查

### 6.1 数学公式与代码映射

| 数学公式 | 代码位置 | 代码表达式 |
|----------|---------|-----------|
| $\alpha = N(P_{fa}^{-1/N} - 1)$ | `cfar_alpha` 函数体 | `N * (pfa ** (-1.0 / N) - 1)` |
| $N = 2R$（1D） | `ca_cfar` 行 305 | `N = 2 * reference_cells` |
| $N = 2R_x(2R_y+2G_y+1) + 2(2G_x+1)R_y$（2D） | `ca_cfar` 行 338–339 | 两行赋值 |
| $S[k] = \sum_{i=0}^{k} x[i]$（1D 前缀和） | `ca_cfar` 行 304 | `cp.cumsum(array, axis=0, dtype=cp.float32)` |
| $S[m,n] = \sum\sum x[i,j]$（2D 前缀和） | `ca_cfar` 行 336–337 | 两次 `cp.cumsum` |
| $T = \frac{\alpha}{N} \cdot \text{ref\_sum}$（1D） | 1D kernel 行 24–25 | `T = big_area-small_area; T = alpha/N*T;` |
| $T = \frac{\alpha}{N} \cdot \text{ref\_sum}$（2D） | 2D kernel 行 54–55 | `T = outer_area-inner_area; T = alpha/N*T;` |
| 检测：$x > T$ | `ca_cfar` 行 366 | `array - mask > 0` |

### 6.2 边界处理

| 边界情况 | 处理方式 | 代码位置 |
|---------|---------|---------|
| 数组维度 > 2 | 抛出 TypeError | 行 297–298 |
| 1D 数组太短（$L \leq 2G + 2R$） | 抛出 ValueError | 行 302–303 |
| 2D guard/reference 不是二元组 | 抛出 TypeError | 行 326–327 |
| 2D 行/列维度太短 | 抛出 ValueError | 行 330–335 |
| 1D kernel 边界 $i = 0$（`bl = -1`） | `big_area = intermediate[br]`（跳过 `S[-1]` 项） | 1D kernel 行 18–21 |
| 2D kernel 边界 $i=0$ 或 $j=0$ | 四种分支分别处理容斥公式中 $S[-1,\cdot]$ 和 $S[\cdot,-1]$ 为 0 的情况 | 2D kernel 行 40–48 |
| 2D 内矩形边界 | 无需特殊处理（内矩形不触及数组边界） | 2D kernel 行 50–52 |

### 6.3 dtype 与数值稳定性

- 前缀和强制 `dtype=cp.float32`，即使输入为 float64 也降精度。大数组累积和可能溢出 float32 范围（$\sim 3.4 \times 10^{38}$），但对典型雷达功率值（$\sim 10^6$ 以内）和数组长度（$\sim 10^4$）安全。
- `alpha` 传入 kernel 前用 `cp.float32(alpha)` 强制转换，确保 kernel 内 float 运算一致。
- `mask` 初始化为 float32 零数组，与 kernel 输出类型匹配。

### 6.4 时间与空间复杂度

| 操作 | 1D 时间 | 2D 时间 | 空间 |
|------|---------|---------|------|
| `cp.cumsum` | $O(L)$ | $O(MN_d)$（两次） | $O(L)$ / $O(MN_d)$ |
| `cfar_alpha` | $O(1)$ | $O(1)$ | $O(1)$ |
| CUDA kernel | $O(L)$（每个 CUT $O(1)$） | $O(MN_d)$（每个 CUT $O(1)$） | $O(L)$ / $O(MN_d)$（mask） |
| **总计** | $O(L)$ | $O(MN_d)$ | $O(L)$ / $O(MN_d)$ |

对比朴素方法（每个 CUT 逐元素累加参考单元）：

| 方法 | 1D 时间 | 2D 时间 |
|------|---------|---------|
| 朴素 | $O(LR)$ | $O(MN_d R_x R_y)$ |
| 前缀和 | $O(L)$ | $O(MN_d)$ |

### 6.5 可能的性能瓶颈

1. **前缀和计算是串行的**：`cp.cumsum` 是前缀和（scan）操作，理论上需要 $O(\log n)$ 步的并行 scan，但对 GPU 来说已经足够高效。
2. **2D kernel 的边界分支**：`if (i>0 && j>0)` 等四路分支只在数组最边缘的 CUT 触发，warp 内绝大多数线程走同一分支，对性能影响极小。
3. **float32 精度**：强制 float32 可能对高动态范围雷达数据引入量化误差，但这是工程权衡。

### 6.6 建议阅读顺序

1. **先读 `cfar_alpha`（行 243–261）**：最简单的标量函数，理解 $\alpha$ 的计算。
2. **读 `ca_cfar` 的 Python 部分（行 264–366）**：关注维度检查、前缀和构建、$N$ 计算、kernel 启动配置。
3. **读 `_ca_cfar_1d_kernel`（行 439–473）**：比 2D 简单，先理解 `big_area - small_area` 的前缀和差值法。
4. **读 `_ca_cfar_2d_kernel`（行 369–436）**：在 1D 基础上增加第二维循环、四角容斥和边界处理。

### 6.7 阅读检查问题

1. `cfar_alpha` 的返回值在 `ca_cfar` 中如何使用？为什么用 `cp.float32(alpha)` 转换？
2. 1D 和 2D 分支中，`intermediate` 数组分别是什么？2D 为什么需要两次 `cumsum`？
3. 1D kernel 中 `br, bl, sr, sl` 的含义分别是什么？`big_area - small_area` 等于什么？
4. 2D kernel 中 `tro, tlo, blo, bro` 对应容斥公式的哪些项？为什么 `inner_area` 不需要边界处理？
5. 1D kernel 中 `if (i>0)` 分支处理什么情况？此时 `bl` 的值是多少？
6. `mask` 中不可检测位置的值为什么是 0？这对检测结果有什么影响？
7. `array - mask > 0` 与 `array > mask` 是否完全等价？在 float32 运算中是否有差异？
