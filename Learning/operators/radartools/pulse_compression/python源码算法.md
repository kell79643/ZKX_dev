# pulse_compression Python 源码算法

> 本文件对应阶段二：cuSignal 23.08.00 Python 源码算法。

---

## 1. 代码定位与接口概览

| 项目 | 值 |
|------|-----|
| 公开导入路径 | `cusignal.pulse_compression` |
| 顶层导出入口 | `cusignal/__init__.py`:59 |
| 模块导出入口 | `cusignal/radartools/__init__.py`:19 |
| 函数定义文件 | `cusignal/radartools/radartools.py` |
| 函数符号名 | `pulse_compression` |
| 只读基准行号 | 第 21-75 行 |
| 学习副本行号 | 第 23-89 行（含学习注释） |
| 直接依赖 | `cusignal/windows/windows.py` 中的 `get_window`（只读基准第 2099 行） |

---

## 2. 当前算子的完整相关源码

以下摘录只读基准（`ZKX/cusignal-23.08.00`）中 `pulse_compression` 的完整定义，包括签名、完整 docstring、函数体。同一文件中 `pulse_doppler`、`ambgfun`、`cfar_alpha`、`ca_cfar` 等其他算子不属于当前算子，不予摘录。

```python
# 只读基准 cusignal/radartools/radartools.py:14-18
from math import ceil, log2

import cupy as cp

from ..windows.windows import get_window
```

```python
# 只读基准 cusignal/radartools/radartools.py:21-75
def pulse_compression(x, template, normalize=False, window=None, nfft=None):
    """
    Pulse Compression is used to increase the range resolution and SNR
    by performing matched filtering of the transmitted pulse (template)
    with the received signal (x)

    Parameters
    ----------
    x : ndarray
        Received signal, assume 2D array with [num_pulses, sample_per_pulse]

    template : ndarray
        Transmitted signal, assume 1D array

    normalize : bool
        Normalize transmitted signal

    window : array_like, callable, string, float, or tuple, optional
        Specifies the window applied to the signal in the Fourier
        domain.

    nfft : int, size of FFT for pulse compression. Default is number of
        samples per pulse

    Returns
    -------
    compressedIQ : ndarray
        Pulse compressed output
    """
    [num_pulses, samples_per_pulse] = x.shape

    if nfft is None:
        nfft = samples_per_pulse

    if window is not None:
        Nx = len(template)
        if callable(window):
            W = window(cp.fft.fftfreq(Nx))
        elif isinstance(window, cp.ndarray):
            if window.shape != (Nx,):
                raise ValueError("window must have the same length as data")
            W = window
        else:
            W = get_window(window, Nx, False)

        template = cp.multiply(template, W)

    if normalize is True:
        template = cp.divide(template, cp.linalg.norm(template))

    fft_x = cp.fft.fft(x, nfft)
    fft_template = cp.conj(cp.tile(cp.fft.fft(template, nfft), (num_pulses, 1)))
    compressedIQ = cp.fft.ifft(cp.multiply(fft_x, fft_template), nfft)

    return compressedIQ
```

### 直接依赖 `get_window` 的签名与语义

`pulse_compression` 调用 `get_window(window, Nx, False)` 时，参数含义：

- `window`：窗函数名称字符串（如 `"hamming"`）、浮点数（Kaiser 窗的 beta）或元组（如 `("kaiser", 4.0)`）
- `Nx`：窗函数长度，必须等于模板长度 `len(template)`
- `fftbins=False`：生成对称窗（`sym=True`），用于滤波器设计而非频谱分析。对称窗在两端衰减到零，适合匹配滤波模板加权

`get_window` 内部通过 `_win_equiv` 字典将窗函数名称映射到对应的实现函数，并传入 `(Nx, *args, sym)` 参数。完整实现见 `cusignal/windows/windows.py:2099-2203`。

---

## 3. Docstring 逐行翻译与解释

| 行号 | 原文 | 翻译/解释 |
|------|------|-----------|
| 22 | `"""` | docstring 开始 |
| 23-24 | `Pulse Compression is used to increase the range resolution and SNR by performing matched filtering of the transmitted pulse (template) with the received signal (x)` | 脉冲压缩用于通过将发射脉冲（模板）与接收信号（x）进行匹配滤波来提高距离分辨率和信噪比 |
| 27 | `Parameters` | 参数说明开始 |
| 29 | `x : ndarray` | x：n 维数组 |
| 30 | `Received signal, assume 2D array with [num_pulses, sample_per_pulse]` | 接收信号，假设为 2D 数组，形状为 [脉冲数, 每脉冲采样数] |
| 32 | `template : ndarray` | template：n 维数组 |
| 33 | `Transmitted signal, assume 1D array` | 发射信号，假设为 1D 数组 |
| 35 | `normalize : bool` | normalize：布尔值 |
| 36 | `Normalize transmitted signal` | 是否对发射信号（模板）进行归一化 |
| 38 | `window : array_like, callable, string, float, or tuple, optional` | window：可选的窗函数，类型可以是类数组、可调用对象、字符串、浮点数或元组 |
| 39-40 | `Specifies the window applied to the signal in the Fourier domain.` | 指定施加到傅里叶域信号的窗函数。（注：实际代码是在时域对模板加权，docstring 描述有歧义，但效果等价） |
| 42 | `nfft : int, size of FFT for pulse compression. Default is number of samples per pulse` | nfft：整数，脉冲压缩的 FFT 长度。默认为每脉冲采样数 |
| 45 | `Returns` | 返回值说明 |
| 47 | `compressedIQ : ndarray` | compressedIQ：n 维数组 |
| 48 | `Pulse compressed output` | 脉冲压缩输出 |
| 49 | `"""` | docstring 结束 |

---

## 4. 源代码逐行解释

以下按只读基准行号（第 21-75 行）逐行解释函数体内每一行非空内容。

| 基准行号 | 代码 | 解释 |
|----------|------|------|
| 21 | `def pulse_compression(x, template, normalize=False, window=None, nfft=None):` | 函数签名。5 个参数：接收信号 x、模板 template、归一化开关 normalize（默认 False）、窗函数 window（默认 None）、FFT 长度 nfft（默认 None） |
| 50 | `[num_pulses, samples_per_pulse] = x.shape` | 解包接收信号形状。`num_pulses` 为脉冲数（行数，慢时间维度），`samples_per_pulse` 为每脉冲采样数（列数，快时间维度）。**注意**：此处要求 x 必须是 2D 数组，否则解包失败并抛出 ValueError |
| 52 | `if nfft is None:` | 判断是否未指定 FFT 长度 |
| 53 | `nfft = samples_per_pulse` | 默认 FFT 长度取每脉冲采样数。物理含义：对快时间方向做与距离单元数等长的 FFT |
| 55 | `if window is not None:` | 判断是否指定了窗函数 |
| 56 | `Nx = len(template)` | 取模板长度作为窗函数长度。窗函数序列 W 必须与 template 等长才能逐点相乘 |
| 57 | `if callable(window):` | 分支 1：window 是可调用对象（如 lambda 或自定义函数） |
| 58 | `W = window(cp.fft.fftfreq(Nx))` | 用归一化频率 `fftfreq(Nx)` = [0, 1/Nx, ..., (Nx/2-1)/Nx, -Nx/2/Nx, ..., -1/Nx] 作为自变量调用 window，生成窗序列。这是 SciPy `scipy.signal.get_window` 中 callable 分支的标准做法 |
| 59 | `elif isinstance(window, cp.ndarray):` | 分支 2：window 已经是 CuPy 数组 |
| 60 | `if window.shape != (Nx,):` | 检查窗数组长度是否等于模板长度 |
| 61 | `raise ValueError("window must have the same length as data")` | 长度不匹配时报错 |
| 62 | `W = window` | 直接使用传入的数组作为窗序列 |
| 63 | `else:` | 分支 3：window 是字符串（如 `"hamming"`）、浮点数（Kaiser beta）或元组（如 `("kaiser", 4.0)`） |
| 64 | `W = get_window(window, Nx, False)` | 调用 `get_window` 生成窗序列。`fftbins=False` → `sym=True`，生成对称窗（两端值为 0），用于匹配滤波模板的时域加权 |
| 66 | `template = cp.multiply(template, W)` | 时域加权：模板逐点乘以窗函数。数学表示 $template[n] \leftarrow template[n] \cdot W[n]$。这将修改 template 变量的绑定（但若 template 是 GPU 数组则原地/新分配取决于 CuPy 内部实现） |
| 68 | `if normalize is True:` | 判断是否归一化。注意用 `is True` 而非 `if normalize`，因此只接受布尔值 True，不接受 truthy 值如 1 |
| 69 | `template = cp.divide(template, cp.linalg.norm(template))` | 归一化：$template[n] \leftarrow template[n] / \|template\|$，其中 $\|template\| = \sqrt{\sum_m |template[m]|^2}$ 是 L2 范数。归一化后 $\|template\| = 1$ |
| 71 | `fft_x = cp.fft.fft(x, nfft)` | **频域匹配滤波步骤 1**：对 2D 接收信号 x 沿最后一轴做 nfft 点 FFT。`cp.fft.fft` 对 2D 数组默认沿 axis=-1（列方向/快时间方向）。输出形状 `[num_pulses, nfft]`，dtype 为复数 |
| 72 | `fft_template = cp.conj(cp.tile(cp.fft.fft(template, nfft), (num_pulses, 1)))` | **频域匹配滤波步骤 2+3**：三步合一——(a) 对 1D 模板做 nfft 点 FFT 得 1D 频谱；(b) `cp.conj` 取复共轭，实现频域匹配滤波器 $H[k] = S^*[k]$（共轭等价于时域时间反转共轭 $h[n]=s^*[T-n]$）；(c) `cp.tile(..., (num_pulses, 1))` 将 1D 频谱共轭沿行方向复制 num_pulses 次，变为 `[num_pulses, nfft]` 的 2D 数组，使每个脉冲行都能与模板频谱共轭逐点相乘 |
| 73 | `compressedIQ = cp.fft.ifft(cp.multiply(fft_x, fft_template), nfft)` | **频域匹配滤波步骤 4+5**：`cp.multiply(fft_x, fft_template)` 频域逐点相乘（接收信号频谱 × 模板频谱共轭），`cp.fft.ifft(..., nfft)` 做 nfft 点 IFFT 还原到时域。输出 compressedIQ 形状 `[num_pulses, nfft]`，dtype 为复数。数学表示 $compressedIQ = \text{IFFT}(\text{FFT}(x) \cdot \text{conj}(\text{FFT}(template)))$ |
| 75 | `return compressedIQ` | 返回脉冲压缩结果 |

---

## 5. 调用链与算法总结

### 调用链

```
cusignal.pulse_compression(x, template, normalize, window, nfft)
  └── cusignal/radartools/__init__.py:19  →  cusignal/radartools/radartools.py:21  pulse_compression()
        ├── [window 分支 3] cusignal/windows/windows.py:2099  get_window(window, Nx, False)
        │     └── _win_equiv[winstr]  →  具体窗函数实现（如 hamming、hann 等）
        ├── cp.fft.fft(x, nfft)           # CuPy GPU FFT
        ├── cp.fft.fft(template, nfft)    # CuPy GPU FFT
        ├── cp.conj(...)                  # CuPy 逐元素共轭
        ├── cp.tile(...)                  # CuPy 数组复制
        ├── cp.multiply(...)              # CuPy 逐元素相乘
        └── cp.fft.ifft(...)              # CuPy GPU IFFT
```

### 算法总结

`pulse_compression` 实现了**频域匹配滤波**，是脉冲压缩的标准数字实现：

1. **预处理**：可选的时域窗函数加权（抑制旁瓣）和模板 L2 归一化（统一幅度）
2. **核心计算**：FFT → 共轭相乘 → IFFT，利用卷积定理将时域卷积转为频域乘法
3. **并行化**：通过 tile 将 1D 模板频谱共轭广播到所有脉冲行，利用 CuPy GPU 并行加速

时间复杂度：$O(N_p \cdot N \log N)$，其中 $N_p$ = num_pulses，$N$ = nfft
空间复杂度：$O(N_p \cdot N)$，主要来自 fft_x 和 fft_template 两个 `[num_pulses, nfft]` 数组

---

## 6. 数学映射

| 源码行 | 代码表达式 | 数学公式 | 原理编号 |
|--------|-----------|----------|----------|
| 64 | `get_window(window, Nx, False)` | $W[n],\ n=0,\ldots,N_x-1$ | 原理 4 |
| 66 | `cp.multiply(template, W)` | $s_w[n] = s[n] \cdot W[n]$ | 原理 4 |
| 69 | `cp.divide(template, cp.linalg.norm(template))` | $s_{norm}[n] = s[n] / \|s\|$ | 原理 5 |
| 71 | `cp.fft.fft(x, nfft)` | $X[k,:] = \text{DFT}(x[:,n])$ | 原理 3 |
| 72 | `cp.conj(cp.fft.fft(template, nfft))` | $H[k] = S^*[k]$ | 原理 2, 3 |
| 72 | `cp.tile(..., (num_pulses, 1))` | $H[k,:] \rightarrow [H[k,:]; \ldots; H[k,:]]$（$N_p$ 行） | 原理 6 |
| 73 | `cp.multiply(fft_x, fft_template)` | $Y[k,:] = X[k,:] \cdot H[k,:]$ | 原理 2, 3 |
| 73 | `cp.fft.ifft(..., nfft)` | $y[:,n] = \text{IDFT}(Y[k,:])$ | 原理 3 |

---

## 7. 边界处理、异常检查与数值稳定性

| 场景 | 处理方式 | 源码行 |
|------|---------|--------|
| x 不是 2D 数组 | `[num_pulses, samples_per_pulse] = x.shape` 解包失败，Python 自动抛出 ValueError | 50 |
| window 是 cp.ndarray 但长度不匹配 | `raise ValueError("window must have the same length as data")` | 60-61 |
| nfft 未指定 | 默认 `nfft = samples_per_pulse` | 52-53 |
| template 长度 > samples_per_pulse 且 nfft=samples_per_pulse | **无显式检查**，可能产生循环卷积混叠。这是 cuSignal 的隐含假设：模板长度 ≤ 每脉冲采样数 |
| normalize=True 但 template 为全零向量 | `cp.linalg.norm(template)` = 0，除法产生 inf/nan，**无显式保护** |
| 窗函数名称无效 | `get_window` 内部抛出 `ValueError("Unknown window type.")` | 64 → get_window |

---

## 8. 时间复杂度、空间复杂度与性能瓶颈

### 时间复杂度

设 $N_p$ = num_pulses，$N$ = nfft，$N_t$ = len(template)。

| 步骤 | 操作 | 复杂度 |
|------|------|--------|
| 窗函数生成 | `get_window(window, Nx, False)` | $O(N_t)$ |
| 时域加权 | `cp.multiply(template, W)` | $O(N_t)$ |
| 归一化 | `cp.linalg.norm` + `cp.divide` | $O(N_t)$ |
| FFT(x) | `cp.fft.fft(x, nfft)` — $N_p$ 个 $N$ 点 FFT | $O(N_p \cdot N \log N)$ |
| FFT(template) + conj + tile | 1 个 $N$ 点 FFT + conj + 复制 | $O(N \log N + N_p \cdot N)$ |
| 频域相乘 | `cp.multiply(fft_x, fft_template)` | $O(N_p \cdot N)$ |
| IFFT | `cp.fft.ifft(..., nfft)` — $N_p$ 个 $N$ 点 IFFT | $O(N_p \cdot N \log N)$ |

**总复杂度**：$O(N_p \cdot N \log N)$，由 FFT/IFFT 主导。

### 空间复杂度

- `fft_x`：$N_p \times N$ 复数
- `fft_template`：$N_p \times N$ 复数
- `compressedIQ`：$N_p \times N$ 复数
- **总计**：$O(N_p \cdot N)$ 复数元素

### 性能瓶颈

1. **FFT/IFFT 是计算密集核心**：两个 `cp.fft.fft`（其中一个对 2D 数组）和一个 `cp.fft.ifft` 占绝大部分时间
2. **tile 引入额外显存**：模板频谱共轭从 1D 复制为 $N_p \times N$ 的 2D 数组，当 $N_p$ 很大时显存开销不可忽视。优化方向：用 CuPy broadcasting 代替 tile，避免显存复制
3. **中间数组 multiply**：`cp.multiply(fft_x, fft_template)` 创建临时数组，可通过 fused kernel 合并到 IFFT 前避免

---

## 9. 阅读检查

### 建议阅读顺序

1. 先读函数签名（第 21 行）和 docstring（第 22-49 行），理解输入输出语义
2. 读第 50 行的形状解包，确认 2D 数据模型
3. 读第 52-53 行的 nfft 默认值，理解 FFT 长度选取
4. 读第 55-66 行的窗函数加权逻辑，关注三个分支的处理差异
5. 读第 68-69 行的归一化
6. 重点读第 71-73 行的三行核心计算，将每行映射到频域匹配滤波五步骤
7. 结合数学映射表（第 6 节）验证代码与原理的对应

### 检查问题

1. 为什么 `cp.fft.fft(x, nfft)` 不需要指定 `axis` 参数？CuPy 对 2D 数组的默认 FFT 轴是什么？
2. `cp.conj` 在频域取共轭如何等价于时域的时间反转共轭？如果直接用时域方法实现匹配滤波，代码会怎样写？
3. `cp.tile` 的作用是什么？如果去掉 tile 直接用 1D 频谱与 2D fft_x 相乘，CuPy 的 broadcasting 规则允许吗？结果一样吗？
4. `normalize is True` 为什么用 `is` 而不是 `==`？如果传入 `normalize=1` 会怎样？
5. 如果 `template` 长度大于 `samples_per_pulse` 且 `nfft=None`，输出会正确吗？为什么？
6. `get_window(window, Nx, False)` 中 `False` 的含义是什么？它如何影响窗函数的对称性？
7. 三行核心计算（71-73）中，哪一行对应原理 2 的频域定义 $H(f)=S^*(f)$？哪一行对应卷积定理的频域相乘？
8. 窗函数在时域对模板加权后，频域匹配滤波输出与无加权时的差异是什么？对主瓣和旁瓣分别有什么影响？
