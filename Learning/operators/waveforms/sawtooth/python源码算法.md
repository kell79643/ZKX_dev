# sawtooth 算子 — Python 源码算法

## 一、公开导入路径与定义位置

| 项目 | 内容 |
| --- | --- |
| 公开导入路径 | `cusignal.sawtooth` |
| 定义文件 | `Learning/cusignal-23.08.00/python/cusignal/waveforms/waveforms.py` |
| 函数定义起始行 | 第 59 行（学习副本，含学习注释；只读基准原行号为第 45 行） |
| Kernel 定义起始行 | 第 20 行（学习副本；只读基准原行号为第 17 行） |
| 模块导出 | `Learning/cusignal-23.08.00/python/cusignal/waveforms/__init__.py` 第 17 行 |
| 包级导出 | `Learning/cusignal-23.08.00/python/cusignal/__init__.py` 第 81 行 |
| 符号名 | `sawtooth`（公开函数）、`_sawtooth_kernel`（内部 kernel，下划线前缀表示私有） |

---

## 二、函数签名与参数语义

```python
def sawtooth(t, width=1.0):
```

| 参数 | 类型 | 默认值 | 语义 |
| --- | --- | --- | --- |
| `t` | `array_like`（标量、list、numpy ndarray、CuPy ndarray） | 无 | 时间数组，单位为弧度。周期隐含为 $2\pi$。若需物理频率 $f$ Hz 的波形，调用时传入 $t' = 2\pi f t$ |
| `width` | `array_like`（标量或与 `t` 等长的数组） | `1.0` | 上升斜坡占整个周期的比例。$w=1$ 纯上升锯齿，$w=0$ 纯下降锯齿，$w=0.5$ 三角波。可为数组以实现时变波形 |

| 返回值 | 类型 | 语义 |
| --- | --- | --- |
| `y` | `cupy.ndarray`，dtype 为 `float64` | 输出锯齿/三角波形数组，值域 $[-1, 1]$（`width` 越界时为 NaN），shape 与 `t` 广播后一致 |

**dtype 说明**：kernel 声明输出为 `"float64 y"`，因此无论输入 `t` 是 `float32` 还是 `float64`，输出始终为 `float64`。

---

## 三、调用链

```
cusignal.sawtooth(t, width)
    │
    ├── cp.asarray(t)      → 将 t 转为 CuPy 数组
    ├── cp.asarray(width)   → 将 width 转为 CuPy 数组
    │
    └── _sawtooth_kernel(t, w)   → CuPy ElementwiseKernel（GPU kernel）
            │
            ├── mask1: width 越界检查
            │   └── out = NaN（若 w > 1 或 w < 0）
            │
            ├── tmod = fmod(t, 2π)
            │
            ├── mask2: 上升段判断
            │   └── out = tmod / (π·w) - 1
            │
            └── mask3: 下降段判断
                └── out = (π(w+1) - tmod) / (π(1-w))
```

调用链非常短，仅一层：公开函数 → GPU kernel。无中间辅助函数、无 FFT、无卷积。

---

## 四、按源码执行顺序拆解

### 4.1 `_sawtooth_kernel` 定义（第 20–55 行，学习副本行号）

```python
_sawtooth_kernel = cp.ElementwiseKernel(
    "T t, T w",
    "float64 y",
    """...""",
    "_sawtooth_kernel",
    options=("-std=c++11",),
)
```

**CuPy ElementwiseKernel 语法说明**：

- `cp.ElementwiseKernel` 是 CuPy 提供的轻量级 GPU kernel 定义工具，将逐元素运算直接编译为 CUDA kernel。
- 第一个参数 `"T t, T w"` 声明输入：`T` 是类型参数，CuPy 会根据实际传入数组的 dtype 自动实例化为 `float` 或 `double`。
- 第二个参数 `"float64 y"` 声明输出：固定为 `double` 类型。
- 第三个参数是 kernel 代码字符串，使用 C++ 语法，CuPy 编译时自动添加 CUDA kernel 包装。
- `options=("-std=c++11",)` 指定编译选项，因为 kernel 中使用了 C++11 的花括号初始化（`double out {}`）。

### 4.2 Kernel 内部执行逻辑

#### 步骤 1：初始化局部输出变量

```cpp
double out {};
```

- 声明 `double` 类型局部变量 `out`，花括号 `{}` 是 C++11 的值初始化（value initialization），对 `double` 等价于初始化为 `0.0`。
- `out` 暂存当前线程的计算结果，最后赋给输出参数 `y`。

#### 步骤 2：width 越界检查（mask1）

```cpp
const bool mask1 { ( ( w > 1 ) || ( w < 0 ) ) };
if ( mask1 ) {
    out = nan("0xfff8000000000000ULL");
}
```

- `mask1`：当 $w > 1$ 或 $w < 0$ 时为 `true`。
- `nan("0xfff8000000000000ULL")`：生成 IEEE 754 双精度静默 NaN。位模式 `0xfff8000000000000` 含义：符号位=1、指数全1、尾数最高位=1（静默 NaN 标志），其余尾数为0。
- **注意**：此步骤只给 `out` 赋值，不提前返回。后续 mask2、mask3 的条件中包含 `(1 - mask1)` 守卫，确保越界时不会进入计算分支。

#### 步骤 3：取模归约

```cpp
const T tmod { fmod( t, 2.0 * M_PI ) };
```

- `fmod(t, 2π)`：C 标准库的浮点取模，结果符号与 `t` 相同，范围 $[0, 2\pi)$（对正数 `t`）。
- `M_PI`：CUDA math 库提供的 $\pi$ 常量，定义为 `3.14159265358979323846`。
- 对应数学原理中的 $t_{\text{mod}} = t \bmod 2\pi$。

**`fmod` 与 Python `%` 的区别**：Python 的 `%` 对负数取模结果为正（如 `-0.5 % 2π ≈ 5.78`），而 C 的 `fmod` 结果符号与被除数相同（如 `fmod(-0.5, 2π) ≈ -0.5`）。当 `t` 含负值时，cuSignal 的 `fmod` 可能产生负的 `tmod`，此时 `tmod < w·2π` 条件仍然成立（负数 < 正数），上升段公式正常计算。但数学上，负 $t$ 对应的周期位置应为 $t_{\text{mod}} + 2\pi$。**这是一个潜在的边界问题**，下文第八节详述。

#### 步骤 4：上升段判断（mask2）

```cpp
const bool mask2 { ( ( 1 - mask1 ) && ( tmod < ( w * 2.0 * M_PI ) ) ) };
if ( mask2 ) {
    out = tmod / ( M_PI * w ) - 1;
}
```

- `(1 - mask1)`：`mask1` 为 `bool`，`1 - true = 0`，`1 - false = 1`。当 width 越界时此条件为假，阻止进入上升段。
- `tmod < w * 2π`：判断当前元素的取模时间是否落在上升段。
- 上升段公式 `out = tmod / (π·w) - 1`：对应数学原理 1 的 $y = \frac{t_{\text{mod}}}{\pi w} - 1$。

**映射**：

| 数学公式 | 代码 | 对应行号 |
| --- | --- | --- |
| $y = \frac{t_{\text{mod}}}{\pi w} - 1$ | `tmod / ( M_PI * w ) - 1` | 第 40 行（学习副本） |

#### 步骤 5：下降段判断（mask3）

```cpp
const bool mask3 { ( ( 1 - mask1 ) && ( 1 - mask2 ) ) };
if ( mask3 ) {
    out = ( M_PI * ( w + 1 ) - tmod ) / ( M_PI * ( 1 - w ) );
}
```

- `(1 - mask2)`：mask2 为假时（即 `tmod >= w·2π`），此条件为真。
- 下降段公式对应数学原理 1 的 $y = \frac{\pi(w+1) - t_{\text{mod}}}{\pi(1-w)}$。

**映射**：

| 数学公式 | 代码 | 对应行号 |
| --- | --- | --- |
| $y = \frac{\pi(w+1) - t_{\text{mod}}}{\pi(1-w)}$ | `( M_PI * ( w + 1 ) - tmod ) / ( M_PI * ( 1 - w ) )` | 第 47 行（学习副本） |

#### 步骤 6：写回输出

```cpp
y = out;
```

将局部变量 `out` 赋给 CuPy 的输出参数 `y`，CuPy 自动将结果写回 GPU 显存对应的输出数组位置。

### 4.3 `sawtooth()` 函数（第 59–96 行，学习副本行号）

```python
def sawtooth(t, width=1.0):
    t, w = cp.asarray(t), cp.asarray(width)
    y = _sawtooth_kernel(t, w)
    return y
```

**逐行说明**：

1. **第 91 行**：`cp.asarray(t)` 将任意 `array_like` 输入转为 CuPy ndarray。如果输入已经是 CuPy 数组，`asarray` 不做复制；如果是 numpy 数组或 Python list，则复制到 GPU 显存。`cp.asarray(width)` 同理。
2. **第 94 行**：调用 `_sawtooth_kernel(t, w)`。CuPy ElementwiseKernel 的调用机制：
   - 自动按输入数组的长度确定 GPU 线程数
   - 支持 CuPy broadcasting：如果 `t` 和 `w` shape 不同但可广播，自动扩展
   - 预分配输出数组（dtype 为 `float64`，shape 为广播后的 shape）
   - 启动 CUDA kernel，每个线程处理一个元素
3. **第 96 行**：返回输出数组 `y`。

---

## 五、数学公式与代码语句的逐项映射

| 数学公式 | 代码语句 | 学习副本行号 | 只读基准行号 |
| --- | --- | --- | --- |
| $t_{\text{mod}} = t \bmod 2\pi$ | `fmod( t, 2.0 * M_PI )` | 34 | 27 |
| $\text{越界检查：} w > 1 \text{ 或 } w < 0$ | `( w > 1 ) \|\| ( w < 0 )` | 27 | 22 |
| $\text{越界输出：NaN}$ | `nan("0xfff8000000000000ULL")` | 30 | 24 |
| $y = \frac{t_{\text{mod}}}{\pi w} - 1$（上升段） | `tmod / ( M_PI * w ) - 1` | 40 | 31 |
| $y = \frac{\pi(w+1) - t_{\text{mod}}}{\pi(1-w)}$（下降段） | `( M_PI * ( w + 1 ) - tmod ) / ( M_PI * ( 1 - w ) )` | 47 | 36 |
| $y \leftarrow \text{out}$ | `y = out` | 50 | 38 |

---

## 六、底层机制

### 6.1 CuPy ElementwiseKernel

`cp.ElementwiseKernel` 是 CuPy 的核心 GPU 编程抽象，它将用户编写的逐元素 C++/CUDA 代码字符串在首次调用时 JIT 编译为 CUDA kernel，之后缓存复用。

**工作流程**：
1. 首次调用时，CuPy 将 kernel 代码字符串包装为完整的 CUDA kernel 函数
2. 使用 NVRTC（NVIDIA Runtime Compiler）编译为 PTX/CUBIN
3. 按输入数组长度计算 grid/block 维度，启动 kernel
4. 后续调用直接使用缓存的编译结果

**线程索引**：ElementwiseKernel 自动提供内置变量 `i`（当前线程的全局一维索引），sawtooth kernel 未使用此变量，因为计算仅依赖当前元素值。

### 6.2 GPU 并行策略

- 每个 GPU 线程独立处理一个元素（一个 `t[i]` 和对应的 `w[i]`）
- 无线程间同步需求，无共享内存使用
- 线程数 = 输入数组长度，由 CuPy 自动分配 grid/block

---

## 七、边界处理、异常检查与数值稳定性

### 7.1 width 越界

- $w > 1$ 或 $w < 0$ → 输出 NaN
- 实现方式：通过 mask1 守卫，不抛 Python 异常，不中断 kernel 执行

### 7.2 width = 0 和 width = 1

- **$w = 0$**：`mask2` 条件 `tmod < 0` 对所有 $t_{\text{mod}} \ge 0$ 为假（因为 $w \cdot 2\pi = 0$），全部走 mask3 下降段。下降段分母 `M_PI * (1 - w) = M_PI`，不除零。公式退化为 `out = (M_PI - tmod) / M_PI = 1 - tmod/π`，产生从 $+1$ 到 $-1$ 的下降斜坡。
- **$w = 1$**：`mask2` 条件 `tmod < 2π` 对所有 $t_{\text{mod}} \in [0, 2\pi)$ 为真，全部走 mask2 上升段。上升段分母 `M_PI * w = M_PI`，不除零。公式退化为 `out = tmod / M_PI - 1`，产生从 $-1$ 到 $+1$ 的上升斜坡。

### 7.3 负时间值与 fmod

C 标准库的 `fmod` 对负数的行为：`fmod(-1.0, 2π) ≈ -1.0`（结果符号与被除数相同）。而数学上锯齿波在负时间处应有周期延拓，`-1.0` 对应的周期位置应为 $2\pi - 1.0 \approx 5.28$。

**实际影响**：
- 当 $t < 0$ 时，`tmod` 可能为负，`tmod < w·2π` 仍成立（负数 < 正数），走上升段
- 上升段公式 `tmod/(πw) - 1` 对负 `tmod` 也能计算，但结果不在 $[-1, 1]$ 的预期范围内
- 例如 $t = -\pi$，$w = 1$：`tmod = fmod(-π, 2π) ≈ -π`，`out = -π/π - 1 = -2`，超出 $[-1, 1]$

**这与 SciPy 的行为不同**：SciPy 使用 Python 的 `%` 运算符（对浮点数为 `np.mod`/`np.fmod` 的正余数版本），负数取模结果为正。cuSignal 的 `fmod` 行为在负时间时可能产生非预期输出。**这是 cuSignal 与 SciPy 的已知差异**。

### 7.4 NaN 传播

- 若 `t` 中含 NaN：`fmod(NaN, 2π)` 返回 NaN，后续比较结果不确定，最终 `out` 可能未被任何分支赋值（保持初始 0.0），也可能被某个分支以 NaN 计算
- 实际行为取决于 CUDA `fmod` 和比较运算对 NaN 的处理，通常 NaN 比较为 false，因此 mask2 和 mask3 均可能为假，`out` 保持 0.0

---

## 八、时间复杂度、空间复杂度及性能瓶颈

| 指标 | 分析 |
| --- | --- |
| 时间复杂度 | $O(N)$，$N$ 为输入数组长度。每个元素一次 `fmod` + 一次比较 + 一次除法 + 一次加减 |
| 空间复杂度 | $O(N)$，输入数组 + 输出数组，无额外临时缓冲区 |
| kernel 启动开销 | ElementwiseKernel 首次调用有 JIT 编译开销（约秒级），后续调用使用缓存 |
| 性能瓶颈 | `fmod` 是 kernel 中最重的运算（涉及浮点除法和取整），分支预测可能因 width 数组导致 warp divergence |

---

## 九、建议用户亲自阅读源码的顺序

1. **先看 `sawtooth()` 函数**（第 59–96 行）：理解公开 API 的三层操作——输入转换 → kernel 调用 → 返回
2. **再看 `_sawtooth_kernel` 的参数声明**（第 20–22 行）：理解 ElementwiseKernel 的输入/输出类型签名
3. **然后看 kernel 内部执行顺序**：
   - mask1 越界检查（第 27–31 行）
   - tmod 取模（第 34 行）
   - mask2 上升段判断与公式（第 36–41 行）
   - mask3 下降段判断与公式（第 44–48 行）
   - out 写回 y（第 50 行）
4. **每段代码应观察的问题**：
   - mask1/mask2/mask3 的守卫逻辑如何保证三种情况互斥且完整覆盖？
   - `(1 - mask1)` 这种将 `bool` 隐式转为 `int` 再参与逻辑运算的写法，在 CUDA 中是否安全？
   - `fmod` 对负数的行为与数学周期延拓有何差异？
   - 为什么输出 dtype 固定为 `float64` 而不是与输入 `T` 一致？
   - `width` 为数组时，CuPy 如何实现 `t` 和 `w` 的元素级配对？

---

## 十、cuSignal sawtooth 与 SciPy sawtooth 的关键差异

| 对比项 | cuSignal | SciPy |
| --- | --- | --- |
| 取模方式 | C `fmod(t, 2π)`，负数结果为负 | Python `t % (2π)`（等价 `np.fmod` 正余数），负数结果为正 |
| 负时间行为 | 可能产生超出 $[-1, 1]$ 的值 | 始终在 $[-1, 1]$ 内 |
| 执行方式 | GPU CUDA kernel，逐元素并行 | CPU NumPy 向量化，逐元素并行 |
| 输出 dtype | 固定 `float64` | 与输入 dtype 一致 |
| NaN 处理 | IEEE 754 位模式硬编码 | NumPy 默认 NaN |
