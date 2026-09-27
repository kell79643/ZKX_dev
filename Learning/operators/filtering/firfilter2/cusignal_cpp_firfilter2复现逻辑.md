# cusignal_cpp_firfilter2 复现逻辑

## 版本索引

| 版本 | Git 提交 | 状态 | 说明 |
| --- | --- | --- | --- |
| [V1](#v1当前学习版本fdd55ac) | `fdd55ac8415d70379eb38a2c299047f90bcf0a41` | 当前实现 | CPU 独立 reference；GPU 小系数融合路径和大系数通用路径 |

## V1：当前学习版本（fdd55ac）

### 1. 版本身份与可追溯性

- 完整 SHA：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`
- 分支：`final-prep/benchmark-evidence-v1`
- 提交时间：`2026-08-15T23:45:58+08:00`
- 提交主题：`test(archive): 增加任务结果范围治理审计`
- 检查日期：`2026-08-16`
- 当前算子相关的头文件、CPU、GPU、kernel 和直接算子测试均无未提交修改，可以作为正式 V1。

本文所有 C++/CUDA 行号均指上述提交。历史源码可用 `git -C ZKX show <SHA>:<路径>` 读取。

### 2. 文件、符号和职责

| 层 | 共享根目录 `ZKX_dev/` 下路径 | 符号 | V1 行号 | 职责 |
| --- | --- | --- | --- | --- |
| 参数/结果类型 | `ZKX/cusignal_cpp/src/filtering/filtering_typed.h` | `Firfilter2PadType`、`Firfilter2Method`、`Firfilter2Options`、结果结构 | 271-296 | API 数据契约 |
| 布局计划 | 同上 | `Firfilter2LayoutPlan`、`prepare_firfilter2_layout` | 483-497 | shape/axis/edge 展平信息 |
| GPU API | 同上 | `firfilter2_device<T>` | 570-582 | 对外 device 入口 |
| CPU API | 同上 | `firfilter2_typed_cpu<T>` | 824-838 | 对外 reference 入口 |
| 公共校验/布局 | `ZKX/cusignal_cpp/src/filtering/filtering_typed.cpp` | `prepare_firfilter2_layout` | 1226-1303 | 校验参数并生成布局 |
| CPU 辅助 | 同上 | `firfilter2_host_odd_reflect`、`firfilter2_output_dtype` | 1597-1627 | dtype 保持的 odd padding；输出类型选择 |
| CPU 核心 | 同上 | `firfilter2_typed_cpu` | 1630-1724 | 独立 Host 参考算法 |
| GPU wrapper | 同上 | `firfilter2_device` | 1726-1753 | 结果分配与 GPU 路径分派 |
| GPU 调度 | `ZKX/cusignal_cpp/src/filtering/filtering_typed.cu` | `firfilter2_fp32_compute_device` | 185-286 | 小/大系数路径选择及 kernel 顺序 |
| INT32 存储路径 | 同上 | `firfilter2_fp64_storage_compute_device` | 287-310 | FP32 结果直接写 FP64 位模式 |
| GPU 核心 | `ZKX/cusignal_cpp/src/filtering/filtering_kernels.cuh` | `firfilter2_*` | 873-1195 | padding、融合 FIR、状态、反转、裁边 |
| 单算子探针 | `ZKX/test_all/operators/cases/filtering/firfilter2/operator_probe.cu` | `options`、`run<T>` | 31-37 | CPU/GPU 对照及 JSON 落盘 |
| API smoke | `ZKX/test_all/operators/d4_filtering_extra_api_smoke.cu` | 直接调用 | 70-71 | identity FIR device API |
| 精度测试 | `ZKX/test_all/operators/d6_operator_fp32_accuracy.cu` | `runner.run` lambda | 608-613 | 非 identity FIR 与 host 对照 |

### 3. 接口与数据结构

```cpp
enum class Firfilter2PadType { odd, even, constant, none };
enum class Firfilter2Method { pad, gust };
enum class Firfilter2OutputDtype { fp32, fp64 };

struct Firfilter2Options {
    std::vector<int> shape;
    int axis = -1;
    Firfilter2PadType padtype = Firfilter2PadType::odd;
    std::optional<int> padlen;
    Firfilter2Method method = Firfilter2Method::pad;
    std::optional<int> irlen;
};
```

`shape` 描述展平连续存储的逻辑多维形状；`axis` 支持负索引；`padlen` 缺省时为 `3*b.size()`；`gust` 保留枚举但固定拒绝；pad 方法忽略 `irlen`。

CPU 和 device 结果都是 dtype variant：`INT32` 输入返回 `fp64` 容器，其余 FP32/FP16/INT16/INT8 返回 `fp32`。GPU INT32 小系数路径仍只做 FP32 算术，随后把 float 精确扩展成 double 存储位。

### 4. 共享布局计划

`prepare_firfilter2_layout` 按以下顺序工作：

1. 拒绝空系数、超过 `int` 索引范围的系数或输入。
2. 只接受 `pad/gust`，并对 `gust` 抛异常，匹配 cuSignal 23.08.00。
3. 验证四种 padtype。
4. 空 `shape` 时退化成一维 `{input_size}`；否则验证 extent 为正且乘积等于输入元素数。
5. 把负 `axis` 加 rank 归一化，再检查范围。
6. `state_count=M-1`；none 的 `edge=0`，否则取显式 `padlen` 或默认 `3M`。
7. 要求 `sample_count > edge`，并检查扩展长度及 workspace 不溢出 `int`。
8. 计算 row-major 展平参数：

$$
\text{inner}=\prod_{d>axis}shape[d],\qquad
\text{outer}=\frac{|x|}{shape[axis]\cdot inner}.
$$

目标轴 line 由 `(outer, inner)` 唯一定位；轴上相邻样本在展平数组中相隔 `inner_count`。

### 5. CPU 调用链与算法

```text
firfilter2_typed_cpu<T>
  → prepare_firfilter2_layout
  → firfilter2_output_dtype<T>
  → 系数 T 转 double
  → 后缀和构造 base_state
  → 对每条 axis line：
      收集 input_line<T>
      保持 T 语义进行 odd/even/constant/none padding
      padding T 转 double
      正向 direct FIR + 首端点缩放状态
      reverse
      反向 direct FIR + 正向末端点缩放状态
      reverse
      crop edge
      写 FP32 或 FP64 结果
```

#### 5.1 dtype 保持的 odd padding

数学表达式为 $2e-r$。整数 `T` 模拟对应固定位宽的模运算和有符号回绕；`__half` 在每一步经 `store/load` 保持半精度舍入；float 也先把 `2*endpoint` 存回 `T` 再相减。这复现 Python/CuPy 在输入 dtype 上先完成 padding 的行为。

#### 5.2 稳态状态与 FIR

对 FIR $b_0,\ldots,b_{M-1}$：

$$
zi[s]=\sum_{k=s+1}^{M-1}b_k,\qquad 0\le s<M-1.
$$

这是 $a=[1]$ 时 `lfilter_zi` 的简化。`filter_line(input, scale)` 在开头加入 `base_state[sample]*scale`，再累加合法范围的 direct convolution。

`line_count=outer_count*inner_count`。每条 line 按 stride `inner_count` 收集；正向 `scale=extended.front()`，反向 `scale=forward.back()`。两次 `std::reverse` 分别准备反向输入和恢复方向，最后读取 `backward[edge+sample]`。

CPU reference 不调用 GPU、科学库或 FFT。

### 6. GPU 调用链

```text
firfilter2_device<T>
  → prepare_firfilter2_layout
  → firfilter2_output_dtype<T>
  ├─ FP32 输出 → firfilter2_fp32_compute_device
  └─ INT32 / FP64 容器
      ├─ M <= 16 → firfilter2_fp64_storage_compute_device
      └─ M > 16  → FP32 computed → finalize_fp64_device_storage
```

#### 6.1 小系数融合路径（$M\le16$）

padtype 编码为 `0=odd, 1=even, 2=constant/none`，仅启动一次 `firfilter2_small_fused_axis_kernel<T>`。每个线程负责最终输出中的一个元素：

$$
index=blockIdx.x\cdot blockDim.x+threadIdx.x.
$$

线程反推出 `inner/sample/outer`，按需读取 padding；`firfilter2_small_forward_value` 计算正向值；`firfilter2_small_fused_axis_value` 直接组合反向 FIR 和状态项，不物化 padded、forward、reverse 或 crop workspace。

INT32 用相同 float 结果，再由 `firfilter2_fp32_to_fp64_storage_bits` 把 binary32 的符号、指数、尾数扩展为 binary64 位模式；这是精确 widening，不是 FP64 算术。

#### 6.2 大系数通用路径（$M>16$）

kernel 顺序为：

1. 系数 `T→float`；
2. `pad_axis_kernel`；
3. `base_state_kernel`；
4. `scale_state_axis_kernel(use_last=false)`；
5. 正向 `firfilter_axis_output_kernel`；
6. `reverse_axis_kernel`；
7. `scale_state_axis_kernel(use_last=true)`；
8. 反向 `firfilter_axis_output_kernel`；
9. `reverse_axis_kernel`；
10. `crop_axis_kernel`。

同一默认 stream 的启动顺序构成依赖；API 内没有 Host 同步，D2H 或测试的 `synchronize_stream()` 才建立 Host 可见性。

### 7. kernel 线程职责

| kernel | 启动元素数 | 每线程职责 |
| --- | --- | --- |
| `small_fused_axis_kernel` | 原输入元素数 | 计算一个最终输出 |
| `pad_axis_kernel` | 扩展元素数 | 计算一个 padding/原始区样本 |
| `base_state_kernel` | `M-1` | 计算一个状态后缀和 |
| `scale_state_axis_kernel` | `outer*(M-1)*inner` | 计算一个缩放状态 |
| `firfilter_axis_output_kernel` | 扩展元素数 | 计算一次 FIR 的一个输出 |
| `reverse_axis_kernel` | 扩展元素数 | 复制轴上镜像位置 |
| `crop_axis_kernel` | 原输入元素数 | 读取 `edge+sample` 写最终元素 |

索引分解为：

$$
inner=index\bmod inner\_count,
$$

$$
sample=\left\lfloor\frac{index}{inner\_count}\right\rfloor\bmod sample\_count,
$$

$$
outer=\left\lfloor\frac{index}{sample\_count\cdot inner\_count}\right\rfloor.
$$

所有 kernel 先用 `if(index>=total)return;` 保护不满 block 的线程。

### 8. Python—CPU—GPU 三方映射

| 原理或公式 | cuSignal Python | C++ CPU | C++ GPU | 差异 |
| --- | --- | --- | --- | --- |
| padtype / 默认 $3M$ | `filtering.py:363-401` | `.cpp:1267-1285,1680-1704` | `.cu:235-241`；kernels `873-976` | C++ 先按 T padding |
| $zi[s]=\sum_{k>s}b_k$ | `firfilter_zi→lfilter_zi` | `.cpp:1646-1654` | kernels `979-985` 或 `1122-1133` | FIR 特例数学等价 |
| 正向 FIR | `filtering.py:492` | `.cpp:1655-1668,1708` | 融合 `989-1017` 或 `.cu:254-259` | direct，不用 FFT |
| 反转 | `497,500` | `.cpp:1710,1712` | 融合索引或 `1158-1175` | 融合不物化数组 |
| 反向 FIR | `497` | `.cpp:1709-1712` | 融合 `1020-1056` 或 `.cu:270-276` | 算法相同 |
| 裁边 | `502-504` | `.cpp:1713-1720` | 融合直接选 sample 或 `1177-1195` | 融合无独立 crop |
| 输出 dtype | `cp.result_type` | `.cpp:1622-1627` | `.cpp:1735-1750` | INT32 容器 FP64，其余 FP32 |

共同不变量是 $H(e^{j\omega})H(e^{-j\omega})$。融合路径只是消除中间存储和 kernel 边界，属于“原理不变、实现优化”。

### 9. 模板、内存、错误与精度

- `T` 由 `detail::is_simple_signal_input_v<T>` 限制为五种业务类型，`static_assert` 编译期检查。
- `.cpp:2781` 显式实例化 API；`.cu:683-688` 实例化 device helper。
- 输入是连续 row-major；`shape` 只是逻辑视图，不支持任意 stride。
- CPU line/vector 是 Host workspace；GPU 小路径无中间 workspace，大路径有多个 O(extended_size) device buffer。
- 参数错误抛 `std::invalid_argument`；`gust` 固定拒绝，pad 下 `irlen` 忽略。
- GPU 主算术为 FP32；CPU 用 double 计算后收敛至契约 dtype。两次 FIR 和不同求值顺序可能产生小量舍入差异。
- `M=1` 时 `state_count=0`，跳过状态 kernel；`sample_count` 必须严格大于 `edge`。
- 不使用 FFT、科学库、RAND 或 device FP64 算术。
- 本阶段未重新构建或运行，不能把仓库历史 PASS 冒充本轮验证。

### 10. 测试与风险

`operator_probe.cu:31-37` 构造二维 shape、三抽头系数，分别测 CPU/GPU，比较输出并把 dtype、axis、padding、shape、误差、耗时和内存阶段写入 JSON。README 声明覆盖五 dtype、三规模和三种 padding。

`d4_filtering_extra_api_smoke.cu:70-71` 验证 constant、padlen 0、identity FIR；`d6_operator_fp32_accuracy.cu:608-613` 用非 identity FIR 对照 host。

风险：探针固定三抽头，只命中融合路径；大系数通用路径需独立用例。还应补 none、负 axis、默认 padlen、`M=1`、边界异常和 gust 拒绝。

### 11. 自检问题与参考答案

1. **问：V1 为什么可追溯？**  
   **答：**完整 SHA 为 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`，相关文件无 dirty，证据给出提交路径、符号和行号。
2. **问：CPU 核心和 GPU wrapper 如何区分？**  
   **答：**CPU `.cpp:1630-1724` 独立完成算法；wrapper `.cpp:1726-1753` 只校验、分配和分派，device 核心在 `.cu/.cuh`。
3. **问：稳态状态为何是后缀和？**  
   **答：**FIR 的 $a=[1]$，单位阶跃稳态简化为 $zi[s]=\sum_{k=s+1}^{M-1}b_k$；CPU `1649-1654` 与 kernel `979-985/1122-1133` 对应。
4. **问：每个融合线程负责什么？**  
   **答：**`kernels.cuh:1094-1105` 中每线程计算一个最终输出，内部按需求 padding、正向值和反向值。
5. **问：大路径如何同步？**  
   **答：**默认 stream 内十步 kernel 按序；API 不 Host 同步，D2H 或显式 `synchronize_stream()` 才等待。
6. **问：INT32 为何返回 FP64 却没有 FP64 算术？**  
   **答：**计算为 float；`kernels.cuh:1058-1090` 只做 binary32→binary64 位扩展。
7. **问：融合是否改变数学算法？**  
   **答：**不改变；仍是 padding、两次端点状态、正反 FIR 和裁边，只是不物化中间数组。
8. **问：最大测试缺口是什么？**  
   **答：**三抽头探针只覆盖 $M\le16$，需用 $M>16$ 验证通用多 kernel 路径。

## 版本差异与原理不变量

目前只有 V1。未来追加 V2 时不得改写 V1 的 SHA、行号和结论。数学不变量为：

$$
H_{fb}(e^{j\omega})=H(e^{j\omega})H(e^{-j\omega})=|H(e^{j\omega})|^2.
$$
