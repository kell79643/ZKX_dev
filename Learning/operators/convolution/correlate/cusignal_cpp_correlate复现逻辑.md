# cusignal_cpp_correlate复现逻辑

## 版本索引

- **当前实现：[V1：原始学习版本（fdd55ac8）](#v1原始学习版本fdd55ac8)**
- 尚无 V2/V3。

## V1：原始学习版本（fdd55ac8）

### 1. 版本身份与范围

- 完整 Git SHA：`fdd55ac8415d70379eb38a2c299047f90bcf0a41`
- 分支：`final-prep/benchmark-evidence-v1`
- 提交时间：`2026-08-15T23:45:58+08:00`
- 检查日期：2026-08-16
- 相关文件 dirty 状态：`git status --short -- <相关文件>` 无输出，因此本 V1 可追溯。
- 输入范围：连续 row-major 实数，公开业务类型为 FP32、FP16、INT32、INT16、INT8；没有公开复数模板实例。
- 本阶段只读 `ZKX/cusignal_cpp`，未修改源码，未构建、运行或连接 ZQ500。

### 2. 带 SHA 的代码定位

以下每条均绑定提交 `fdd55ac8415d70379eb38a2c299047f90bcf0a41`：

| 层次 | `ZKX_dev/` 相对路径 | 符号 | 该提交行号 |
| --- | --- | --- | --- |
| 公开方法/参数 | `ZKX/cusignal_cpp/src/convolution/convolution_typed.h` | `CorrelateMethod`、`CorrelateNDOptions` | 58-93 |
| shape helper | 同上 | `correlate_output_size` | 95-106 |
| FFT workspace | 同上 | `CorrelateDeviceWorkspace`、`CorrelateNDWorkspace` | 108-190 |
| CPU/API 声明 | 同上 | `correlate_typed_cpu`、`correlate_device` | 250-337 |
| shape 实现 | `ZKX/cusignal_cpp/src/convolution/convolution_typed.cpp` | `correlate_output_size`、`correlate_nd_output_shape` | 134-201 |
| CPU 一维 | 同上 | `correlate_typed_cpu<T>(..., mode, method)` | 284-340 |
| CPU N 维 | 同上 | `correlate_typed_cpu<T>(..., options)` | 342-413 |
| GPU workspace/调度 | `ZKX/cusignal_cpp/src/convolution/convolution_typed.cu` | `reset`、`correlate_device` | 134-380 |
| GPU kernels | `ZKX/cusignal_cpp/src/convolution/convolution_kernels.cuh` | direct/pack/multiply/crop kernels | 75-246 |
| 针对性测试 | `ZKX/cusignal_cpp/test/signal_processing/e3_convolution_type_smoke.cu` | `run_type<T>` | 54 起 |

### 3. 对外接口与职责分层

`CorrelateMethod` 提供 `direct`、`fft`、`auto_select`。`auto_select` 不形成第三种数学语义：一维/CPU 选 direct，多维选 FFT。`CorrelateNDOptions` 携带两个 shape、mode 和 method。

两类 GPU 入口必须区分：

- `correlate_device(in1,in2,out,mode)`：无 workspace，对应一维 direct。
- `correlate_device(...,CorrelateDeviceWorkspace&)`：一维 FFT。
- `correlate_device(...,CorrelateNDWorkspace&)`：任意 rank，rank 0 走标量 kernel，rank 1 可 direct/FFT，rank > 1 只允许 FFT。

CPU `correlate_typed_cpu` 是独立 reference，不读 GPU 结果，不调用 `FFTInterface`。它的 `method=fft` 主要模拟 FP32 FFT 边界转换，核心仍用直接求和产生期望值。

### 4. CPU 一维调用链与算法

```text
correlate_typed_cpu<T>
→ 校验 T/method/mode
→ correlate_output_size
→ 计算 full_len 与居中 crop start
→ 每个 out_idx 扫描 in2
→ 把 full_idx 和 kernel j 映射到 in1 shifted
→ 边界内乘加
→ 按 dtype/method 写回
```

核心索引是

$$
i=\text{full\_idx}-(L_2-1-j).
$$

若 $0\le i<L_1$，则累加 $x[i]y[j]$。因为公开范围是实数，共轭是恒等操作；$j$ 从 0 递增而 $i$ 按反转相对位置对齐，对应 $x*y[-n]$。

dtype 策略：

- INT32 direct：无符号模 $2^{32}$ 乘加，最后按二补码解释。
- 窄整数和浮点：FP32 累加；浮点使用 `std::fma`。
- FFT 参考语义：FP32 结果 ties-to-even 舍入，再同宽环绕，不饱和。

### 5. CPU N 维调用链

CPU N 维先由 `correlate_nd_output_shape` 校验 rank/method/mode 并计算输出。标量直接相乘；rank 1 转调一维 overload；rank > 1 对每个输出坐标枚举整个 kernel 坐标，逐轴计算

$$
s_a=c_a+o_a-(M_a-1-k_a),
$$

其中 $c_a$ 是 crop 起点，$o_a$ 是输出坐标，$k_a$ 是 kernel 坐标。任一轴越界就跳过该乘积。连续 row-major stride 把坐标变回线性索引。

CPU 时间复杂度约为 $O(PQ)$，$P$ 为输出元素数，$Q$ 为第二输入元素数；结果空间 $O(P)$，坐标/stride 辅助空间 $O(r)$。

### 6. GPU direct：wrapper 与 kernel

GPU direct wrapper 先用 `validate_1d` 检查：业务 dtype、mode、输出长度、32 位 device 索引容量以及输入/输出不别名。空输入稳定返回。然后以 `out.size()` 为总工作量启动 `correlate_real_direct_kernel<T>`。

每个 CUDA 线程负责一个 `output_index`：

1. `full_index=start+output_index`。
2. `begin/end` 预先缩小到不越界的 kernel 范围。
3. 循环内计算 `input_index=full_index-(len2-1-kernel_index)`。
4. `correlate_direct_accumulate` 实现与 CPU 相同的 dtype 策略。
5. `correlate_direct_store` 做最终边界转换。

grid/block 由通用 `cuda_utils::launch_1d_kernel` 决定；本 kernel 的工作划分是“一线程一输出”，线程内顺序扫描重叠模板。没有 shared-memory reduction，也没有 kernel 内同步。默认 stream 上异步启动，读取前由调用者同步。

### 7. GPU FFT：workspace、数据搬运与计算

#### 7.1 一维

`CorrelateDeviceWorkspace::reset` 计算

$$
L_{full}=L_1+L_2-1,\qquad
L_{fft}=2^{\lceil\log_2L_{full}\rceil},
$$

分配六块 `ComplexFloat[L_fft]` scratch：两块时域、两块频域、乘积频域和 IFFT 时域，并创建一个 `FFTInterface` plan。

执行顺序：

```text
pack_1d(in1, reverse=false) → FFT ─┐
                                      ├→ multiply → IFFT → crop/store
pack_1d(in2, reverse=true)  → FFT ─┘
```

`pack_1d_kernel` 在装入第二输入时使用 `source=length-1-index`，因公开输入是实数，反转就等于共轭反转。`multiply_kernel` 执行标准复乘。`crop_1d_kernel` 取 IFFT 实部，再按 dtype 舍入/环绕或浮点存储。

该路径不做隐式 H2D/D2H，workspace 可跨调用复用，但同一 workspace 不得并发使用。空输入不创建 plan。

#### 7.2 N 维

`CorrelateNDWorkspace::reset` 为每轴计算 full shape、二次幂 FFT shape、crop start 和 row-major strides，将 shape/stride/crop 小数组复制到 device，并为每一轴创建 `FFTInterface`。

`pack_nd_real_kernel` 将线性索引解码为坐标；`reverse=true` 时每轴做 `shape[axis]-1-coordinate`。`transform_nd` 从最后一轴向前处理：`pack_nd_axis_kernel` 把当前轴的每条线打包成批次连续数据，调用 batched 1D FFT/IFFT，再用 `unpack_nd_axis_kernel` 恢复 row-major 布局。

两输入变换后逐元素复乘，再做逆向 N 维变换。`crop_nd_kernel` 将输出线性索引解码为坐标，加上逐轴 crop start，从 FFT buffer 读实部并转回 T。

FFT 时间约为 $O(F\sum_a\log F_a)$，$F=\prod_aF_a$；workspace 主要空间为多块 $O(F)$ `ComplexFloat` 数组。

### 8. template、类型分发与实例化

CPU/GPU 公开函数和 direct/pack/crop kernel 均以 `template<typename T>` 编写。`detail::is_convolution_input_v<T>` 在编译期限制业务类型；`ConvolutionTypePolicy<T>` 统一加载/存储边界；`if constexpr` 在整数和浮点路径间做编译期分支。

`convolution_typed.cpp:489-490` 和 `convolution_typed.cu:428-430` 使用宏展开显式实例化，使支持类型在库中生成稳定符号。FFT 统一使用 `ComplexFloat`，是 ZQ500 禁止 device FP64 下的项目边界；整数结果必须再按公开 T 环绕转换。

### 9. 与 cuSignal Python 的对应和差异

| 原理或公式 | cuSignal Python 落实位置 | C++ CPU 落实位置 | C++ GPU 落实位置 | 差异说明 |
| --- | --- | --- | --- | --- |
| $R=x*y^*[-n]$ | `correlate.py:138-152` | `convolution_typed.cpp:304-330` | direct `kernels.cuh:86-108`；FFT pack `110-128` | C++ 公开为实数，共轭为恒等 |
| full/same/valid | Python 卷积截取/direct 边界 | `correlate_output_size` 和 `start` | wrapper 传 `typed_output_start` | `same` 始终以原 in1 长度/shape 为准 |
| direct | Python 预编译 fatbin | 独立 host 求和 | 项目 CUDA kernel | C++ 扩展 FP16/INT16/INT8，明确 dtype 环绕语义 |
| FFT | `_reverse_and_conj`+CuPy FFT | CPU 模拟 FFT 边界 | `FFTInterface`+ComplexFP32 | C++ 为 ZQ500 避免 device FP64；填零选二次幂 |
| N 维 | CuPy `fftn/ifftn` | 独立 N 维枚举 reference | 逐轴 batched 1D FFT | 表达不同，线性相关与 crop 不变 |

Python 23.08 的显式复数 direct kernel 有漏共轭问题，而 `cusignal_cpp` 没有宣称支持复数业务输入，因此 C++ 实数子集不继承该矛盾。

### 10. 测试证据与未覆盖风险

`e3_convolution_type_smoke.cu:54` 起对每个 T：

- 用独立 CPU reference 生成 direct/FFT 期望值；
- 运行 GPU direct 和一维 FFT workspace 路径；
- 检查非法 mode、非法 method、空输入仍校验 mode、输出与输入别名拒绝；
- 覆盖 `in1` 短于 `in2` 时的 `same/valid`；
- 后续代码还构造 3D shape，覆盖 N 维 FFT 路径。

本次未运行测试，因而只能说明“存在此测试证据和断言设计”，不声称 V1 在当前环境重新通过。未覆盖/仍需关注：超大 shape 极限、workspace 并发误用、FFT 较大误差、非有限浮点输入、多轴奇偶 shape crop 组合以及未公开的复数语义。

### 11. 学习自检问题与参考答案

1. **问：V1 绑定哪个版本？**  
   **答：**`fdd55ac8415d70379eb38a2c299047f90bcf0a41`，分支 `final-prep/benchmark-evidence-v1`，相关文件 clean；见本文第 1 节。
2. **问：CPU reference 和 GPU 的职责有何不同？**  
   **答：**CPU 独立枚举并生成期望值；GPU direct 用一线程一输出 kernel，GPU FFT 用 workspace 和 `FFTInterface`，不做隐式 host 传输。
3. **问：相关公式如何落到 FFT 代码？**  
   **答：**`pack_1d_kernel`/`pack_nd_real_kernel` 反转第二实数输入，两者 FFT 后复乘，IFFT 得 $x*y[-n]$，最后 crop。
4. **问：template 和 dtype 策略如何配合？**  
   **答：**`T` 决定公开边界，`is_convolution_input_v` 限制类型，`ConvolutionTypePolicy` 做加载/存储，整数 direct 使用模乘加，FFT 用 FP32 后舍入环绕。
5. **问：GPU direct 如何并行？**  
   **答：**`launch_1d_kernel` 按输出元素数启动，每个线程计算一个输出并在线程内顺序扫描重叠 kernel；见 `kernels.cuh:86-108`。
6. **问：哪些同步由库内完成？**  
   **答：**kernel/FFT 在默认 stream 按提交顺序建立依赖，函数不做主机同步；读取 device 结果前由调用者同步。
7. **问：ZQ500 精度边界是什么？**  
   **答：**device FFT 不用 FP64，统一 `ComplexFloat`；因此整数 FFT 与某些 cuSignal 内部提升路径可有误差差异，再按 T 进行 ties-to-even 舍入和同宽环绕。
8. **问：测试边界是什么？**  
   **答：**typed smoke 覆盖五种类型、direct/FFT、mode、短长交换、异常和 3D；本次没有实际运行，且超大 shape、并发 workspace 和极端数值仍需专项验证。

## 版本差异与原理不变量

| 比较项 | V1（fdd55ac8） | 后续版本 | 原理不变量 |
| --- | --- | --- | --- |
| CPU | 一维/N 维独立枚举 | 待追加 | 线性互相关的滞后与 crop 语义 |
| GPU direct | 一线程一输出 | 待追加 | 重叠区间乘加 |
| GPU FFT | 反转第二输入+逐轴 FFT | 待追加 | $R=x*y[-n]$ 与线性填零 |
| 精度 | ComplexFP32 FFT、按 T 边界转换 | 待追加 | 公开 dtype/shape/mode 契约 |
