# ZQ500 DLFFT CPU 内存泄漏量化说明

## 1. 结论

2026-07-17 在 ZQ500 `gpu_02` 容器内完成的供应商库最小复现表明：当前 ZQ500 SDK
预编译库 `/zq500/sdk/lib/libdlfft.so` 的 `cufftExecC2C` 执行路径存在可重复的
CPU live heap 近似线性增长。

在固定一个长度为 8 的 FFT plan、固定两块 device buffer、预热 10 轮后，执行组在
100 个采样轮次内共调用 `cufftExecC2C` 1,900 次，CPU live heap 净增
**234,608 B（229.109 KiB）**；100 个采样步全部上涨、0 次回落。相同程序的控制组不
调用 `cufftExecC2C`，CPU live heap 净增长为 **0 B**。两组 GPU 占用净增长均为
**0 B**。

因此，本次实测确认的是 **DLFFT execute 路径的同进程 CPU 慢泄漏**，没有观察到该
最小用例的 GPU 显存泄漏。进程退出后操作系统能够回收进程资源，但常驻进程反复调用
DLFFT 时，CPU 内存会持续累积，不能把“进程退出可回收”写成“库本身无泄漏”。

## 2. 最小复现如何排除项目代码

最小程序为
[`test_all/operators/dlfft_vendor_heap_repro.cu`](../../../../test_all/operators/dlfft_vendor_heap_repro.cu)，
只链接 `dlfft`、`curt` 和用于符号归属检查的系统 `libdl`，没有链接项目
`FFTInterface`、算子实现、Task1 或 Task2。

控制组与执行组使用同一份源码和同一个二进制，共同执行以下步骤：

- 创建 1 个长度为 8、batch 为 1 的 C2C FFT plan；
- 申请 2 块相同大小的 GPU 输入/输出 buffer；
- 执行相同的初始化、10 轮预热、100 轮采样、同步和资源清理；
- 每轮均调用 `cudaDeviceSynchronize()`，采样前执行 `malloc_trim(0)`；
- 结束时均调用 1 次 `cufftDestroy` 和 2 次 `cudaFree`。

两组在测量窗内唯一的业务变量是：

```cpp
if (execute) {
    cufftExecC2C(plan, input, output, CUFFT_FORWARD);
}
```

`--control` 不执行上述调用；`--execute` 每轮执行 19 次。plan 数量、device buffer
数量和采样逻辑均未随轮次增加，所以测得的增长不能归因于项目反复创建 plan、反复申请
输入输出显存或测试数据容器扩容。

## 3. 实测数据

### 3.1 最小 A/B 复现

| 指标 | Control：不调用 execute | Execute：调用 `cufftExecC2C` |
| --- | ---: | ---: |
| FFT 长度 | 8 | 8 |
| plan 创建数 | 1 | 1 |
| device buffer 数 | 2 | 2 |
| 预热轮数 | 10 | 10 |
| 采样轮数 | 100 | 100 |
| 每轮 execute 次数 | 0 | 19 |
| 测量窗 execute 总次数 | 0 | 1,900 |
| CPU live heap 净增长 | **0 B** | **234,608 B** |
| CPU 上涨采样步数 | 0 | **100** |
| CPU 回落采样步数 | 0 | **0** |
| GPU 占用净增长 | **0 B** | **0 B** |
| 测试状态 | `stable` | `leak-reproduced` |
| 退出码 | 0 | 2（主动表示复现成功，不是程序崩溃） |

执行组的实测平均增量为：

```text
234,608 B / 1,900 次 = 123.478 B/次 cufftExecC2C
```

“100 次上涨、0 次回落”比单纯比较首尾值更重要：它说明增长不是偶发的一次性初始化或
上下波动的 allocator 噪声，而是在整个采样窗内保持单一方向。

### 3.2 早期探针与业务路径交叉验证

在固化供应商最小复现之前，固定 plan、每轮 19 次 FFT 的早期探针也得到相同趋势：

| 探针 | CPU live heap 净增长 | 上涨步数 | 回落步数 | 结果 |
| --- | ---: | ---: | ---: | --- |
| 8 个采样轮次，152 次 execute | 19,632 B | 7 | 0 | 持续增长 |
| 100 个采样轮次，1,900 次 execute | 232,576 B | 99 | 0 | `failed` |
| 增强后的最小 A/B 复现，1,900 次 execute | 234,608 B | 100 | 0 | `leak-reproduced` |

正式 DLFFT filtering 聚合入口还曾测得：

```text
[MEMORY][SUMMARY] label=filtering cpu_growth_bytes=136320
cpu_positive_steps=7 cpu_negative_steps=0 gpu_growth_bytes=0 status=failed
```

业务聚合入口的数据只作为交叉验证；归因应以第 3.1 节排除了项目 FFTInterface、算子和
Task 的最小 A/B 复现为准。

## 4. 线性累积的量级参考

下表按最小复现实测斜率 **123.478 B/次** 等比例计算，用于说明常驻进程长期运行的风险。
这些数值是线性趋势外推，不是对应调用次数的再次实测结果，不应用作性能或容量承诺。

| 累计 `cufftExecC2C` 次数 | 按实测斜率推算的 CPU live heap 增长 |
| ---: | ---: |
| 10,000 | 1.178 MiB |
| 100,000 | 11.776 MiB |
| 1,000,000 | 117.758 MiB |
| 10,000,000 | 1.150 GiB |

这类单次约百字节的泄漏在短测试中不显眼，但在服务进程、循环任务或高频 FFT 调用中会
日积月累，所以不能用较大的绝对阈值掩盖，也不能只运行少量轮次后判断“无泄漏”。

## 5. 链接和 API 归属证据

最小目标的 CMake 链接项为：

```cmake
target_link_libraries(dlfft_vendor_heap_repro PRIVATE dlfft curt dl)
```

远程实测形成了四层归属证据：

1. 链接命令包含 `-ldlfft`；
2. `readelf -d` 显示 ELF 依赖 `libdlfft.so`；
3. `ldd` 显示 `libdlfft.so => /zq500/sdk/lib/libdlfft.so`；
4. 程序在测量前通过 `dlsym`/`dladdr` 输出
   `cufftExecC2C object=/zq500/sdk/lib/libdlfft.so`。

`libdlfft.so` 是 ZQ500 SDK 提供的预编译动态库，不是本项目编译生成。复赛官方资料
`参赛者复赛资料/科学计算库实现列表.xlsx` 的 `FFT!A11:C11` 将 `cufftExecC2C` 列为
“FFT核心执行函数”，并说明该类接口由 `dlfftExec<...>` 实现。

## 6. 测量指标和结论边界

- CPU 使用 glibc `<malloc.h>` 的 `mallinfo2().uordblks`，表示当前进程 glibc allocator
  中仍处于已分配状态的总空间；它适合观察本例的 Host live heap 趋势，但不等于 RSS，
  也不能覆盖绕过 glibc 的全部分配方式。
- GPU 使用 ZQ500 CUDA Runtime 支持的 `cudaMemGetInfo()`，以 `total - free` 观察设备
  已占用显存趋势；它是设备级视图，因此测试时需要排除其他 GPU 进程干扰。
- 本结果证明增长边界已经收敛到当前 SDK 的 FFT execute 路径，但没有源代码权限，不能
  进一步断言是 `libdlfft.so` 内部哪一条私有 `malloc` 未释放。
- 实测每轮都同步 GPU，控制组稳定，执行组全窗单调增长；因此不能把 234,608 B 解释为
  异步任务尚未结束或测量 API 自身的固定初始化开销。
- GPU 净增长为 0 只代表当前最小用例和采样环境未发现显存持续增长，不应外推为所有尺寸、
  并发和异常路径下的绝对证明。

## 7. 当前工程处理方式

项目保留两个显式互斥的 FFT 后端：

```text
-DUSE_DLFFT=ON  -DUSE_THRUST=OFF  # ZQ500 官方 DLFFT，存在上述 CPU 慢泄漏限制
-DUSE_DLFFT=OFF -DUSE_THRUST=ON   # 项目自研 GPU FFT，正式无泄漏验收使用该后端
```

目前 SDK 清单只提供 `cufftDestroy` 作为 plan 级释放 API，没有可供项目补调的
`dlfftFinalize` 或 `cufftCleanup`。重建/销毁 plan、按官方 sample 使用 stream、关闭
`DLPTI_AUTO_LOAD` 等变体仍出现线性增长；context reset 也无法安全嵌入同进程重复任务。

若必须使用当前 DLFFT，可将一次完整工作负载放入短生命周期 worker，让进程退出兜底回收；
这只能隔离泄漏对常驻父进程的影响，不能宣称 `libdlfft.so` 的同进程泄漏已经修复。要求
同进程长期重复调用时，根本修复仍需由 ZQ500 SDK 供应商完成。

## 8. 数据来源

- [`docs/development/DLFFT_VENDOR_HEAP_REPRO_EVIDENCE_20260717.md`](../../../../docs/development/DLFFT_VENDOR_HEAP_REPRO_EVIDENCE_20260717.md)：最小复现、链接归属、测量 API 与正式 A/B 数据。
- [`docs/development/MEMORY_STABILITY_VALIDATION_20260716.md`](../../../../docs/development/MEMORY_STABILITY_VALIDATION_20260716.md)：早期探针、filtering 业务路径、清理方案复核与定位历史。
- [`test_all/operators/dlfft_vendor_heap_repro.cu`](../../../../test_all/operators/dlfft_vendor_heap_repro.cu)：最小复现源代码。
