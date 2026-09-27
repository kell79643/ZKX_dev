# 数值敏感算子高精度参考最终结果

## 结论

2026-07-17 在 ZQ500 上完成最终深度验证：7 个代表算子、20 个场景、FP32/FP16 共
40 条精度记录全部通过，2 条 Task1 峰值业务记录全部通过。typed CPU/GPU 的实现一致性
继续成立；高精度 CPU 基准进一步揭示，FP16 的误差会随长序列、递归和敏感输入明显放大。

因此项目精度决策是：

- Task1 正式链路继续使用 FP32；
- Task2 正式链路继续使用 FP32，其直接相关的 `firfilter`、`correlate` 已获得深度证据；
- FP16 只能在非敏感、非长累加路径中，经具体业务指标复测后局部采用；
- 不把这组高精度 CPU 运行耗时计入性能加速比。

完整最终报告、结构化数据和答辩材料见
[高精度 CPU 最终证据目录](../04_原始证据/operators/operator_high_precision_final_20260717/README.md)。

## 两个 CPU 对比的分工

```text
实现一致性误差 = GPU(dtype X) - typed CPU reference(dtype X)
高精度基准总误差 = GPU(dtype X) - long-double reference(共同原始高精度输入)
```

前者判断 GPU 实现是否符合当前 dtype 契约；后者判断选择该 dtype 后相对高精度基准总共
损失多少。日志中的 `diagnostic_compute_*` 只是把已编码输入交给同一个高精度公式复算，
用于分析误差来源，不是第三套 CPU reference。

## 深度场景的最大观测值

下表取每个算子/dtype 在本轮各场景中的最大高精度基准绝对误差：

| 路径 | FP32 | FP16 | 最大值对应场景 |
|---|---:|---:|---|
| `correlate` | `5.87e-6` | `1.45e-2` | long / sensitive |
| `sosfilt` | `1.10e-6` | `4.29e-3` | sensitive |
| `lombscargle` | `2.84e-7` | `2.57e-3` | long |
| `firfilter` | `1.93e-7` | `8.94e-4` | sensitive |
| `upfirdn` | `7.26e-8` | `5.32e-4` | sensitive |
| `pulse_compression` | `1.64e-6` | `1.15e-3` | long / sensitive |
| `pulse_doppler` | `1.49e-6` | `1.19e-3` | Task1 Hamming |
| Task1 两算子链 | `2.61e-5` | `4.11e-3` | 32×256 任务场景 |

Task1 场景中，FP32、FP16 的 GPU、typed CPU 和高精度参考都定位到 Doppler bin 5、
range bin 80。这个结果说明当前单目标固定噪声场景的峰值业务语义未被改变，但不能据此
推广为 FP16 在弱目标、多目标、近阈值 CFAR 或更长积累条件下均安全。

## 为什么到这里可以停止扩展

继续给 53 个算子机械补 `long double` 实现的边际价值已经很低。本轮已覆盖四类主要风险：
长乘加、递归状态、非均匀谱估计、重采样/匹配滤波，并补齐 Task1 的脉冲多普勒和真实规模
两算子传播。结果已经足以支持当前 FP32/FP16 选择，同时 Task2 直接使用的 FIR 和相关路径
也已覆盖。

满足以下停止条件：

1. 代表性风险类型已覆盖；
2. baseline、long、sensitive 和任务参数均有证据；
3. 误差已传播到真实 Task1 业务指标；
4. 结论能改变或确认 dtype 决策；
5. 日志、CSV、源码指纹和边界说明可复核。

后续仅在新增低精度正式路径、业务容差变化、发现新的数值失稳、或整数算子明确量化语义时，
按风险增加高精度参考，不再以“覆盖全部算子”为目标。
