# 任务算子分类与覆盖状态图表说明

## 图表选择

1. 面向任务的算子分类树：使用“图”。
   这张图表达的是层级关系：项目目标算子 -> 任务一/任务二 -> 处理场景 -> 具体算子。表格不适合展示父子结构和任务归属。

2. 项目中所有算子覆盖状态：使用“图”，主体为覆盖矩阵。
   覆盖状态需要同时比较 22 个算子和多条验证链路，矩阵热图比普通表更容易看出缺失项、通过项和任务一/任务二的整体状态。右侧附加通过率柱状图和最大 MSE 量级条形图，用于说明一致性验证质量。

## 覆盖状态指标

覆盖矩阵使用 5 个二值指标：

- Python 参考：`cpu_vs_python` 行存在 Python 参考输出路径。
- CPU 输出：存在 CPU 项目输出路径。
- GPU 输出：存在 GPU 项目输出路径。
- CPU-Python：`cpu_vs_python` 的 `mse <= threshold`。
- GPU-CPU：`gpu_vs_cpu` 的 `mse <= threshold`。

综合覆盖率定义为以上 5 个指标全部满足的算子占比。误差质量使用每个算子的最大 MSE，并以 `log10(max_mse + eps)` 展示，同时标出 `1e-6` 阈值线。

## 数据来源

代码按顺序读取：

1. 当前目录下的 `operator_consistency_results.csv`
2. `/workspace/CONSISTENCY/consistency_figure/operator_consistency_results.csv`
3. `/workspace/CONSISTENCY/operator_consistency_results.csv`

输出目录为当前文件夹下的 `output`。

## 运行入口

在 MATLAB 中进入本目录后运行：

```matlab
generate_task_operator_coverage_figures
```

也可以分别运行：

```matlab
fig_task_operator_classification_tree
fig_operator_coverage_status
```
