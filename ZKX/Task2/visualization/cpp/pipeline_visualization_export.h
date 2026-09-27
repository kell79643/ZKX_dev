#pragma once

#include <task2/task2_common.h>

namespace task2 {

// 当前正式 pipeline 的可视化数据出口：设置 ZKX_VISUALIZATION_DIR 时，将已经产生的
// Host 中间结果导出为 CSV。
// Python 可视化只读取这些实际输出，不重新执行波形、滤波、特征或估计算法。
void export_visualization_data(const PipelineState& state, int completed_steps);

}  // namespace task2
