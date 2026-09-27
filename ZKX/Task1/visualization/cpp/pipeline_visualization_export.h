#pragma once

#include <task1/task1_common.h>

namespace task1 {

// 当前正式 pipeline 的可视化数据出口：设置 ZKX_VISUALIZATION_DIR 时，将已经产生的
// Host 中间结果导出为 CSV。
// 导出发生在步骤计时和精度比较完成之后；未设置环境变量时不产生文件。
void export_visualization_data(const PipelineState& state, int completed_steps);

}  // namespace task1
