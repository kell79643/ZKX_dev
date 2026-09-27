#pragma once

namespace task1 {

struct TaskConfig;
int run_task1_until(
    int stop_after, const char* evidence_id, const TaskConfig& config);

}  // namespace task1
