#pragma once

#include "accuracy/accuracy_types.h"

namespace task1::accuracy {

Step1ValidationData generate_step1_cpu_reference(const TaskConfig& config);
Step1ValidationData collect_step1_validation(const PipelineState& state);

}  // namespace task1::accuracy
