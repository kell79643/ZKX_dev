#pragma once

#include "accuracy/accuracy_types.h"

namespace task2::accuracy {
Step5ValidationData collect_step5_cpu_validation(const PipelineState& state);
Step5ValidationData collect_step5_cuda_validation(const PipelineState& state);
}
