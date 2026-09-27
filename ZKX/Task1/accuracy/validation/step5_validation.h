#pragma once

#include "accuracy/accuracy_types.h"

namespace task1::accuracy {

Step5ValidationData generate_step5_cpu_reference(const PipelineState& state);
Step5ValidationData collect_step5_validation(const PipelineState& state);

}  // namespace task1::accuracy
