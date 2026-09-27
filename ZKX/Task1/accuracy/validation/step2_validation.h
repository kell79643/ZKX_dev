#pragma once

#include "accuracy/accuracy_types.h"

namespace task1::accuracy {

Step2ValidationData generate_step2_cpu_reference(const PipelineState& state);
Step2ValidationData collect_step2_validation(const PipelineState& state);

}  // namespace task1::accuracy
