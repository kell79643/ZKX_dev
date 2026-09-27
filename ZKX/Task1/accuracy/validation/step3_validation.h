#pragma once

#include "accuracy/accuracy_types.h"

namespace task1::accuracy {

Step3ValidationData generate_step3_cpu_reference(const PipelineState& state);
Step3ValidationData collect_step3_validation(const PipelineState& state);

}  // namespace task1::accuracy
