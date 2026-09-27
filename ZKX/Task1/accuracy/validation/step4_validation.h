#pragma once

#include "accuracy/accuracy_types.h"

namespace task1::accuracy {

Step4ValidationData generate_step4_cpu_reference(const PipelineState& state);
Step4ValidationData collect_step4_validation(const PipelineState& state);
std::vector<cusignal::CfarDetection> generate_step4_same_power_detections_cpu(
    const PipelineState& state);

}  // namespace task1::accuracy
