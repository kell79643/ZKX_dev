#pragma once

#include "accuracy/accuracy_types.h"

namespace task2::accuracy {
Step4ValidationData collect_step4_validation(const PipelineState& state);
Step4ValidationData collect_step4_validation_with_precomputed_correlation(
    const PipelineState& state, std::vector<float> correlation_cpu);
}
