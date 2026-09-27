#pragma once

#include "accuracy/accuracy_types.h"

namespace task2::accuracy {
AccuracyEvidence compare_step5_accuracy(
    const PipelineState& state, const Step5ValidationData& validation);
}
