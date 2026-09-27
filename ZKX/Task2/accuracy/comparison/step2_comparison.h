#pragma once

#include "accuracy/accuracy_types.h"

namespace task2::accuracy {
AccuracyEvidence compare_step2_accuracy(
    const PipelineState& state, const Step2ValidationData& validation);
}
