#pragma once

#include "accuracy/accuracy_types.h"

namespace task2::accuracy {
AccuracyEvidence compare_step3_accuracy(
    const PipelineState& state, const Step3ValidationData& validation);
}
