#pragma once

#include "accuracy/accuracy_types.h"

namespace task2::accuracy {
AccuracyEvidence compare_step1_accuracy(
    const PipelineState& state, const Step1ValidationData& validation);
}
