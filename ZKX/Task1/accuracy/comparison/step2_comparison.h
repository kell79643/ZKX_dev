#pragma once

#include "accuracy/accuracy_types.h"

namespace task1::accuracy {

AccuracyEvidence compare_step2_accuracy(
    const PipelineState& state, const Step2ValidationData& validation);

}  // namespace task1::accuracy
