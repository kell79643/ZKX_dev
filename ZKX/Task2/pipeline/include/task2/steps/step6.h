#pragma once

#include <task2/task2_common.h>

namespace task2 {

void prepare_step6_host_inputs(PipelineState& state);
StepEvidence run_step6_prepared(PipelineState& state);
StepEvidence run_step6(PipelineState& state);

}  // namespace task2
