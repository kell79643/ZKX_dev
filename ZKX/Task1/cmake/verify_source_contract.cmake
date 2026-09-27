# Task1 配置期静态门禁；不把缺少运行入口的检查登记为 CTest。
get_filename_component(TASK1_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(TASK1_REQUIRED_FILES
    pipeline/include/task1/task1_config.h pipeline/include/task1/task1_entry.h
    pipeline/include/task1/task1_common.h pipeline/src/task1_common.cu
    pipeline/include/task1/task1_runner.h pipeline/src/task1_runner.cu
    pipeline/include/task1/task1_abnormal_validation.h
    pipeline/apps/pipeline/main.cu
    visualization/cpp/pipeline_visualization_export.h visualization/cpp/pipeline_visualization_export.cpp
    accuracy/accuracy_types.h accuracy/accuracy_utils.h accuracy/accuracy_utils.cpp
    accuracy/validation/step1_validation.h accuracy/validation/step1_validation.cu
    accuracy/validation/step2_validation.h accuracy/validation/step2_validation.cu
    accuracy/validation/step3_validation.h accuracy/validation/step3_validation.cu
    accuracy/validation/step4_validation.h accuracy/validation/step4_validation.cu
    accuracy/validation/step5_validation.h accuracy/validation/step5_validation.cu
    accuracy/comparison/step1_comparison.h accuracy/comparison/step1_comparison.cpp
    accuracy/comparison/step2_comparison.h accuracy/comparison/step2_comparison.cpp
    accuracy/comparison/step3_comparison.h accuracy/comparison/step3_comparison.cpp
    accuracy/comparison/step4_comparison.h accuracy/comparison/step4_comparison.cpp
    accuracy/comparison/step5_comparison.h accuracy/comparison/step5_comparison.cpp
    pipeline/include/task1/steps/step1.h pipeline/src/steps/step1/step1.cu pipeline/apps/steps/step1/main.cu
    pipeline/include/task1/steps/step2.h pipeline/src/steps/step2/step2.cu pipeline/apps/steps/step2/main.cu
    pipeline/include/task1/steps/step3.h pipeline/src/steps/step3/step3.cu pipeline/apps/steps/step3/main.cu
    pipeline/include/task1/steps/step4.h pipeline/src/steps/step4/step4.cu pipeline/apps/steps/step4/main.cu
    pipeline/include/task1/steps/step5.h pipeline/src/steps/step5/step5.cu pipeline/apps/steps/step5/main.cu
    CMakeLists.txt)
foreach(TASK1_REQUIRED_FILE IN LISTS TASK1_REQUIRED_FILES)
    if(NOT EXISTS "${TASK1_SOURCE_ROOT}/${TASK1_REQUIRED_FILE}")
        message(FATAL_ERROR "Task1 required source is missing: ${TASK1_REQUIRED_FILE}")
    endif()
endforeach()

foreach(TASK1_STEP RANGE 1 5)
    set(TASK1_VALIDATION_FILE
        "${TASK1_SOURCE_ROOT}/accuracy/validation/step${TASK1_STEP}_validation.cu")
    set(TASK1_COMPARISON_FILE
        "${TASK1_SOURCE_ROOT}/accuracy/comparison/step${TASK1_STEP}_comparison.cpp")
    file(READ "${TASK1_VALIDATION_FILE}" TASK1_VALIDATION_TEXT)
    file(READ "${TASK1_COMPARISON_FILE}" TASK1_COMPARISON_TEXT)
    if(TASK1_VALIDATION_TEXT MATCHES
       "max_abs_error|relative_l2_error|relative_error|compare_step[1-5]_accuracy")
        message(FATAL_ERROR
            "Task1 validation layer computes accuracy or pass/fail: step${TASK1_STEP}")
    endif()
    if(TASK1_COMPARISON_TEXT MATCHES
       "DeviceArray|cuda[A-Z]|__global__|_device[<(]")
        message(FATAL_ERROR
            "Task1 comparison layer contains a GPU path: step${TASK1_STEP}")
    endif()
    if(NOT TASK1_COMPARISON_TEXT MATCHES "max_abs_error" OR
       NOT TASK1_COMPARISON_TEXT MATCHES "mean_squared_error" OR
       NOT TASK1_COMPARISON_TEXT MATCHES "relative_l2_error" OR
       NOT TASK1_COMPARISON_TEXT MATCHES "relative_linf_error" OR
       NOT TASK1_COMPARISON_TEXT MATCHES "relative_error[ ]*<=[ ]*evidence[.]relative_tolerance")
        message(FATAL_ERROR
            "Task1 comparison must retain MSE, RMSE, relative L2 and relative L_inf: step${TASK1_STEP}")
    endif()
endforeach()

set(TASK1_CPU_REFERENCE_CONTRACTS
    "accuracy/validation/step1_validation.cu|waveform_cpu"
    "accuracy/validation/step1_validation.cu|echo_cpu"
    "accuracy/validation/step2_validation.cu|pulse_compression_typed_cpu<float>"
    "accuracy/validation/step3_validation.cu|pulse_doppler_typed_cpu<float>"
    "accuracy/validation/step4_validation.cu|cfar_alpha_typed_cpu<float>"
    "accuracy/validation/step4_validation.cu|ca_cfar_typed_cpu<float>"
    "accuracy/validation/step5_validation.cu|ambgfun_typed_cpu<float>")
foreach(TASK1_CPU_REFERENCE_CONTRACT IN LISTS TASK1_CPU_REFERENCE_CONTRACTS)
    string(REPLACE "|" ";" TASK1_CPU_REFERENCE_PARTS
        "${TASK1_CPU_REFERENCE_CONTRACT}")
    list(GET TASK1_CPU_REFERENCE_PARTS 0 TASK1_CPU_REFERENCE_FILE)
    list(GET TASK1_CPU_REFERENCE_PARTS 1 TASK1_CPU_REFERENCE_TOKEN)
    file(READ "${TASK1_SOURCE_ROOT}/${TASK1_CPU_REFERENCE_FILE}"
        TASK1_CPU_REFERENCE_TEXT)
    string(FIND "${TASK1_CPU_REFERENCE_TEXT}" "${TASK1_CPU_REFERENCE_TOKEN}"
        TASK1_CPU_REFERENCE_POSITION)
    if(TASK1_CPU_REFERENCE_POSITION EQUAL -1)
        message(FATAL_ERROR
            "Task1 corresponding CPU reference is missing: ${TASK1_CPU_REFERENCE_FILE} -> ${TASK1_CPU_REFERENCE_TOKEN}")
    endif()
endforeach()

set(TASK1_FORBIDDEN_LEGACY_FILES
    Makefile signal_simulator_cuda.h signal_simulator_cuda.cu
    step1/test_step1.cu step2/test_step2.cu step3/test_step3.cu
    step3/task1_cfar_device.h step4/test_step4.cu step4/task1_ambgfun_device.h)
foreach(TASK1_FORBIDDEN_FILE IN LISTS TASK1_FORBIDDEN_LEGACY_FILES)
    if(EXISTS "${TASK1_SOURCE_ROOT}/${TASK1_FORBIDDEN_FILE}")
        message(FATAL_ERROR "Task1 legacy implementation is still active: ${TASK1_FORBIDDEN_FILE}")
    endif()
endforeach()
if(EXISTS "${TASK1_SOURCE_ROOT}/task1_cpu" OR
   EXISTS "${TASK1_SOURCE_ROOT}/Task1_targetwave_form")
    message(FATAL_ERROR
        "Task1 obsolete CPU/target-waveform source tree must not re-enter the active delivery")
endif()

set(TASK1_FORMAL_API_FILES
    "pipeline/src/steps/step2/step2.cu|pulse_compression_device<float>"
    "pipeline/src/steps/step3/step3.cu|pulse_doppler_device<float>"
    "pipeline/src/steps/step4/step4.cu|cfar_alpha_device<float>"
    "pipeline/src/steps/step4/step4.cu|ca_cfar_device<float>"
    "pipeline/src/steps/step5/step5.cu|ambgfun_device<float>")
foreach(TASK1_API_CONTRACT IN LISTS TASK1_FORMAL_API_FILES)
    string(REPLACE "|" ";" TASK1_API_PARTS "${TASK1_API_CONTRACT}")
    list(GET TASK1_API_PARTS 0 TASK1_API_FILE)
    list(GET TASK1_API_PARTS 1 TASK1_API_TOKEN)
    file(READ "${TASK1_SOURCE_ROOT}/${TASK1_API_FILE}" TASK1_API_TEXT)
    string(FIND "${TASK1_API_TEXT}" "${TASK1_API_TOKEN}" TASK1_API_POSITION)
    if(TASK1_API_POSITION EQUAL -1)
        message(FATAL_ERROR
            "Task1 formal API mapping is missing: ${TASK1_API_FILE} -> ${TASK1_API_TOKEN}")
    endif()
endforeach()

set(TASK1_ACTIVE_IMPLEMENTATION_TEXT "")
foreach(TASK1_SOURCE_FILE IN LISTS TASK1_REQUIRED_FILES)
    if(TASK1_SOURCE_FILE MATCHES "[.](cu|h|cpp)$")
        file(READ "${TASK1_SOURCE_ROOT}/${TASK1_SOURCE_FILE}" TASK1_SOURCE_TEXT)
        string(APPEND TASK1_ACTIVE_IMPLEMENTATION_TEXT "\n${TASK1_SOURCE_TEXT}")
    endif()
endforeach()
if(TASK1_ACTIVE_IMPLEMENTATION_TEXT MATCHES
   "cuDoubleComplex|cuFloatComplex|cufft|curand|task1_cfar_device|task1_ambgfun_device")
    message(FATAL_ERROR "Task1 active source contains an NVIDIA-only or private legacy path")
endif()
if(TASK1_ACTIVE_IMPLEMENTATION_TEXT MATCHES "#[ \t]*include[^\n]*[.](cpp|cu)[\"<]")
    message(FATAL_ERROR "Task1 active source directly includes an implementation file")
endif()

message(STATUS
    "[TASK1][CONTRACT] official_steps=5 formal_apis=5 accuracy_validation_files=5 accuracy_comparison_files=5 comparison_gpu_paths=0 cpu_references=5 relative_error_judgements=5 absolute_error_retained=5 legacy_paths=0 nvidia_only_path=0 status=verified")
