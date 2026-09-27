cmake_minimum_required(VERSION 3.16)
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
include("${REPO_ROOT}/cmake/ZQ500Build.cmake")
project(task1_scaled_pipeline_stability LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_TASK1 OFF CACHE BOOL "" FORCE)
set(BUILD_TASK2 OFF CACHE BOOL "" FORCE)
add_subdirectory("${REPO_ROOT}/cusignal_cpp" cusignal_cpp)

set(TASK1_STABILITY_SOURCES
    app/stability_main.cu
    src/task1_common.cu
    src/steps/step1.cu src/steps/step2.cu src/steps/step3.cu
    src/steps/step4.cu src/steps/step5.cu
    src/accuracy/validation/step1_validation.cu
    src/accuracy/validation/step2_validation.cu
    src/accuracy/validation/step3_validation.cu
    src/accuracy/validation/step4_validation.cu
    src/accuracy/validation/step5_validation.cu
    "${REPO_ROOT}/test_all/support/config/json_value.cpp")
zq500_mark_cuda_sources(${TASK1_STABILITY_SOURCES})

function(add_task1_stability_scale target pulses samples)
    add_executable(${target} ${TASK1_STABILITY_SOURCES})
    configure_zq500_target(${target})
    target_include_directories(${target} PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/include"
        "${CMAKE_CURRENT_LIST_DIR}/include/accuracy"
        "${CMAKE_CURRENT_LIST_DIR}/include/accuracy/validation"
        "${REPO_ROOT}"
        "${REPO_ROOT}/cusignal_cpp/src/runtime")
    target_compile_definitions(${target} PRIVATE
        TASK1_STABILITY_NUM_PULSES=${pulses}
        TASK1_STABILITY_SAMPLES_PER_PULSE=${samples})
    target_link_libraries(${target} PRIVATE signal_lib fft_core)
endfunction()

add_task1_stability_scale(task1_stability_256 32 256)
add_task1_stability_scale(task1_stability_512 64 512)
add_task1_stability_scale(task1_stability_1024 128 1024)
