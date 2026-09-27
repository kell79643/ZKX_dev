cmake_minimum_required(VERSION 3.16)
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
include("${REPO_ROOT}/cmake/ZQ500Build.cmake")
project(task2_scaled_pipeline_stability LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_TASK1 OFF CACHE BOOL "" FORCE)
set(BUILD_TASK2 OFF CACHE BOOL "" FORCE)
add_subdirectory("${REPO_ROOT}/cusignal_cpp" cusignal_cpp)

set(TASK2_STABILITY_SOURCES
    app/stability_main.cu
    src/task2_common.cu
    src/steps/step1.cu src/steps/step2.cu src/steps/step3.cu
    src/steps/step4.cu src/steps/step5.cu src/steps/step6.cu
    src/accuracy/validation/step1_validation.cpp
    src/accuracy/validation/step2_validation.cpp
    src/accuracy/validation/step3_validation.cpp
    src/accuracy/validation/step4_validation.cpp
    src/accuracy/validation/step5_cpu_validation.cpp
    src/accuracy/validation/step5_validation.cu
    src/accuracy/validation/step6_validation.cpp
    "${REPO_ROOT}/test_all/support/config/json_value.cpp")
zq500_mark_cuda_sources(${TASK2_STABILITY_SOURCES})

function(add_task2_stability_scale target signal_length)
    add_executable(${target} ${TASK2_STABILITY_SOURCES})
    configure_zq500_target(${target})
    target_include_directories(${target} PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/include"
        "${CMAKE_CURRENT_LIST_DIR}/include/accuracy"
        "${CMAKE_CURRENT_LIST_DIR}/include/accuracy/validation"
        "${REPO_ROOT}"
        "${REPO_ROOT}/cusignal_cpp/src/runtime")
    target_compile_definitions(${target} PRIVATE
        TASK2_STABILITY_SAMPLES=${signal_length})
    target_link_libraries(${target} PRIVATE signal_lib fft_core)
endfunction()

add_task2_stability_scale(task2_stability_4096 4096)
add_task2_stability_scale(task2_stability_16384 16384)
add_task2_stability_scale(task2_stability_65536 65536)
