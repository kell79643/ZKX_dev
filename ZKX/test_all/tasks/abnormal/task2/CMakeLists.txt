cmake_minimum_required(VERSION 3.16)
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
include("${REPO_ROOT}/cmake/ZQ500Build.cmake")
project(abnormal_input_task2_abnormal LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

zq500_require_explicit_fft_backend()
add_executable(task2_task_abnormal task2_abnormal_runner.cpp)
configure_zq500_target(task2_task_abnormal)
target_include_directories(task2_task_abnormal PRIVATE
    "${REPO_ROOT}/Task2/pipeline/include")
