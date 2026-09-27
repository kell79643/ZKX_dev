cmake_minimum_required(VERSION 3.16)
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
include("${REPO_ROOT}/cmake/ZQ500Build.cmake")
project(abnormal_input_task1_abnormal LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

zq500_require_explicit_fft_backend()
add_executable(task1_task_abnormal task1_abnormal_runner.cpp)
configure_zq500_target(task1_task_abnormal)
target_include_directories(task1_task_abnormal PRIVATE
    "${REPO_ROOT}/Task1/pipeline/include")
