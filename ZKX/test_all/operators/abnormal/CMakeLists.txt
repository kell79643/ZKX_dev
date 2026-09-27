cmake_minimum_required(VERSION 3.16)
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
include("${REPO_ROOT}/cmake/ZQ500Build.cmake")
project(abnormal_input_operator_abnormal LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(abnormal_input_operator_abnormal operator_abnormal_runner.cpp)
find_library(ABNORMAL_INPUT_DLFFT_LIBRARY dlfft HINTS "${ZQ500_SDK_DIR}/lib")
if(NOT ABNORMAL_INPUT_DLFFT_LIBRARY)
    message(FATAL_ERROR "abnormal_input_operator_abnormal requires ZQ500 dlfft")
endif()
target_include_directories(abnormal_input_operator_abnormal PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${REPO_ROOT}/cusignal_cpp/include"
    "${ZQ500_SDK_DIR}/include")
target_link_directories(abnormal_input_operator_abnormal PRIVATE "${ZQ500_SDK_DIR}/lib")
target_link_libraries(abnormal_input_operator_abnormal PRIVATE curt "${ABNORMAL_INPUT_DLFFT_LIBRARY}")
