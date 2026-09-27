cmake_minimum_required(VERSION 3.16)

get_filename_component(ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)
include("${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/cmake/ZQ500Build.cmake")
project(remaining_operator_case_filtering_channelize_poly LANGUAGES CXX)

set(CHANNELIZE_POLY_SOURCES
    operator_probe.cu
    "${CMAKE_CURRENT_LIST_DIR}/../../../shared/windows/sha256.cpp"
    "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/test_all/support/metrics/accuracy.cpp"
    "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/test_all/support/metrics/memory.cpp"
    "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/cusignal_cpp/src/runtime/fp64_storage_finalize.cu"
    "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/cusignal_cpp/src/operators/filtering/filtering_typed.cpp"
    "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/cusignal_cpp/src/operators/filtering/filtering_typed.cu"
    "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/cusignal_cpp/src/operators/filter_design/filter_design_typed.cpp"
    "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/cusignal_cpp/src/operators/filter_design/filter_design_typed.cu"
    "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/cusignal_cpp/src/operators/windows/windows_typed.cpp"
    "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/cusignal_cpp/src/operators/windows/windows_typed.cu"
    "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/cusignal_cpp/src/backends/fft/fft_interface.cpp"
    "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/cusignal_cpp/src/backends/fft/fft_interface_cpu.cpp")

set_source_files_properties(${CHANNELIZE_POLY_SOURCES} PROPERTIES LANGUAGE CXX)
add_executable(remaining_operator_case_filtering_channelize_poly ${CHANNELIZE_POLY_SOURCES})
target_compile_features(remaining_operator_case_filtering_channelize_poly PRIVATE cxx_std_17)
target_include_directories(remaining_operator_case_filtering_channelize_poly PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../../../shared/windows"
    "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/cusignal_cpp/include"`r`n        "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/cusignal_cpp/src/runtime"
    "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/test_all/support/metrics"
    "${ZQ500_SDK_DIR}/include")
target_link_directories(remaining_operator_case_filtering_channelize_poly PRIVATE
    "${ZQ500_SDK_DIR}/lib")
zq500_link_selected_fft(remaining_operator_case_filtering_channelize_poly)
target_compile_definitions(remaining_operator_case_filtering_channelize_poly PRIVATE
    ZKX_REMAINING_OPERATOR_CASE_CHANNELIZE_FFT_BACKEND)
target_compile_options(remaining_operator_case_filtering_channelize_poly PRIVATE
    ${ZQ500_GPU_OPTIONS})
target_link_options(remaining_operator_case_filtering_channelize_poly PRIVATE -Wl,--gc-sections)
