cmake_minimum_required(VERSION 3.16)
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)
include("${REPO_ROOT}/cmake/ZQ500Build.cmake")
project(remaining_operator_case_convolution_correlate2d LANGUAGES CXX)
set(SOURCES
    operator_probe.cu
    "${CMAKE_CURRENT_LIST_DIR}/../../../shared/windows/sha256.cpp"
    "${REPO_ROOT}/test_all/support/metrics/accuracy.cpp"
    "${REPO_ROOT}/test_all/support/metrics/memory.cpp"
    "${REPO_ROOT}/cusignal_cpp/src/operators/convolution/convolution_typed.cpp"
    "${REPO_ROOT}/cusignal_cpp/src/operators/convolution/convolution_typed.cu"
    # 仅闭合共享 convolution 翻译单元里一维 FFT 显式实例的链接符号；
    # correlate2d 本身固定调用 boundary-aware direct kernel，证据 backend=not_applicable。
    "${REPO_ROOT}/cusignal_cpp/src/backends/fft/fft_interface.cpp"
    "${REPO_ROOT}/cusignal_cpp/src/backends/fft/fft_interface_cpu.cpp")
set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
add_executable(remaining_operator_case_convolution_correlate2d ${SOURCES})
target_compile_features(remaining_operator_case_convolution_correlate2d PRIVATE cxx_std_17)
target_include_directories(remaining_operator_case_convolution_correlate2d PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../../../shared/windows" "${REPO_ROOT}/cusignal_cpp/include" "${REPO_ROOT}/cusignal_cpp/src/runtime"
    "${REPO_ROOT}/test_all/support/metrics" "${ZQ500_SDK_DIR}/include")
target_link_directories(remaining_operator_case_convolution_correlate2d PRIVATE "${ZQ500_SDK_DIR}/lib")
zq500_link_fft_thrust(remaining_operator_case_convolution_correlate2d)
target_compile_definitions(remaining_operator_case_convolution_correlate2d PRIVATE USE_THRUST)
target_compile_options(remaining_operator_case_convolution_correlate2d PRIVATE ${ZQ500_GPU_OPTIONS})
target_link_options(remaining_operator_case_convolution_correlate2d PRIVATE -Wl,--gc-sections)
