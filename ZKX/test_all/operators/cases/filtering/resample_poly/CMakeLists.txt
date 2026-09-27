cmake_minimum_required(VERSION 3.16)
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)
include("${REPO_ROOT}/cmake/ZQ500Build.cmake")
project(remaining_operator_case_filtering_resample_poly LANGUAGES CXX)
set(SOURCES operator_probe.cu "${CMAKE_CURRENT_LIST_DIR}/../../../shared/windows/sha256.cpp"
 "${REPO_ROOT}/test_all/support/metrics/accuracy.cpp" "${REPO_ROOT}/test_all/support/metrics/memory.cpp"
 "${REPO_ROOT}/cusignal_cpp/src/runtime/fp64_storage_finalize.cu"
 "${REPO_ROOT}/cusignal_cpp/src/operators/filtering/filtering_typed.cpp" "${REPO_ROOT}/cusignal_cpp/src/operators/filtering/filtering_typed.cu"
 "${REPO_ROOT}/cusignal_cpp/src/operators/filter_design/filter_design_typed.cpp" "${REPO_ROOT}/cusignal_cpp/src/operators/filter_design/filter_design_typed.cu"
 "${REPO_ROOT}/cusignal_cpp/src/operators/windows/windows_typed.cpp" "${REPO_ROOT}/cusignal_cpp/src/operators/windows/windows_typed.cu"
 # 只闭合共享filtering/filter_design翻译单元；resample_poly自身不调用FFT。
 "${REPO_ROOT}/cusignal_cpp/src/backends/fft/fft_interface.cpp" "${REPO_ROOT}/cusignal_cpp/src/backends/fft/fft_interface_cpu.cpp")
set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
add_executable(remaining_operator_case_filtering_resample_poly ${SOURCES})
target_compile_features(remaining_operator_case_filtering_resample_poly PRIVATE cxx_std_17)
target_include_directories(remaining_operator_case_filtering_resample_poly PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../../../shared/windows" "${REPO_ROOT}/cusignal_cpp/include" "${REPO_ROOT}/cusignal_cpp/src/runtime" "${REPO_ROOT}/test_all/support/metrics" "${ZQ500_SDK_DIR}/include")
target_link_directories(remaining_operator_case_filtering_resample_poly PRIVATE "${ZQ500_SDK_DIR}/lib")
zq500_link_fft_thrust(remaining_operator_case_filtering_resample_poly)
target_compile_definitions(remaining_operator_case_filtering_resample_poly PRIVATE USE_THRUST)
target_compile_options(remaining_operator_case_filtering_resample_poly PRIVATE ${ZQ500_GPU_OPTIONS})
target_link_options(remaining_operator_case_filtering_resample_poly PRIVATE -Wl,--gc-sections)
