set(OPERATOR_CASE_FILTER_SUPPORT_DIR "${CMAKE_CURRENT_LIST_DIR}")
function(operator_case_add_filter_target target operator_name)
  set(sources "${OPERATOR_CASE_FILTER_SUPPORT_DIR}/task2_filter_probe.cu"
    "${REPO_ROOT}/test_all/operators/shared/windows/sha256.cpp"
    "${REPO_ROOT}/test_all/support/metrics/accuracy.cpp" "${REPO_ROOT}/test_all/support/metrics/memory.cpp"
    "${REPO_ROOT}/cusignal_cpp/src/runtime/fp64_storage_finalize.cu"
    "${REPO_ROOT}/cusignal_cpp/src/operators/windows/windows_typed.cpp" "${REPO_ROOT}/cusignal_cpp/src/operators/windows/windows_typed.cu"
    "${REPO_ROOT}/cusignal_cpp/src/operators/filter_design/filter_design_typed.cpp" "${REPO_ROOT}/cusignal_cpp/src/operators/filter_design/filter_design_typed.cu"
    "${REPO_ROOT}/cusignal_cpp/src/operators/filtering/filtering_typed.cpp" "${REPO_ROOT}/cusignal_cpp/src/operators/filtering/filtering_typed.cu"
    "${REPO_ROOT}/cusignal_cpp/src/backends/fft/fft_interface.cpp" "${REPO_ROOT}/cusignal_cpp/src/backends/fft/fft_interface_cpu.cpp")
  set_source_files_properties(${sources} PROPERTIES LANGUAGE CXX)
  add_executable(${target} ${sources})
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_compile_definitions(${target} PRIVATE ZKX_OPERATOR_CASE_FILTER_OPERATOR="${operator_name}" USE_THRUST)
  target_include_directories(${target} PRIVATE "${REPO_ROOT}/test_all/operators/shared/windows" "${REPO_ROOT}/test_all/support/metrics" "${REPO_ROOT}/cusignal_cpp/include" "${REPO_ROOT}/cusignal_cpp/src/runtime" "${ZQ500_SDK_DIR}/include")
  target_link_directories(${target} PRIVATE "${ZQ500_SDK_DIR}/lib")
  zq500_link_fft_thrust(${target})
  target_compile_options(${target} PRIVATE ${ZQ500_GPU_OPTIONS})
  target_link_options(${target} PRIVATE -Wl,--gc-sections)
endfunction()
