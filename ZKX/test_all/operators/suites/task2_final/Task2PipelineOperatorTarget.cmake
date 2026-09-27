set(TASK2_PIPELINE_SUPPORT_DIR "${CMAKE_CURRENT_LIST_DIR}")
function(operator_case_add_task2_pipeline_target target operator_name)
  set(sources
    "${TASK2_PIPELINE_SUPPORT_DIR}/task2_final_probe.cu"
    "${REPO_ROOT}/test_all/operators/shared/windows/sha256.cpp"
    "${REPO_ROOT}/test_all/support/metrics/accuracy.cpp"
    "${REPO_ROOT}/test_all/support/metrics/memory.cpp"
    "${REPO_ROOT}/cusignal_cpp/src/operators/bsplines/bsplines_typed.cpp"
    "${REPO_ROOT}/cusignal_cpp/src/operators/bsplines/bsplines_typed.cu"
    "${REPO_ROOT}/cusignal_cpp/src/operators/peak_finding/peak_finding_typed.cpp"
    "${REPO_ROOT}/cusignal_cpp/src/operators/peak_finding/peak_finding_typed.cu"
    "${REPO_ROOT}/cusignal_cpp/src/operators/estimation/estimation_typed.cpp"
    "${REPO_ROOT}/cusignal_cpp/src/operators/estimation/estimation_typed.cu")
  set_source_files_properties(${sources} PROPERTIES LANGUAGE CXX)
  add_executable(${target} ${sources})
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_compile_definitions(${target} PRIVATE ZKX_TASK2_PIPELINE_OPERATOR="${operator_name}")
  target_include_directories(${target} PRIVATE
    "${REPO_ROOT}/test_all/operators/shared/windows"
    "${REPO_ROOT}/test_all/support/metrics"
    "${REPO_ROOT}/cusignal_cpp/include" "${REPO_ROOT}/cusignal_cpp/src/runtime" "${ZQ500_SDK_DIR}/include")
  target_link_directories(${target} PRIVATE "${ZQ500_SDK_DIR}/lib")
  target_link_libraries(${target} PRIVATE curt)
  target_compile_options(${target} PRIVATE ${ZQ500_GPU_OPTIONS})
  target_link_options(${target} PRIVATE -Wl,--gc-sections)
endfunction()
