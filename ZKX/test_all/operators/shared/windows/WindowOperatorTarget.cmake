get_filename_component(ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
set(ZKX_REMAINING_OPERATOR_CASE_WINDOW_SHARED_DIR
    "${CMAKE_CURRENT_LIST_DIR}")
include("${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/cmake/ZQ500Build.cmake")

function(zkx_add_remaining_operator_case_window_operator target entry_source)
    set(cuda_sources
        "${entry_source}"
        "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/cusignal_cpp/src/operators/windows/windows_typed.cpp"
        "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/cusignal_cpp/src/operators/windows/windows_typed.cu"
        "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/cusignal_cpp/src/runtime/fp64_storage_finalize.cu")

    set(window_uses_fft FALSE)
    if(ARGC GREATER 2)
        if(NOT ARGV2 STREQUAL "FFT_BACKEND" OR ARGC GREATER 3)
            message(FATAL_ERROR
                "zkx_add_remaining_operator_case_window_operator accepts only the optional FFT_BACKEND marker")
        endif()
        list(APPEND cuda_sources
            "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/cusignal_cpp/src/backends/fft/fft_interface.cpp"
            "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/cusignal_cpp/src/backends/fft/fft_interface_cpu.cpp")
        set(window_uses_fft TRUE)
    endif()
    set(common_sources
        "${ZKX_REMAINING_OPERATOR_CASE_WINDOW_SHARED_DIR}/sha256.cpp"
        "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/test_all/support/config/json_value.cpp"
        "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/test_all/support/metrics/identity.cpp"
        "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/test_all/support/metrics/statistics.cpp"
        "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/test_all/support/metrics/accuracy.cpp"
        "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/test_all/support/metrics/memory.cpp"
        "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/test_all/support/metrics/csv_writer.cpp"
        # 剩余算子 case 的终端表与 CSV 共用公共渲染器；保留为独立目标源码。
        "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/test_all/support/metrics/terminal_table.cpp")

    set_source_files_properties(${cuda_sources} PROPERTIES LANGUAGE CXX)
    add_executable(${target} ${cuda_sources} ${common_sources})
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_options(${target} PRIVATE ${ZQ500_GPU_OPTIONS})
    target_include_directories(${target} PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}"
        "${ZKX_REMAINING_OPERATOR_CASE_WINDOW_SHARED_DIR}"
        "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/cusignal_cpp/include"
        "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/cusignal_cpp/src/runtime"
        "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/test_all/support/config"
        "${ZKX_REMAINING_OPERATOR_CASE_REPO_ROOT}/test_all/support/metrics"
        "${ZQ500_SDK_DIR}/include")
    target_link_directories(${target} PRIVATE "${ZQ500_SDK_DIR}/lib")
    target_link_libraries(${target} PRIVATE curt)
    target_link_options(${target} PRIVATE -Wl,--gc-sections)
    if(window_uses_fft)
        zq500_link_selected_fft(${target})
        target_compile_definitions(${target} PRIVATE
            CUSIGNAL_WINDOWS_FFT_CHEBWIN
            ZKX_REMAINING_OPERATOR_CASE_WINDOW_FFT_BACKEND)
    endif()
endfunction()
