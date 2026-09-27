cmake_minimum_required(VERSION 3.16)
include("${CMAKE_CURRENT_LIST_DIR}/../../../shared/windows/WindowOperatorTarget.cmake")
project(remaining_operator_case_windows_chebwin LANGUAGES CXX)
zkx_add_remaining_operator_case_window_operator(remaining_operator_case_windows_chebwin
    "${CMAKE_CURRENT_LIST_DIR}/operator_entry.cu"
    FFT_BACKEND)
