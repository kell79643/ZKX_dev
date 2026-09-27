# ZQ500 CMake 构建入口（基于 /zq500/sdk/samples/cuda/vectorAdd/CMakeLists.txt）。
#
# 官方模式：CMake 3.16 兼容、project(... LANGUAGES CXX)、dlicc 编译器、.cu 文件
# 作为 CXX 源码、链接 curt，并由 -x cuda 与官方样例中的架构参数编译 GPU 源码。

cmake_minimum_required(VERSION 3.16)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/ZQ500Build.cmake")

project(cusignal_cpp LANGUAGES CXX)

# 53x5 契约是正式源码、测试期望和完成计数的唯一数据源。该门禁保持
# CMake 3.16 兼容；gpu_02 当前实测 3.22.1，最终容器验证仍放在交付末步。
set(CONTRACT_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/contracts/verify_operator_contracts.cmake")

set(OPERATOR_LIBRARY_EVIDENCE_GIT_COMMIT "unbound" CACHE STRING
    "Git commit bound to operator library runtime evidence")
set(OPERATOR_LIBRARY_EVIDENCE_TEST_TIME "unbound" CACHE STRING
    "ISO-8601 test time bound to operator library runtime evidence")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

zq500_require_explicit_fft_backend()

if(USE_DLFFT)
    set(FFT_IMPL src/backends/fft/fft_dlfft.cu)
    set(FFT_BACKEND_AUDIT_SOURCES src/backends/fft/fft_cufft.cu)
else()
    # Thrust 后端由唯一共享目标 libzkx_fft_thrust.so 提供，不嵌入消费者源码列表。
    set(FFT_IMPL)
    set(FFT_BACKEND_AUDIT_SOURCES src/backends/fft/fft_thrust.cu)
endif()

# operator-type-coverage 正式 GPU 路径。业务 Module 仅使用统一的 *_typed.cu；此清单由下面的
# 配置期门禁保证不引入 FP64。
set(OPERATOR_TYPE_FORMAL_GPU_SOURCES
    ${FFT_IMPL}
    # dlfft 入口通过文本 include 复用共享封装；仅在该后端下纳入 FP64 扫描。
    ${FFT_BACKEND_AUDIT_SOURCES}
    src/runtime/operator_type_dispatch.cu
    src/runtime/fp64_storage_finalize.cu
    src/operators/bsplines/bsplines_typed.cu
    src/operators/convolution/convolution_typed.cu
    src/operators/demod/demod_typed.cu
    src/operators/estimation/estimation_typed.cu
    src/operators/filter_design/filter_design_typed.cu
    src/operators/filtering/filtering_typed.cu
    src/operators/peak_finding/peak_finding_typed.cu
    src/operators/radartools/radartools_typed.cu
    src/operators/spectral_analysis/spectral_analysis_typed.cu
    src/operators/waveforms/waveforms_typed.cu
    src/operators/wavelets/wavelets_typed.cu
    src/operators/windows/windows_typed.cu
)

# 正式 typed 源码直接包含的共享 kernel 头同样属于 operator-type-coverage GPU 主路径；纳入扫描，
# 防止把 FP64 实现藏进头文件而绕过仅扫描翻译单元的门禁。
set(OPERATOR_TYPE_FORMAL_GPU_HEADERS
    src/runtime/operator_compute_policy.h
    src/operators/bsplines/bsplines_kernels.cuh
    src/operators/convolution/convolution_kernels.cuh
    src/operators/demod/demod_kernels.cuh
    src/operators/estimation/estimation_kernels.cuh
    src/operators/filter_design/filter_design_kernels.cuh
    src/operators/filtering/filtering_kernels.cuh
    src/operators/peak_finding/peak_finding_kernels.cuh
    src/operators/radartools/radartools_kernels.cuh
    src/operators/spectral_analysis/spectral_analysis_kernels.cuh
    src/operators/waveforms/waveforms_kernels.cuh
    src/operators/wavelets/wavelets_kernels.cuh
    src/operators/windows/windows_kernels.cuh
)

foreach(OPERATOR_TYPE_GPU_SOURCE IN LISTS OPERATOR_TYPE_FORMAL_GPU_SOURCES OPERATOR_TYPE_FORMAL_GPU_HEADERS)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${OPERATOR_TYPE_GPU_SOURCE}" OPERATOR_TYPE_GPU_SOURCE_TEXT)
    if(OPERATOR_TYPE_GPU_SOURCE_TEXT MATCHES "host_output_finalize[.]h")
        message(FATAL_ERROR
            "operator-type-coverage GPU source includes host-only FP64 finalization: ${OPERATOR_TYPE_GPU_SOURCE}")
    endif()
    if(OPERATOR_TYPE_GPU_SOURCE_TEXT MATCHES "(^|[^A-Za-z0-9_])double([^A-Za-z0-9_]|$)" OR
       OPERATOR_TYPE_GPU_SOURCE_TEXT MATCHES "ComplexDouble" OR
       OPERATOR_TYPE_GPU_SOURCE_TEXT MATCHES "cuDoubleComplex")
        message(FATAL_ERROR
            "operator-type-coverage formal GPU source contains forbidden FP64 token: ${OPERATOR_TYPE_GPU_SOURCE}")
    endif()
endforeach()

set(SIGNAL_MODULES
    src/backends/fft/fft_interface.cpp
    src/backends/fft/fft_interface_cpu.cpp
    src/runtime/operator_type_dispatch.cu
    src/runtime/fp64_storage_finalize.cu
    src/operators/bsplines/bsplines_typed.cpp
    src/operators/bsplines/bsplines_typed.cu
    src/operators/convolution/convolution_typed.cpp
    src/operators/convolution/convolution_typed.cu
    src/operators/demod/demod_typed.cpp
    src/operators/demod/demod_typed.cu
    src/operators/estimation/estimation_typed.cpp
    src/operators/estimation/estimation_typed.cu
    src/operators/filter_design/filter_design_typed.cpp
    src/operators/filter_design/filter_design_typed.cu
    src/operators/filtering/filtering_typed.cpp
    src/operators/filtering/filtering_typed.cu
    src/operators/peak_finding/peak_finding_typed.cpp
    src/operators/peak_finding/peak_finding_typed.cu
    src/operators/radartools/radartools_typed.cpp
    src/operators/radartools/radartools_typed.cu
    src/operators/spectral_analysis/spectral_analysis_typed.cpp
    src/operators/spectral_analysis/spectral_analysis_typed.cu
    src/operators/waveforms/waveforms_typed.cpp
    src/operators/waveforms/waveforms_typed.cu
    src/operators/wavelets/wavelets_typed.cpp
    src/operators/wavelets/wavelets_typed.cu
    src/operators/windows/windows_typed.cpp
    src/operators/windows/windows_typed.cu
)

set(ALL_GPU_SOURCES
    ${FFT_IMPL}
    src/runtime/operator_type_dispatch.cu
    src/runtime/fp64_storage_finalize.cu
    src/operators/bsplines/bsplines_typed.cu
    src/operators/convolution/convolution_typed.cu
    src/operators/demod/demod_typed.cu
    src/operators/estimation/estimation_typed.cu
    src/operators/filter_design/filter_design_typed.cu
    src/operators/filtering/filtering_typed.cu
    src/operators/peak_finding/peak_finding_typed.cu
    src/operators/radartools/radartools_typed.cu
    src/operators/spectral_analysis/spectral_analysis_typed.cu
    src/operators/waveforms/waveforms_typed.cu
    src/operators/wavelets/wavelets_typed.cu
    src/operators/windows/windows_typed.cu
)
zq500_mark_cuda_sources(${ALL_GPU_SOURCES})

if(USE_DLFFT)
    add_library(fft_core OBJECT ${FFT_IMPL})
    configure_zq500_target(fft_core)

    add_library(fft_core_shared SHARED ${FFT_IMPL})
    configure_zq500_target(fft_core_shared)
    set_target_properties(fft_core_shared PROPERTIES
        OUTPUT_NAME "fft_core" POSITION_INDEPENDENT_CODE ON)
else()
    zq500_add_fft_thrust_shared()
    add_library(fft_core INTERFACE)
    target_link_libraries(fft_core INTERFACE zkx_fft_thrust)
    add_library(fft_core_shared ALIAS zkx_fft_thrust)
endif()

add_library(signal_lib OBJECT ${SIGNAL_MODULES})
configure_zq500_target(signal_lib)
target_include_directories(signal_lib PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime")
target_link_libraries(signal_lib PRIVATE fft_core)

set(PUBLIC_API_DOCS_VERIFY_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/validation/verify_public_api_docs.cmake")
add_custom_target(public_api_docs_gate
    COMMAND ${CMAKE_COMMAND}
        -DPUBLIC_API_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}
        -P ${PUBLIC_API_DOCS_VERIFY_SCRIPT}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    USES_TERMINAL
)

# 仅在全部源码、测试和证据闭环后执行；任何未批准的dtype契约都会失败。
add_custom_target(operator_contract_completion_gate
    COMMAND ${CMAKE_COMMAND}
        -DCONTRACT_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}
        -DCONTRACT_REQUIRE_ALL_APPROVED=ON
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/contracts/verify_operator_contracts.cmake
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    USES_TERMINAL
)

# 算子库验收与 Task2 分开进行。只有明确需要联编 Task2 时才打开该选项，避免
# Task2 的活动状态影响 53 个算子的配置、构建和五类型门禁。
option(BUILD_TASK2 "Build the Task2 pipeline together with cusignal_cpp" OFF)
if(BUILD_TASK2)
    set(TASK2_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Task2")
    if(NOT EXISTS "${TASK2_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "BUILD_TASK2=ON but Task2 source is missing: ${TASK2_SOURCE_DIR}")
    endif()
    add_subdirectory("${TASK2_SOURCE_DIR}"
                     "${CMAKE_CURRENT_BINARY_DIR}/components/task2_pipeline")
endif()

# Task1 与 Task2 使用相同的 ZQ500 typed 算子库和公共构建规则，但保持独立目标与
# 结果目录。Task1 的唯一活动实现由 Task1 根目录装配。
option(BUILD_TASK1 "Build the Task1 pipeline together with cusignal_cpp" OFF)
if(BUILD_TASK1)
    set(TASK1_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Task1")
    if(NOT EXISTS "${TASK1_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "BUILD_TASK1=ON but Task1 source is missing: ${TASK1_SOURCE_DIR}")
    endif()
    add_subdirectory("${TASK1_SOURCE_DIR}"
                     "${CMAKE_CURRENT_BINARY_DIR}/components/task1_pipeline")
endif()

# 旧 test_all_signals/benchmark/profile 依赖已归档的 double `signal_processing.h`
# 及 *_cuda.h，不能继续作为最新 typed 源码的活动目标。五类型正确性由上方 operator-type-coverage
# smoke 覆盖；正式 typed 性能门禁在算子契约全部闭环后另行登记。
