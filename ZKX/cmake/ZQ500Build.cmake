include_guard(GLOBAL)

# 统一的 ZQ500 CMake 基线。该文件可在 project() 前包含，以便所有独立入口都使用
# SDK 初始化后提供的 dlcc；CUDA 源码仍按官方样例作为 CXX 编译。
if(NOT DEFINED ENV{DLICC_PATH} OR "$ENV{DLICC_PATH}" STREQUAL "")
    message(FATAL_ERROR
        "DLICC_PATH is not set. Enter gpu_02 and run: source /zq500/sdk/env.sh")
endif()

get_filename_component(ZQ500_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(ZQ500_SDK_DIR "$ENV{DLICC_PATH}/..")
set(SDK_DIR "${ZQ500_SDK_DIR}")

if(WIN32)
    set(CMAKE_CXX_COMPILER "clang-cl.exe")
else()
    set(CMAKE_CXX_COMPILER "$ENV{DLICC_PATH}/dlcc")
endif()

if(DEFINED ENV{DLI_V2} AND NOT "$ENV{DLI_V2}" STREQUAL "")
    set(ZQ500_GPU_OPTIONS
        --offload-arch=dlgput64,dlgpux64
        -dlgpu-full-shfl
        -x cuda)
    add_compile_definitions(DLI_V2)
    message(STATUS "Using ZQ500 DLI_V2 architecture options")
else()
    set(ZQ500_GPU_OPTIONS --cuda-gpu-arch=dlgpuc64 -x cuda)
    message(STATUS "Using ZQ500 DLI_V1 architecture options")
endif()
set(ZQ500_CUDA_COMPILE_OPTIONS ${ZQ500_GPU_OPTIONS})

# 根工程和仍需独立配置的算子 case 共用同一产物布局。无论 CMake 从哪个源码入口
# 配置，可执行文件都在 <binaryDir>/bin，库都在 <binaryDir>/lib；调用脚本不得再
# 猜测 add_subdirectory 生成的内部层级。
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

function(zq500_require_explicit_fft_backend)
    foreach(FFT_OPTION USE_DLFFT USE_THRUST)
        if(NOT DEFINED ${FFT_OPTION})
            message(FATAL_ERROR
                "FFT backend must be explicit: pass both "
                "-DUSE_DLFFT=ON/OFF and -DUSE_THRUST=ON/OFF")
        endif()
        string(TOUPPER "${${FFT_OPTION}}" FFT_OPTION_VALUE)
        if(NOT FFT_OPTION_VALUE STREQUAL "ON" AND
           NOT FFT_OPTION_VALUE STREQUAL "OFF")
            message(FATAL_ERROR
                "${FFT_OPTION} must be explicitly ON or OFF, got '${${FFT_OPTION}}'")
        endif()
        set(${FFT_OPTION} "${FFT_OPTION_VALUE}" CACHE BOOL
            "Explicit ZQ500 FFT backend selector" FORCE)
    endforeach()

    if(USE_DLFFT STREQUAL USE_THRUST)
        message(FATAL_ERROR
            "Exactly one FFT backend must be ON: "
            "USE_DLFFT=${USE_DLFFT}, USE_THRUST=${USE_THRUST}")
    endif()

    if(USE_DLFFT)
        find_library(ZQ500_DLFFT_LIBRARY dlfft HINTS "${ZQ500_SDK_DIR}/lib")
        if(NOT ZQ500_DLFFT_LIBRARY)
            message(FATAL_ERROR
                "ZQ500 dlfft library was not found under ${ZQ500_SDK_DIR}/lib")
        endif()
        set(DLFFT_LIBRARY "${ZQ500_DLFFT_LIBRARY}" PARENT_SCOPE)
        set(ZQ500_FFT_BACKEND "DLFFT" PARENT_SCOPE)
        message(STATUS "ZQ500 FFT backend: DLFFT")
    else()
        set(ZQ500_FFT_BACKEND "THRUST" PARENT_SCOPE)
        message(STATUS "ZQ500 FFT backend: THRUST")
    endif()
endfunction()

function(zq500_mark_cuda_sources)
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CXX)
endfunction()

function(zq500_add_fft_thrust_shared)
    if(TARGET zkx_fft_thrust)
        return()
    endif()

    set(ZQ500_FFT_THRUST_SOURCES
        "${ZQ500_REPO_ROOT}/cusignal_cpp/src/backends/fft/fft_thrust.cu"
        "${ZQ500_REPO_ROOT}/cusignal_cpp/src/backends/fft/fft_thrust_cufft_compat.cu")
    add_library(zkx_fft_thrust SHARED ${ZQ500_FFT_THRUST_SOURCES})
    set_source_files_properties(${ZQ500_FFT_THRUST_SOURCES} PROPERTIES LANGUAGE CXX)
    target_compile_features(zkx_fft_thrust PRIVATE cxx_std_17)
    target_include_directories(zkx_fft_thrust PRIVATE
        "${ZQ500_REPO_ROOT}/cusignal_cpp/include"
        "${ZQ500_REPO_ROOT}/cusignal_cpp/src/backends/fft"
        "${ZQ500_REPO_ROOT}/cusignal_cpp/src/runtime"
        "${ZQ500_SDK_DIR}/include")
    target_link_directories(zkx_fft_thrust PRIVATE "${ZQ500_SDK_DIR}/lib")
    target_link_libraries(zkx_fft_thrust PRIVATE curt)
    target_compile_options(zkx_fft_thrust PRIVATE ${ZQ500_GPU_OPTIONS})
    set_target_properties(zkx_fft_thrust PROPERTIES
        OUTPUT_NAME "zkx_fft_thrust"
        POSITION_INDEPENDENT_CODE ON)
endfunction()

function(zq500_assert_no_embedded_fft_thrust target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR
            "zq500_assert_no_embedded_fft_thrust: unknown target ${target}")
    endif()
    get_target_property(ZQ500_TARGET_SOURCES ${target} SOURCES)
    if(NOT ZQ500_TARGET_SOURCES)
        return()
    endif()
    foreach(ZQ500_TARGET_SOURCE IN LISTS ZQ500_TARGET_SOURCES)
        string(REPLACE "\\" "/" ZQ500_TARGET_SOURCE_NORMALIZED
            "${ZQ500_TARGET_SOURCE}")
        if(ZQ500_TARGET_SOURCE_NORMALIZED MATCHES "(^|/)fft_thrust[.]cu$")
            message(FATAL_ERROR
                "Target ${target} embeds fft_thrust.cu directly; "
                "link zkx_fft_thrust / libzkx_fft_thrust.so instead")
        endif()
    endforeach()
endfunction()

function(zq500_link_fft_thrust target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "zq500_link_fft_thrust: unknown target ${target}")
    endif()
    if(DEFINED USE_DLFFT AND USE_DLFFT)
        message(FATAL_ERROR
            "${target}: zq500_link_fft_thrust cannot be used when USE_DLFFT=ON")
    endif()
    if(DEFINED USE_THRUST AND NOT USE_THRUST)
        message(FATAL_ERROR
            "${target}: zq500_link_fft_thrust requires USE_THRUST=ON when the selector is explicit")
    endif()
    zq500_assert_no_embedded_fft_thrust(${target})
    zq500_add_fft_thrust_shared()
    target_link_libraries(${target} PRIVATE zkx_fft_thrust curt)
    target_compile_definitions(${target} PRIVATE USE_THRUST)
endfunction()

# 为直接编译独立算子probe的目标提供统一FFT后端。上层仍调用稳定fft_*接口：
# fft_thrust分支链接libzkx_fft_thrust.so；dlfft分支加入fft_dlfft.cu适配层并
# 链接供应商libdlfft.so。调用者不得再嵌入fft_thrust.cu或硬编码USE_THRUST。
function(zq500_link_selected_fft target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "zq500_link_selected_fft: unknown target ${target}")
    endif()
    zq500_require_explicit_fft_backend()
    zq500_assert_no_embedded_fft_thrust(${target})
    if(USE_DLFFT)
        set(ZQ500_DLFFT_ADAPTER_SOURCE
            "${ZQ500_REPO_ROOT}/cusignal_cpp/src/backends/fft/fft_dlfft.cu")
        target_sources(${target} PRIVATE "${ZQ500_DLFFT_ADAPTER_SOURCE}")
        set_source_files_properties(
            "${ZQ500_DLFFT_ADAPTER_SOURCE}" PROPERTIES LANGUAGE CXX)
        target_link_directories(${target} PRIVATE "${ZQ500_SDK_DIR}/lib")
        target_link_libraries(${target} PRIVATE "${DLFFT_LIBRARY}" curt)
        target_compile_definitions(${target} PRIVATE
            USE_DLFFT ZKX_SELECTED_FFT_BACKEND_NAME="dlfft")
    else()
        zq500_link_fft_thrust(${target})
        target_compile_definitions(${target} PRIVATE
            ZKX_SELECTED_FFT_BACKEND_NAME="fft_thrust")
    endif()
endfunction()

function(configure_zq500_target target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "configure_zq500_target: unknown target ${target}")
    endif()
    target_include_directories(${target} PRIVATE
        "${ZQ500_REPO_ROOT}/demo/include"
        "${ZQ500_REPO_ROOT}/cusignal_cpp/include"
        "${ZQ500_REPO_ROOT}/cusignal_cpp/src/runtime"
        "${ZQ500_REPO_ROOT}/cusignal_cpp/src/backends/fft"
        "${ZQ500_REPO_ROOT}/cusignal_cpp/src/backends/random"
        "${ZQ500_REPO_ROOT}/cusignal_cpp/src/operators/bsplines"
        "${ZQ500_REPO_ROOT}/cusignal_cpp/src/operators/convolution"
        "${ZQ500_REPO_ROOT}/cusignal_cpp/src/operators/demod"
        "${ZQ500_REPO_ROOT}/cusignal_cpp/src/operators/estimation"
        "${ZQ500_REPO_ROOT}/cusignal_cpp/src/operators/filter_design"
        "${ZQ500_REPO_ROOT}/cusignal_cpp/src/operators/filtering"
        "${ZQ500_REPO_ROOT}/cusignal_cpp/src/operators/peak_finding"
        "${ZQ500_REPO_ROOT}/cusignal_cpp/src/operators/radartools"
        "${ZQ500_REPO_ROOT}/cusignal_cpp/src/operators/spectral_analysis"
        "${ZQ500_REPO_ROOT}/cusignal_cpp/src/operators/waveforms"
        "${ZQ500_REPO_ROOT}/cusignal_cpp/src/operators/wavelets"
        "${ZQ500_REPO_ROOT}/cusignal_cpp/src/operators/windows"
        "${ZQ500_SDK_DIR}/include")
    target_link_directories(${target} PRIVATE "${ZQ500_SDK_DIR}/lib")
    target_link_libraries(${target} PRIVATE curt)
    if(USE_DLFFT)
        # DLFFT 后端保持供应商实现边界：Plan/Exec/Destroy 均直接来自 libdlfft.so。
        target_link_libraries(${target} PRIVATE "${DLFFT_LIBRARY}")
        target_compile_definitions(${target} PRIVATE USE_DLFFT)
    elseif(USE_THRUST)
        zq500_link_fft_thrust(${target})
    else()
        message(FATAL_ERROR
            "configure_zq500_target requires zq500_require_explicit_fft_backend()")
    endif()
    target_compile_options(${target} PRIVATE ${ZQ500_GPU_OPTIONS})
endfunction()
