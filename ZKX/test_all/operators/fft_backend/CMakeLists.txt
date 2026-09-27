cmake_minimum_required(VERSION 3.16)
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
include("${REPO_ROOT}/cmake/ZQ500Build.cmake")

project(zq500_fft_backend_tests LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

zq500_require_explicit_fft_backend()
zq500_add_fft_thrust_shared()
if(USE_DLFFT)
    set(ZQ500_SELECTED_FFT_SOURCE
        "${REPO_ROOT}/cusignal_cpp/src/backends/fft/fft_dlfft.cu")
    set(ZQ500_SELECTED_FFT_DEFINITION USE_DLFFT)
    set(ZQ500_SELECTED_FFT_LIBRARIES "${DLFFT_LIBRARY}" curt)
else()
    set(ZQ500_SELECTED_FFT_SOURCE)
    set(ZQ500_SELECTED_FFT_DEFINITION USE_THRUST)
    set(ZQ500_SELECTED_FFT_LIBRARIES zkx_fft_thrust curt)
endif()

add_subdirectory("${REPO_ROOT}/cusignal_cpp"
    "${CMAKE_BINARY_DIR}/components/cusignal_cpp")

function(add_fft_backend_test target source)
    set(sources "${CMAKE_CURRENT_LIST_DIR}/src/${source}"
        ${ZQ500_SELECTED_FFT_SOURCE})
    zq500_mark_cuda_sources(${sources})
    add_executable(${target} ${sources})
    target_compile_definitions(${target} PRIVATE
        ${ZQ500_SELECTED_FFT_DEFINITION})
    target_link_libraries(${target} PRIVATE ${ZQ500_SELECTED_FFT_LIBRARIES})
    target_link_options(${target} PRIVATE -Wl,--gc-sections)
endfunction()

add_fft_backend_test(fft_backend_performance fft_backend_performance.cu)
add_fft_backend_test(fft_backend_stability fft_backend_stability.cu)
add_fft_backend_test(fft_backend_memory_unified fft_backend_memory_unified.cu)

# 动态库平替演示必须由同一份业务源码生成两个二进制，唯一差异是链接库名。
set(FFT_LIBRARY_SWAP_SOURCE
    "${CMAKE_CURRENT_LIST_DIR}/src/fft_library_swap_demo.cu")
zq500_mark_cuda_sources(${FFT_LIBRARY_SWAP_SOURCE})
find_library(FFT_LIBRARY_SWAP_DLFFT_LIBRARY dlfft
    HINTS "${ZQ500_SDK_DIR}/lib")
if(NOT FFT_LIBRARY_SWAP_DLFFT_LIBRARY)
    message(FATAL_ERROR
        "ZQ500 dlfft library was not found under ${ZQ500_SDK_DIR}/lib")
endif()

function(add_fft_library_swap_target target library)
    add_executable(${target} ${FFT_LIBRARY_SWAP_SOURCE})
    target_include_directories(${target} PRIVATE "${ZQ500_SDK_DIR}/include")
    target_link_directories(${target} PRIVATE "${ZQ500_SDK_DIR}/lib")
    target_link_libraries(${target} PRIVATE ${library} curt dl)
    target_compile_options(${target} PRIVATE ${ZQ500_GPU_OPTIONS})
    target_link_options(${target} PRIVATE -Wl,--gc-sections)
endfunction()

add_fft_library_swap_target(
    fft_library_swap_dlfft "${FFT_LIBRARY_SWAP_DLFFT_LIBRARY}")
set_target_properties(fft_library_swap_dlfft PROPERTIES
    BUILD_RPATH "${ZQ500_SDK_DIR}/lib")

add_fft_library_swap_target(
    fft_library_swap_zkx_fft_thrust zkx_fft_thrust)
set_target_properties(fft_library_swap_zkx_fft_thrust PROPERTIES
    BUILD_RPATH "\$ORIGIN/../../fft_thrust/lib")
