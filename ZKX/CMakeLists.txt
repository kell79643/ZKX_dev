cmake_minimum_required(VERSION 3.16)
include("${CMAKE_CURRENT_LIST_DIR}/cmake/ZQ500Build.cmake")

project(zkx_zq500 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()

# 所有目标共享同一产物根，调用方不再依赖 add_subdirectory 生成的深层内部路径。
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# 配置期先做一次活动 CMake 门禁，统一测试目标再复验一次。
set(PROJECT_CMAKE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifyProjectCMake.cmake")
add_custom_target(project_cmake_gate
    COMMAND ${CMAKE_COMMAND}
        -DPROJECT_CMAKE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifyProjectCMake.cmake
    USES_TERMINAL)

# 统一入口装配 FFT backend、公共算子基础设施和 Task1/Task2。53 个正式算子 case
# 保持各目录独立配置，由正式脚本和 demo 调度；聚合目标只构建，不伪造运行结论。
set(BUILD_TASK1 OFF CACHE BOOL "Task1 is assembled by the project root" FORCE)
set(BUILD_TASK2 OFF CACHE BOOL "Task2 is assembled by the project root" FORCE)
add_subdirectory(test_all/operators/fft_backend components/fft_backend_tests)
add_subdirectory(Task1 components/task1_pipeline)
add_subdirectory(Task2 components/task2_pipeline)

set(PROJECT_AGGREGATE_DEPENDENCIES
    project_cmake_gate
    task1_all
    task2_all)
foreach(PROJECT_AGGREGATE_DEPENDENCY IN LISTS PROJECT_AGGREGATE_DEPENDENCIES)
    if(NOT TARGET ${PROJECT_AGGREGATE_DEPENDENCY})
        message(FATAL_ERROR
            "test_all dependency target is missing: ${PROJECT_AGGREGATE_DEPENDENCY}")
    endif()
endforeach()

add_custom_target(test_all DEPENDS ${PROJECT_AGGREGATE_DEPENDENCIES})
