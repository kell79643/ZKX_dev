include("${CMAKE_CURRENT_LIST_DIR}/cmake/verify_source_contract.cmake")
find_package(Threads REQUIRED)

set(TASK1_PIPELINE_SOURCES
    pipeline/src/task1_common.cu
    pipeline/src/task1_runner.cu
    pipeline/src/steps/step1/step1.cu
    pipeline/src/steps/step2/step2.cu
    pipeline/src/steps/step3/step3.cu
    pipeline/src/steps/step4/step4.cu
    pipeline/src/steps/step5/step5.cu
    accuracy/accuracy_utils.cpp
    accuracy/validation/step1_validation.cu
    accuracy/validation/step2_validation.cu
    accuracy/validation/step3_validation.cu
    accuracy/validation/step4_validation.cu
    accuracy/validation/step5_validation.cu
    accuracy/comparison/step1_comparison.cpp
    accuracy/comparison/step2_comparison.cpp
    accuracy/comparison/step3_comparison.cpp
    accuracy/comparison/step4_comparison.cpp
    accuracy/comparison/step5_comparison.cpp
    visualization/cpp/pipeline_visualization_export.cpp)

set(TASK1_SUPPORT_SOURCES
    ../test_all/support/config/json_value.cpp
    ../test_all/support/metrics/identity.cpp
    ../test_all/support/metrics/statistics.cpp
    ../test_all/support/metrics/memory.cpp
    ../test_all/support/metrics/csv_writer.cpp)

set(TASK1_CUDA_SOURCES
    ${TASK1_PIPELINE_SOURCES}
    pipeline/apps/pipeline/main.cu
    pipeline/apps/steps/step1/main.cu
    pipeline/apps/steps/step2/main.cu
    pipeline/apps/steps/step3/main.cu
    pipeline/apps/steps/step4/main.cu
    pipeline/apps/steps/step5/main.cu
    stability/memory/memory_stability_main.cu
    performance/cpp/step1_benchmark_main.cu
    performance/cpp/step2_benchmark_main.cu
    performance/cpp/step3_benchmark_main.cu
    performance/cpp/step4_benchmark_main.cu
    performance/cpp/step5_benchmark_main.cu
    performance/cpp/pipeline_benchmark_main.cu)
zq500_mark_cuda_sources(${TASK1_CUDA_SOURCES})

set(TASK1_EVIDENCE_GIT_COMMIT "unbound" CACHE STRING
    "Git commit bound to Task1 runtime evidence")
set(TASK1_EVIDENCE_GIT_DIRTY "unknown" CACHE STRING
    "Whether the Task1 evidence build had uncommitted changes")
set(TASK1_EVIDENCE_TEST_TIME "unbound" CACHE STRING
    "ISO-8601 test time bound to Task1 runtime evidence")

add_library(task1_pipeline_core OBJECT ${TASK1_PIPELINE_SOURCES})
configure_zq500_target(task1_pipeline_core)
target_include_directories(task1_pipeline_core
    PUBLIC "${CMAKE_CURRENT_LIST_DIR}/pipeline/include"
    PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}"
        "${CMAKE_CURRENT_LIST_DIR}/accuracy"
        "${CMAKE_CURRENT_LIST_DIR}/visualization/cpp"
        "${CMAKE_SOURCE_DIR}/cusignal_cpp/src/runtime")
target_compile_definitions(task1_pipeline_core PRIVATE
    TASK1_EVIDENCE_GIT_COMMIT="${TASK1_EVIDENCE_GIT_COMMIT}"
    TASK1_EVIDENCE_GIT_DIRTY="${TASK1_EVIDENCE_GIT_DIRTY}"
    TASK1_EVIDENCE_TEST_TIME="${TASK1_EVIDENCE_TEST_TIME}")

function(add_task1_entry target source)
    add_executable(${target} ${source} $<TARGET_OBJECTS:task1_pipeline_core>)
    configure_zq500_target(${target})
    target_include_directories(${target} PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/pipeline/include"
        "${CMAKE_CURRENT_LIST_DIR}"
        "${CMAKE_CURRENT_LIST_DIR}/accuracy"
        "${CMAKE_CURRENT_LIST_DIR}/performance/cpp"
        "${CMAKE_CURRENT_LIST_DIR}/visualization/cpp"
        "${CMAKE_SOURCE_DIR}/cusignal_cpp/src/runtime")
    target_link_libraries(${target} PRIVATE signal_lib fft_core)
endfunction()

add_task1_entry(task1_pipeline pipeline/apps/pipeline/main.cu)
foreach(TASK1_STEP RANGE 1 5)
    add_task1_entry(task1_step${TASK1_STEP}
        pipeline/apps/steps/step${TASK1_STEP}/main.cu)
endforeach()
add_task1_entry(task1_memory_stability
    stability/memory/memory_stability_main.cu)
target_link_libraries(task1_memory_stability PRIVATE Threads::Threads)

function(add_task1_benchmark target source)
    add_executable(${target}
        ${source}
        performance/cpp/task1_scale_config.cpp
        accuracy/formal_accuracy.cpp
        ${TASK1_SUPPORT_SOURCES}
        $<TARGET_OBJECTS:task1_pipeline_core>)
    configure_zq500_target(${target})
    target_include_directories(${target} PRIVATE
        "${CMAKE_SOURCE_DIR}"
        "${CMAKE_CURRENT_LIST_DIR}"
        "${CMAKE_CURRENT_LIST_DIR}/pipeline/include"
        "${CMAKE_CURRENT_LIST_DIR}/accuracy"
        "${CMAKE_CURRENT_LIST_DIR}/performance/cpp"
        "${CMAKE_CURRENT_LIST_DIR}/visualization/cpp"
        "${CMAKE_SOURCE_DIR}/cusignal_cpp/src/runtime")
    target_link_libraries(${target} PRIVATE signal_lib fft_core)
    target_compile_definitions(${target} PRIVATE
        TASK1_EVIDENCE_GIT_COMMIT="${TASK1_EVIDENCE_GIT_COMMIT}"
        TASK1_EVIDENCE_GIT_DIRTY="${TASK1_EVIDENCE_GIT_DIRTY}"
        TASK1_EVIDENCE_TEST_TIME="${TASK1_EVIDENCE_TEST_TIME}")
endfunction()

foreach(TASK1_STEP RANGE 1 5)
    add_task1_benchmark(task1_step${TASK1_STEP}_benchmark
        performance/cpp/step${TASK1_STEP}_benchmark_main.cu)
endforeach()
add_task1_benchmark(task1_pipeline_benchmark
    performance/cpp/pipeline_benchmark_main.cu)

# 本目标只聚合实际存在的构建产物；运行与结果判定由任务专属入口完成。
add_custom_target(task1_all DEPENDS
    task1_pipeline task1_memory_stability
    task1_step1 task1_step2 task1_step3 task1_step4 task1_step5
    task1_step1_benchmark task1_step2_benchmark task1_step3_benchmark
    task1_step4_benchmark task1_step5_benchmark task1_pipeline_benchmark)
