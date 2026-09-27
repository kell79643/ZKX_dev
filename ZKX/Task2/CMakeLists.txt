find_package(Threads REQUIRED)

set(TASK2_PIPELINE_SOURCES
    pipeline/src/task2_common.cu
    pipeline/src/task2_runner.cu
    pipeline/src/steps/step1/step1.cu
    pipeline/src/steps/step2/step2.cu
    pipeline/src/steps/step3/step3.cu
    pipeline/src/steps/step4/step4.cu
    pipeline/src/steps/step5/step5.cu
    pipeline/src/steps/step6/step6.cu
    accuracy/accuracy_utils.cpp
    accuracy/validation/step1_validation.cpp
    accuracy/validation/step2_validation.cpp
    accuracy/validation/step3_validation.cpp
    accuracy/validation/step4_validation.cpp
    accuracy/validation/step5_cpu_validation.cpp
    accuracy/validation/step5_validation.cu
    accuracy/validation/step6_validation.cpp
    accuracy/comparison/step1_comparison.cpp
    accuracy/comparison/step2_comparison.cpp
    accuracy/comparison/step3_comparison.cpp
    accuracy/comparison/step4_comparison.cpp
    accuracy/comparison/step5_comparison.cpp
    accuracy/comparison/step6_comparison.cpp
    visualization/cpp/pipeline_visualization_export.cpp)

set(TASK2_SUPPORT_SOURCES
    ../test_all/support/config/json_value.cpp
    ../test_all/support/metrics/accuracy.cpp
    ../test_all/support/metrics/identity.cpp
    ../test_all/support/metrics/statistics.cpp
    ../test_all/support/metrics/memory.cpp
    ../test_all/support/metrics/csv_writer.cpp)

set(TASK2_CUDA_SOURCES
    ${TASK2_PIPELINE_SOURCES}
    pipeline/apps/pipeline/main.cu
    pipeline/apps/steps/step1/main.cu
    pipeline/apps/steps/step2/main.cu
    pipeline/apps/steps/step3/main.cu
    pipeline/apps/steps/step4/main.cu
    pipeline/apps/steps/step5/main.cu
    pipeline/apps/steps/step6/main.cu
    stability/memory/memory_stability_main.cu
    performance/cpp/step1_benchmark_main.cu
    performance/cpp/step2_benchmark_main.cu
    performance/cpp/step3_benchmark_main.cu
    performance/cpp/step4_benchmark_main.cu
    performance/cpp/step5_benchmark_main.cu
    performance/cpp/step6_benchmark_main.cu
    performance/cpp/pipeline_benchmark_main.cu)
zq500_mark_cuda_sources(${TASK2_CUDA_SOURCES})

set(TASK2_EVIDENCE_GIT_COMMIT "unbound" CACHE STRING
    "Git commit bound to Task2 runtime evidence")
set(TASK2_EVIDENCE_GIT_DIRTY "unknown" CACHE STRING
    "Whether the Task2 evidence build had uncommitted changes")
set(TASK2_EVIDENCE_TEST_TIME "unbound" CACHE STRING
    "ISO-8601 test time bound to Task2 runtime evidence")

add_library(task2_pipeline_core OBJECT ${TASK2_PIPELINE_SOURCES})
configure_zq500_target(task2_pipeline_core)
target_include_directories(task2_pipeline_core
    PUBLIC "${CMAKE_CURRENT_LIST_DIR}/pipeline/include"
    PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}"
        "${CMAKE_CURRENT_LIST_DIR}/accuracy"
        "${CMAKE_CURRENT_LIST_DIR}/visualization/cpp"
        "${CMAKE_SOURCE_DIR}/cusignal_cpp/src/runtime")
target_compile_definitions(task2_pipeline_core PRIVATE
    TASK2_EVIDENCE_GIT_COMMIT="${TASK2_EVIDENCE_GIT_COMMIT}"
    TASK2_EVIDENCE_GIT_DIRTY="${TASK2_EVIDENCE_GIT_DIRTY}"
    TASK2_EVIDENCE_TEST_TIME="${TASK2_EVIDENCE_TEST_TIME}")

function(add_task2_entry target source)
    add_executable(${target} ${source} $<TARGET_OBJECTS:task2_pipeline_core>)
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

add_task2_entry(task2_pipeline pipeline/apps/pipeline/main.cu)
foreach(TASK2_STEP RANGE 1 6)
    add_task2_entry(task2_step${TASK2_STEP}
        pipeline/apps/steps/step${TASK2_STEP}/main.cu)
endforeach()
add_task2_entry(task2_memory_stability
    stability/memory/memory_stability_main.cu)
target_link_libraries(task2_memory_stability PRIVATE Threads::Threads)

function(add_task2_benchmark target source)
    add_executable(${target}
        ${source}
        performance/cpp/task2_scale_config.cpp
        accuracy/formal_accuracy.cpp
        ${TASK2_SUPPORT_SOURCES}
        $<TARGET_OBJECTS:task2_pipeline_core>)
    configure_zq500_target(${target})
    target_include_directories(${target} PRIVATE
        "${CMAKE_SOURCE_DIR}"
        "${CMAKE_CURRENT_LIST_DIR}"
        "${CMAKE_CURRENT_LIST_DIR}/pipeline/include"
        "${CMAKE_CURRENT_LIST_DIR}/accuracy"
        "${CMAKE_CURRENT_LIST_DIR}/performance/cpp"
        "${CMAKE_CURRENT_LIST_DIR}/visualization/cpp"
        "${CMAKE_SOURCE_DIR}/cusignal_cpp/src/runtime")
    target_link_libraries(${target} PRIVATE signal_lib fft_core Threads::Threads)
    target_compile_definitions(${target} PRIVATE
        TASK2_EVIDENCE_GIT_COMMIT="${TASK2_EVIDENCE_GIT_COMMIT}"
        TASK2_EVIDENCE_GIT_DIRTY="${TASK2_EVIDENCE_GIT_DIRTY}"
        TASK2_EVIDENCE_TEST_TIME="${TASK2_EVIDENCE_TEST_TIME}")
endfunction()

foreach(TASK2_STEP RANGE 1 6)
    add_task2_benchmark(task2_step${TASK2_STEP}_benchmark
        performance/cpp/step${TASK2_STEP}_benchmark_main.cu)
endforeach()
add_task2_benchmark(task2_pipeline_benchmark
    performance/cpp/pipeline_benchmark_main.cu)

# 本目标只聚合实际存在的构建产物；运行与结果判定由任务专属入口完成。
add_custom_target(task2_all DEPENDS
    task2_pipeline task2_memory_stability
    task2_step1 task2_step2 task2_step3 task2_step4 task2_step5 task2_step6
    task2_step1_benchmark task2_step2_benchmark task2_step3_benchmark
    task2_step4_benchmark task2_step5_benchmark task2_step6_benchmark
    task2_pipeline_benchmark)
