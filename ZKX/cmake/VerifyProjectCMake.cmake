# 活动工程 CMake 一致性门禁。
#
# 自动扫描活动源码树中的全部 CMake 文件；历史归档和第三方参考源码保持原貌，
# 不参与当前工程构建，也不作为当前命名规范的判定对象。
if(NOT DEFINED PROJECT_CMAKE_ROOT OR PROJECT_CMAKE_ROOT STREQUAL "")
    get_filename_component(PROJECT_CMAKE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

# 容器工作区会保留历史冻结快照和旧构建目录，其中部分目录是符号链接。
# 门禁只检查活动源码，禁止 GLOB_RECURSE 沿链接进入快照树。
if(POLICY CMP0009)
    cmake_policy(SET CMP0009 NEW)
endif()
file(GLOB_RECURSE PROJECT_CMAKE_FILES
    LIST_DIRECTORIES FALSE
    RELATIVE "${PROJECT_CMAKE_ROOT}"
    "${PROJECT_CMAKE_ROOT}/CMakeLists.txt"
    "${PROJECT_CMAKE_ROOT}/*.cmake")
list(FILTER PROJECT_CMAKE_FILES EXCLUDE REGEX "(^|/)Archiving/")
list(FILTER PROJECT_CMAKE_FILES EXCLUDE REGEX "(^|/)cusignal-23[.]08[.]00/")
list(FILTER PROJECT_CMAKE_FILES EXCLUDE REGEX "(^|/)cupy-13[.]6[.]0/")
list(FILTER PROJECT_CMAKE_FILES EXCLUDE REGEX "(^|/)[.][^/]+/")
list(FILTER PROJECT_CMAKE_FILES EXCLUDE REGEX "(^|/)build[^/]*/")
list(SORT PROJECT_CMAKE_FILES)

set(PROJECT_CMAKE_FORBIDDEN_PATTERNS
    "project\\([^\\n\\r]*languages[ \t]+cuda"
    "enable_language\\([ \t]*cuda"
    "find_package\\([ \t]*cuda"
    "(^|[^a-z0-9_])nvcc([^a-z0-9_]|$)"
    "cmake_cuda_"
    "cuda_nvcc_flags"
    "(^|[^a-z0-9_])(sm|compute)_[0-9]+([^a-z0-9_]|$)"
    "/usr/local/cuda"
    "复赛"
    "semi.?final"
    "(^|[^a-z0-9_])task[_-]?[efg]([^a-z0-9_]|$)"
    "(^|[^a-z0-9_])[bcdefghi][0-9]+[ab]?([^a-z0-9_]|$)"
    "stage0[0-9]"
    "cusignal_task"
    "build_visualization"
    "task[12]/task[12]_gpu_cpu"
    "test_all/(common|configs|stability)/"
    "test_all/operators/cases/[^/]+/_shared/"
    "test_all/operators/accuracy/"
    "(status|build_run|tests)[=:]passed"
    "src/(convolution|demod|estimation|filter_design|filtering|peak_finding|radartools|spectral_analysis|waveforms|wavelets|windows)/(convolution|demod|estimation|filter_design|filtering|peak_finding|radartools|spectral_analysis|waveforms|wavelets|windows)[.]cpp"
    "src/(convolution|demod|estimation|filter_design|filtering|peak_finding|radartools|spectral_analysis|waveforms|wavelets|windows)/(convolution|demod|estimation|filter_design|filtering|peak_finding|radartools|spectral_analysis|waveforms|wavelets|windows)_cuda[.]cu"
    "target_link_libraries\\([^\\)]*[ \t\r\n](cudart|cufft|curand)([ \t\r\n\\)]|$)")

set(PROJECT_CMAKE_VIOLATIONS)
set(PROJECT_REQUIRED_CMAKE_FILES
    "CMakeLists.txt"
    "Task1/CMakeLists.txt"
    "Task2/CMakeLists.txt"
    "test_all/operators/fft_backend/CMakeLists.txt")
foreach(PROJECT_REQUIRED_CMAKE_FILE IN LISTS PROJECT_REQUIRED_CMAKE_FILES)
    if(NOT EXISTS "${PROJECT_CMAKE_ROOT}/${PROJECT_REQUIRED_CMAKE_FILE}")
        list(APPEND PROJECT_CMAKE_VIOLATIONS
            "${PROJECT_REQUIRED_CMAKE_FILE}:missing-required-cmake")
    endif()
endforeach()

foreach(PROJECT_CMAKE_REL IN LISTS PROJECT_CMAKE_FILES)
    set(PROJECT_CMAKE_FILE "${PROJECT_CMAKE_ROOT}/${PROJECT_CMAKE_REL}")
    file(READ "${PROJECT_CMAKE_FILE}" PROJECT_CMAKE_CONTENT)
    string(TOLOWER "${PROJECT_CMAKE_CONTENT}" PROJECT_CMAKE_CONTENT_LOWER)

    if(NOT PROJECT_CMAKE_REL STREQUAL "cmake/VerifyProjectCMake.cmake")
        foreach(PROJECT_CMAKE_PATTERN IN LISTS PROJECT_CMAKE_FORBIDDEN_PATTERNS)
            if(PROJECT_CMAKE_CONTENT_LOWER MATCHES "${PROJECT_CMAKE_PATTERN}")
                list(APPEND PROJECT_CMAKE_VIOLATIONS
                    "${PROJECT_CMAKE_REL}:${PROJECT_CMAKE_PATTERN}")
            endif()
        endforeach()
    endif()

    string(REGEX MATCH
        "cmake_minimum_required\\([ \t]*version[ \t]+([0-9]+)\\.([0-9]+)"
        PROJECT_CMAKE_MINIMUM_MATCH "${PROJECT_CMAKE_CONTENT_LOWER}")
    if(PROJECT_CMAKE_MINIMUM_MATCH)
        if(CMAKE_MATCH_1 GREATER 3 OR
           (CMAKE_MATCH_1 EQUAL 3 AND CMAKE_MATCH_2 GREATER 16))
            list(APPEND PROJECT_CMAKE_VIOLATIONS
                "${PROJECT_CMAKE_REL}:cmake_minimum_required>${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
        endif()
    endif()
endforeach()

# 可视化入口消费正式 pipeline，必须与统一 preset 的构建目录保持一致。
set(PROJECT_BUILD_LAYOUT_CONSUMERS
    "Task1/visualization/python/visualization_common.py"
    "Task2/visualization/python/visualization_common.py")
foreach(PROJECT_BUILD_LAYOUT_CONSUMER IN LISTS PROJECT_BUILD_LAYOUT_CONSUMERS)
    set(PROJECT_BUILD_LAYOUT_FILE
        "${PROJECT_CMAKE_ROOT}/${PROJECT_BUILD_LAYOUT_CONSUMER}")
    if(EXISTS "${PROJECT_BUILD_LAYOUT_FILE}")
        file(READ "${PROJECT_BUILD_LAYOUT_FILE}" PROJECT_BUILD_LAYOUT_CONTENT)
        string(TOLOWER "${PROJECT_BUILD_LAYOUT_CONTENT}"
            PROJECT_BUILD_LAYOUT_CONTENT_LOWER)
        if(PROJECT_BUILD_LAYOUT_CONTENT_LOWER MATCHES "build_visualization")
            list(APPEND PROJECT_CMAKE_VIOLATIONS
                "${PROJECT_BUILD_LAYOUT_CONSUMER}:build_visualization")
        endif()
    endif()
endforeach()

file(READ "${PROJECT_CMAKE_ROOT}/cmake/ZQ500Build.cmake" PROJECT_ZQ500_BUILD_CONTENT)
string(FIND "${PROJECT_ZQ500_BUILD_CONTENT}" "-dlgpu-full-shfl" PROJECT_FULL_SHFL_INDEX)
if(PROJECT_FULL_SHFL_INDEX EQUAL -1)
    list(APPEND PROJECT_CMAKE_VIOLATIONS
        "cmake/ZQ500Build.cmake:missing-dlgpu-full-shfl")
endif()

list(LENGTH PROJECT_CMAKE_FILES PROJECT_CMAKE_COUNT)
list(LENGTH PROJECT_CMAKE_VIOLATIONS PROJECT_CMAKE_VIOLATION_COUNT)
if(PROJECT_CMAKE_VIOLATION_COUNT GREATER 0)
    list(JOIN PROJECT_CMAKE_VIOLATIONS "\n  " PROJECT_CMAKE_VIOLATION_TEXT)
    message(FATAL_ERROR
        "[PROJECT][CMAKE-SCAN] violations=${PROJECT_CMAKE_VIOLATION_COUNT}\n  ${PROJECT_CMAKE_VIOLATION_TEXT}")
endif()

message(STATUS
    "[PROJECT][CMAKE-SCAN] files=${PROJECT_CMAKE_COUNT} violations=0")
