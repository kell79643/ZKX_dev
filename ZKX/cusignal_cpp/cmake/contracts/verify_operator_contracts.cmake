cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED CONTRACT_SOURCE_ROOT OR CONTRACT_SOURCE_ROOT STREQUAL "")
    get_filename_component(CONTRACT_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
endif()

set(CONTRACT_DATA_FILE
    "${CONTRACT_SOURCE_ROOT}/cmake/contracts/operator_contracts.cmake")
if(NOT EXISTS "${CONTRACT_DATA_FILE}")
    message(FATAL_ERROR "53x5 contract data file is missing: ${CONTRACT_DATA_FILE}")
endif()

# CMake lists use semicolons as element separators even when a row was written as
# a quoted argument.  Evidence prose is pipe-delimited and therefore must not
# contain a raw semicolon, otherwise one logical row is split before field
# validation and the resulting error misleadingly reports a malformed row.
file(STRINGS "${CONTRACT_DATA_FILE}" CONTRACT_RAW_EVIDENCE_ROWS
    REGEX "^[ \t]*\"EV-")
foreach(CONTRACT_RAW_EVIDENCE_ROW IN LISTS CONTRACT_RAW_EVIDENCE_ROWS)
    if(CONTRACT_RAW_EVIDENCE_ROW MATCHES ";")
        message(FATAL_ERROR
            "Contract evidence prose contains a reserved semicolon; use a comma: "
            "${CONTRACT_RAW_EVIDENCE_ROW}")
    endif()
endforeach()
include("${CONTRACT_DATA_FILE}")

set(CONTRACT_ALLOWED_DTYPES FP32 FP16 INT32 INT16 INT8)
set(CONTRACT_ALLOWED_STATUSES draft reviewed approved blocked)

if(NOT DEFINED CONTRACT_WORKSPACE_ROOT OR CONTRACT_WORKSPACE_ROOT STREQUAL "")
    get_filename_component(
        CONTRACT_WORKSPACE_ROOT "${CONTRACT_SOURCE_ROOT}/../.." ABSOLUTE)
    # 本地仓库通常是 ZKX_dev/ZKX/cusignal_cpp，容器同步目录则是
    # /tmp/ZKX_dev/cusignal_cpp。两种布局都保留固定版本源码，选择实际包含
    # 两个参考树的最近祖先；参考树字节身份由同步后的整树一致性审计负责。
    if(NOT EXISTS
       "${CONTRACT_WORKSPACE_ROOT}/cusignal-${CUSIGNAL_REFERENCE_VERSION}" OR
       NOT EXISTS
       "${CONTRACT_WORKSPACE_ROOT}/cupy-${CUPY_REFERENCE_VERSION}")
        get_filename_component(
            CONTRACT_CONTAINER_WORKSPACE_ROOT "${CONTRACT_SOURCE_ROOT}/.." ABSOLUTE)
        if(EXISTS
           "${CONTRACT_CONTAINER_WORKSPACE_ROOT}/cusignal-${CUSIGNAL_REFERENCE_VERSION}" AND
           EXISTS
           "${CONTRACT_CONTAINER_WORKSPACE_ROOT}/cupy-${CUPY_REFERENCE_VERSION}")
            set(CONTRACT_WORKSPACE_ROOT "${CONTRACT_CONTAINER_WORKSPACE_ROOT}")
        endif()
    endif()
endif()
if(NOT DEFINED CONTRACT_CUSIGNAL_REFERENCE_ROOT OR
   CONTRACT_CUSIGNAL_REFERENCE_ROOT STREQUAL "")
    set(CONTRACT_CUSIGNAL_REFERENCE_ROOT
        "${CONTRACT_WORKSPACE_ROOT}/cusignal-${CUSIGNAL_REFERENCE_VERSION}")
endif()
if(NOT DEFINED CONTRACT_CUPY_REFERENCE_ROOT OR
   CONTRACT_CUPY_REFERENCE_ROOT STREQUAL "")
    set(CONTRACT_CUPY_REFERENCE_ROOT
        "${CONTRACT_WORKSPACE_ROOT}/cupy-${CUPY_REFERENCE_VERSION}")
endif()

# Evidence rows are executable provenance rather than free-form notes.  A source
# change invalidates its conclusions and forces the affected contracts to be
# reviewed again.
set(CONTRACT_KNOWN_EVIDENCE_IDS)
set(CONTRACT_APPROVED_EVIDENCE_IDS)
foreach(CONTRACT_ROW IN LISTS CUSIGNAL_CONTRACT_EVIDENCE)
    string(REPLACE "|" ";" CONTRACT_FIELDS "${CONTRACT_ROW}")
    list(LENGTH CONTRACT_FIELDS CONTRACT_FIELD_COUNT)
    if(NOT CONTRACT_FIELD_COUNT EQUAL 13)
        message(FATAL_ERROR "Malformed contract evidence row: ${CONTRACT_ROW}")
    endif()
    list(GET CONTRACT_FIELDS 0 CONTRACT_EVIDENCE_ID)
    list(GET CONTRACT_FIELDS 1 CONTRACT_EVIDENCE_FIELD)
    list(GET CONTRACT_FIELDS 2 CONTRACT_EVIDENCE_PROJECT)
    list(GET CONTRACT_FIELDS 5 CONTRACT_EVIDENCE_FILE)
    list(GET CONTRACT_FIELDS 6 CONTRACT_EVIDENCE_START_LINE)
    list(GET CONTRACT_FIELDS 7 CONTRACT_EVIDENCE_END_LINE)
    list(GET CONTRACT_FIELDS 9 CONTRACT_EVIDENCE_OBSERVATION)
    list(GET CONTRACT_FIELDS 10 CONTRACT_EVIDENCE_CONCLUSION)
    list(GET CONTRACT_FIELDS 11 CONTRACT_EVIDENCE_API)
    list(GET CONTRACT_FIELDS 12 CONTRACT_EVIDENCE_STATUS)

    list(FIND CONTRACT_KNOWN_EVIDENCE_IDS
        "${CONTRACT_EVIDENCE_ID}" CONTRACT_EVIDENCE_DUPLICATE_INDEX)
    if(NOT CONTRACT_EVIDENCE_DUPLICATE_INDEX EQUAL -1)
        message(FATAL_ERROR "Duplicate contract evidence: ${CONTRACT_EVIDENCE_ID}")
    endif()
    list(APPEND CONTRACT_KNOWN_EVIDENCE_IDS "${CONTRACT_EVIDENCE_ID}")

    list(FIND CONTRACT_ALLOWED_STATUSES
        "${CONTRACT_EVIDENCE_STATUS}" CONTRACT_EVIDENCE_STATUS_INDEX)
    if(CONTRACT_EVIDENCE_STATUS_INDEX EQUAL -1)
        message(FATAL_ERROR
            "Invalid evidence status for ${CONTRACT_EVIDENCE_ID}: "
            "${CONTRACT_EVIDENCE_STATUS}")
    endif()
    foreach(CONTRACT_REQUIRED_EVIDENCE_VALUE
        CONTRACT_EVIDENCE_ID CONTRACT_EVIDENCE_FIELD CONTRACT_EVIDENCE_FILE
        CONTRACT_EVIDENCE_OBSERVATION CONTRACT_EVIDENCE_CONCLUSION
        CONTRACT_EVIDENCE_API)
        if("${${CONTRACT_REQUIRED_EVIDENCE_VALUE}}" STREQUAL "" OR
           "${${CONTRACT_REQUIRED_EVIDENCE_VALUE}}" STREQUAL "unresolved")
            message(FATAL_ERROR
                "Evidence ${CONTRACT_EVIDENCE_ID} has unresolved field "
                "${CONTRACT_REQUIRED_EVIDENCE_VALUE}")
        endif()
    endforeach()
    if(NOT CONTRACT_EVIDENCE_START_LINE MATCHES "^[1-9][0-9]*$" OR
       NOT CONTRACT_EVIDENCE_END_LINE MATCHES "^[1-9][0-9]*$" OR
       CONTRACT_EVIDENCE_END_LINE LESS CONTRACT_EVIDENCE_START_LINE)
        message(FATAL_ERROR
            "Invalid source line range for ${CONTRACT_EVIDENCE_ID}: "
            "${CONTRACT_EVIDENCE_START_LINE}-${CONTRACT_EVIDENCE_END_LINE}")
    endif()
    if(CONTRACT_EVIDENCE_PROJECT STREQUAL "cusignal")
        set(CONTRACT_EVIDENCE_ROOT "${CONTRACT_CUSIGNAL_REFERENCE_ROOT}")
    elseif(CONTRACT_EVIDENCE_PROJECT STREQUAL "cupy")
        set(CONTRACT_EVIDENCE_ROOT "${CONTRACT_CUPY_REFERENCE_ROOT}")
    else()
        message(FATAL_ERROR
            "Unknown evidence project for ${CONTRACT_EVIDENCE_ID}: "
            "${CONTRACT_EVIDENCE_PROJECT}")
    endif()
    set(CONTRACT_EVIDENCE_ABSOLUTE_FILE
        "${CONTRACT_EVIDENCE_ROOT}/${CONTRACT_EVIDENCE_FILE}")
    if(NOT EXISTS "${CONTRACT_EVIDENCE_ABSOLUTE_FILE}")
        message(FATAL_ERROR
            "Evidence source is missing for ${CONTRACT_EVIDENCE_ID}: "
            "${CONTRACT_EVIDENCE_ABSOLUTE_FILE}")
    endif()
    # 参考树已经纳入同一 Git 白名单，并在同步后执行本地—容器整树一致性审计。
    # 配置期只验证证据结构和源文件存在，不再重复绑定版本字符串或逐文件 SHA-256。
    if(CONTRACT_EVIDENCE_STATUS STREQUAL "approved")
        list(APPEND CONTRACT_APPROVED_EVIDENCE_IDS "${CONTRACT_EVIDENCE_ID}")
    endif()
endforeach()
list(LENGTH CONTRACT_KNOWN_EVIDENCE_IDS CONTRACT_EVIDENCE_COUNT)

list(LENGTH CUSIGNAL_OPERATOR_CONTRACTS CONTRACT_OPERATOR_COUNT)
if(NOT CONTRACT_OPERATOR_COUNT EQUAL 53)
    message(FATAL_ERROR
        "Operator contract count must be 53, actual=${CONTRACT_OPERATOR_COUNT}")
endif()

set(CONTRACT_OPERATOR_IDS)
set(CONTRACT_OPERATOR_approved 0)
set(CONTRACT_OPERATOR_reviewed 0)
set(CONTRACT_OPERATOR_draft 0)
set(CONTRACT_OPERATOR_blocked 0)
foreach(CONTRACT_ROW IN LISTS CUSIGNAL_OPERATOR_CONTRACTS)
    string(REPLACE "|" ";" CONTRACT_FIELDS "${CONTRACT_ROW}")
    list(LENGTH CONTRACT_FIELDS CONTRACT_FIELD_COUNT)
    if(NOT CONTRACT_FIELD_COUNT EQUAL 7)
        message(FATAL_ERROR "Malformed operator contract row: ${CONTRACT_ROW}")
    endif()
    list(GET CONTRACT_FIELDS 0 CONTRACT_OPERATOR_ID)
    list(GET CONTRACT_FIELDS 4 CONTRACT_PUBLIC_HEADER)
    list(GET CONTRACT_FIELDS 6 CONTRACT_STATUS)

    list(FIND CONTRACT_OPERATOR_IDS "${CONTRACT_OPERATOR_ID}" CONTRACT_DUPLICATE_INDEX)
    if(NOT CONTRACT_DUPLICATE_INDEX EQUAL -1)
        message(FATAL_ERROR "Duplicate operator contract: ${CONTRACT_OPERATOR_ID}")
    endif()
    list(APPEND CONTRACT_OPERATOR_IDS "${CONTRACT_OPERATOR_ID}")

    list(FIND CONTRACT_ALLOWED_STATUSES "${CONTRACT_STATUS}" CONTRACT_STATUS_INDEX)
    if(CONTRACT_STATUS_INDEX EQUAL -1)
        message(FATAL_ERROR
            "Invalid operator contract status for ${CONTRACT_OPERATOR_ID}: ${CONTRACT_STATUS}")
    endif()
    math(EXPR CONTRACT_OPERATOR_${CONTRACT_STATUS}
        "${CONTRACT_OPERATOR_${CONTRACT_STATUS}} + 1")

    if(NOT EXISTS "${CONTRACT_SOURCE_ROOT}/${CONTRACT_PUBLIC_HEADER}")
        message(FATAL_ERROR
            "Public header recorded by ${CONTRACT_OPERATOR_ID} does not exist: "
            "${CONTRACT_PUBLIC_HEADER}")
    endif()
endforeach()

list(LENGTH CUSIGNAL_DTYPE_CONTRACTS CONTRACT_DTYPE_COUNT)
if(NOT CONTRACT_DTYPE_COUNT EQUAL 265)
    message(FATAL_ERROR
        "Dtype contract count must be 265, actual=${CONTRACT_DTYPE_COUNT}")
endif()

set(CONTRACT_DTYPE_KEYS)
set(CONTRACT_DTYPE_approved 0)
set(CONTRACT_DTYPE_reviewed 0)
set(CONTRACT_DTYPE_draft 0)
set(CONTRACT_DTYPE_blocked 0)
foreach(CONTRACT_ROW IN LISTS CUSIGNAL_DTYPE_CONTRACTS)
    string(REPLACE "|" ";" CONTRACT_FIELDS "${CONTRACT_ROW}")
    list(LENGTH CONTRACT_FIELDS CONTRACT_FIELD_COUNT)
    if(NOT CONTRACT_FIELD_COUNT EQUAL 11)
        message(FATAL_ERROR "Malformed dtype contract row: ${CONTRACT_ROW}")
    endif()
    list(GET CONTRACT_FIELDS 0 CONTRACT_OPERATOR_ID)
    list(GET CONTRACT_FIELDS 1 CONTRACT_INPUT_DTYPE)
    list(GET CONTRACT_FIELDS 2 CONTRACT_NATIVE_SUPPORT)
    list(GET CONTRACT_FIELDS 3 CONTRACT_COMPUTE_PATH)
    list(GET CONTRACT_FIELDS 4 CONTRACT_OUTPUT_DTYPE)
    list(GET CONTRACT_FIELDS 5 CONTRACT_SHAPE)
    list(GET CONTRACT_FIELDS 6 CONTRACT_RETURNS)
    list(GET CONTRACT_FIELDS 7 CONTRACT_FP64_FINALIZE)
    list(GET CONTRACT_FIELDS 8 CONTRACT_STATUS)
    list(GET CONTRACT_FIELDS 9 CONTRACT_EVIDENCE_IDS)
    list(GET CONTRACT_FIELDS 10 CONTRACT_EXTENSION_APPROVAL)

    list(FIND CONTRACT_OPERATOR_IDS "${CONTRACT_OPERATOR_ID}" CONTRACT_OPERATOR_INDEX)
    if(CONTRACT_OPERATOR_INDEX EQUAL -1)
        message(FATAL_ERROR
            "Dtype contract references unknown operator: ${CONTRACT_OPERATOR_ID}")
    endif()
    list(FIND CONTRACT_ALLOWED_DTYPES "${CONTRACT_INPUT_DTYPE}" CONTRACT_DTYPE_INDEX)
    if(CONTRACT_DTYPE_INDEX EQUAL -1)
        message(FATAL_ERROR
            "Invalid input dtype for ${CONTRACT_OPERATOR_ID}: ${CONTRACT_INPUT_DTYPE}")
    endif()
    set(CONTRACT_DTYPE_KEY "${CONTRACT_OPERATOR_ID}:${CONTRACT_INPUT_DTYPE}")
    list(FIND CONTRACT_DTYPE_KEYS "${CONTRACT_DTYPE_KEY}" CONTRACT_DUPLICATE_INDEX)
    if(NOT CONTRACT_DUPLICATE_INDEX EQUAL -1)
        message(FATAL_ERROR "Duplicate dtype contract: ${CONTRACT_DTYPE_KEY}")
    endif()
    list(APPEND CONTRACT_DTYPE_KEYS "${CONTRACT_DTYPE_KEY}")

    if(NOT CONTRACT_EVIDENCE_IDS STREQUAL "unresolved" AND
       NOT CONTRACT_EVIDENCE_IDS STREQUAL "")
        string(REPLACE "," ";" CONTRACT_DTYPE_EVIDENCE_IDS
            "${CONTRACT_EVIDENCE_IDS}")
        foreach(CONTRACT_DTYPE_EVIDENCE_ID IN LISTS CONTRACT_DTYPE_EVIDENCE_IDS)
            list(FIND CONTRACT_KNOWN_EVIDENCE_IDS
                "${CONTRACT_DTYPE_EVIDENCE_ID}" CONTRACT_DTYPE_EVIDENCE_INDEX)
            if(CONTRACT_DTYPE_EVIDENCE_INDEX EQUAL -1)
                message(FATAL_ERROR
                    "Dtype contract ${CONTRACT_DTYPE_KEY} references unknown evidence: "
                    "${CONTRACT_DTYPE_EVIDENCE_ID}")
            endif()
            if(CONTRACT_STATUS STREQUAL "approved")
                list(FIND CONTRACT_APPROVED_EVIDENCE_IDS
                    "${CONTRACT_DTYPE_EVIDENCE_ID}"
                    CONTRACT_DTYPE_APPROVED_EVIDENCE_INDEX)
                if(CONTRACT_DTYPE_APPROVED_EVIDENCE_INDEX EQUAL -1)
                    message(FATAL_ERROR
                        "Approved dtype contract ${CONTRACT_DTYPE_KEY} depends on "
                        "non-approved evidence: ${CONTRACT_DTYPE_EVIDENCE_ID}")
                endif()
            endif()
        endforeach()
    endif()

    list(FIND CONTRACT_ALLOWED_STATUSES "${CONTRACT_STATUS}" CONTRACT_STATUS_INDEX)
    if(CONTRACT_STATUS_INDEX EQUAL -1)
        message(FATAL_ERROR
            "Invalid dtype contract status for ${CONTRACT_DTYPE_KEY}: ${CONTRACT_STATUS}")
    endif()
    math(EXPR CONTRACT_DTYPE_${CONTRACT_STATUS}
        "${CONTRACT_DTYPE_${CONTRACT_STATUS}} + 1")

    if(CONTRACT_STATUS STREQUAL "approved")
        foreach(CONTRACT_VALUE
            CONTRACT_NATIVE_SUPPORT CONTRACT_COMPUTE_PATH CONTRACT_OUTPUT_DTYPE
            CONTRACT_SHAPE CONTRACT_RETURNS CONTRACT_FP64_FINALIZE CONTRACT_EVIDENCE_IDS)
            if("${${CONTRACT_VALUE}}" STREQUAL "" OR
               "${${CONTRACT_VALUE}}" STREQUAL "unresolved")
                message(FATAL_ERROR
                    "Approved dtype contract ${CONTRACT_DTYPE_KEY} has unresolved field "
                    "${CONTRACT_VALUE}")
            endif()
        endforeach()
        if(CONTRACT_NATIVE_SUPPORT STREQUAL "unsupported" AND
           (CONTRACT_EXTENSION_APPROVAL STREQUAL "" OR
            CONTRACT_EXTENSION_APPROVAL STREQUAL "not_applicable" OR
            CONTRACT_EXTENSION_APPROVAL STREQUAL "unresolved"))
            message(FATAL_ERROR
                "Approved extension ${CONTRACT_DTYPE_KEY} has no approval record")
        endif()
    endif()
endforeach()

foreach(CONTRACT_OPERATOR_ID IN LISTS CONTRACT_OPERATOR_IDS)
    foreach(CONTRACT_INPUT_DTYPE IN LISTS CONTRACT_ALLOWED_DTYPES)
        set(CONTRACT_DTYPE_KEY "${CONTRACT_OPERATOR_ID}:${CONTRACT_INPUT_DTYPE}")
        list(FIND CONTRACT_DTYPE_KEYS "${CONTRACT_DTYPE_KEY}" CONTRACT_KEY_INDEX)
        if(CONTRACT_KEY_INDEX EQUAL -1)
            message(FATAL_ERROR "Missing dtype contract: ${CONTRACT_DTYPE_KEY}")
        endif()
    endforeach()
endforeach()

# An operator-level approval means all five of its dtype contracts are approved;
# this prevents a summary row from overstating partially reviewed coverage.
foreach(CONTRACT_ROW IN LISTS CUSIGNAL_OPERATOR_CONTRACTS)
    string(REPLACE "|" ";" CONTRACT_FIELDS "${CONTRACT_ROW}")
    list(GET CONTRACT_FIELDS 0 CONTRACT_OPERATOR_ID)
    list(GET CONTRACT_FIELDS 6 CONTRACT_OPERATOR_STATUS)
    if(CONTRACT_OPERATOR_STATUS STREQUAL "approved")
        set(CONTRACT_APPROVED_DTYPE_COUNT 0)
        foreach(CONTRACT_DTYPE_ROW IN LISTS CUSIGNAL_DTYPE_CONTRACTS)
            string(REPLACE "|" ";" CONTRACT_DTYPE_FIELDS "${CONTRACT_DTYPE_ROW}")
            list(GET CONTRACT_DTYPE_FIELDS 0 CONTRACT_DTYPE_OPERATOR_ID)
            list(GET CONTRACT_DTYPE_FIELDS 8 CONTRACT_DTYPE_STATUS)
            if(CONTRACT_DTYPE_OPERATOR_ID STREQUAL CONTRACT_OPERATOR_ID AND
               CONTRACT_DTYPE_STATUS STREQUAL "approved")
                math(EXPR CONTRACT_APPROVED_DTYPE_COUNT
                    "${CONTRACT_APPROVED_DTYPE_COUNT} + 1")
            endif()
        endforeach()
        if(NOT CONTRACT_APPROVED_DTYPE_COUNT EQUAL 5)
            message(FATAL_ERROR
                "Approved operator ${CONTRACT_OPERATOR_ID} must have five approved "
                "dtype rows, actual=${CONTRACT_APPROVED_DTYPE_COUNT}")
        endif()
    endif()
endforeach()

if(CONTRACT_REQUIRE_ALL_APPROVED AND NOT CONTRACT_DTYPE_approved EQUAL 265)
    message(FATAL_ERROR
        "All 265 dtype contracts must be approved for the completion gate; "
        "approved=${CONTRACT_DTYPE_approved}")
endif()

message(STATUS
    "[CONTRACT][EVIDENCE] total=${CONTRACT_EVIDENCE_COUNT} "
    "source_cusignal=${CONTRACT_CUSIGNAL_REFERENCE_ROOT} "
    "source_cupy=${CONTRACT_CUPY_REFERENCE_ROOT}")
message(STATUS
    "[CONTRACT][OPERATORS] total=${CONTRACT_OPERATOR_COUNT} "
    "approved=${CONTRACT_OPERATOR_approved} reviewed=${CONTRACT_OPERATOR_reviewed} "
    "draft=${CONTRACT_OPERATOR_draft} blocked=${CONTRACT_OPERATOR_blocked}")
message(STATUS
    "[CONTRACT][DTYPES] total=${CONTRACT_DTYPE_COUNT} "
    "approved=${CONTRACT_DTYPE_approved} reviewed=${CONTRACT_DTYPE_reviewed} "
    "draft=${CONTRACT_DTYPE_draft} blocked=${CONTRACT_DTYPE_blocked}")
