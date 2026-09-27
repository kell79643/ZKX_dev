if(NOT DEFINED TYPE_COVERAGE_EXECUTABLE OR NOT DEFINED TYPE_COVERAGE_EXPECTED_RECORDS)
    message(FATAL_ERROR "TYPE_COVERAGE_EXECUTABLE and TYPE_COVERAGE_EXPECTED_RECORDS are required")
endif()

execute_process(
    COMMAND "${TYPE_COVERAGE_EXECUTABLE}"
    RESULT_VARIABLE TYPE_COVERAGE_RESULT
    OUTPUT_VARIABLE TYPE_COVERAGE_OUTPUT
    ERROR_VARIABLE TYPE_COVERAGE_ERROR
)

message("${TYPE_COVERAGE_OUTPUT}")
if(NOT "${TYPE_COVERAGE_ERROR}" STREQUAL "")
    message("${TYPE_COVERAGE_ERROR}")
endif()

if(NOT TYPE_COVERAGE_RESULT EQUAL 0)
    message(FATAL_ERROR "type coverage business executable failed with exit code ${TYPE_COVERAGE_RESULT}")
endif()

string(REGEX MATCHALL "\\[type coverage(-DETAIL)?\\]\\[[^]]+\\]\\[[^]]+\\]" TYPE_COVERAGE_RECORDS "${TYPE_COVERAGE_OUTPUT}")
list(LENGTH TYPE_COVERAGE_RECORDS TYPE_COVERAGE_RECORD_COUNT)
if(NOT TYPE_COVERAGE_RECORD_COUNT EQUAL TYPE_COVERAGE_EXPECTED_RECORDS)
    message(FATAL_ERROR
        "type coverage record count mismatch: expected=${TYPE_COVERAGE_EXPECTED_RECORDS} actual=${TYPE_COVERAGE_RECORD_COUNT}")
endif()

string(REGEX MATCHALL "\\[type coverage(-DETAIL)?\\][^\n\r]*pass=0" TYPE_COVERAGE_FAILED_RECORDS "${TYPE_COVERAGE_OUTPUT}")
list(LENGTH TYPE_COVERAGE_FAILED_RECORDS TYPE_COVERAGE_FAILED_COUNT)
if(NOT TYPE_COVERAGE_FAILED_COUNT EQUAL 0)
    message(FATAL_ERROR "type coverage contains ${TYPE_COVERAGE_FAILED_COUNT} failed structured records")
endif()

if(TYPE_COVERAGE_REQUIRE_DETAIL)
    string(REGEX MATCHALL "\\[type coverage-DETAIL\\]\\[[^]]+\\]\\[[^]]+\\]" TYPE_COVERAGE_DETAIL_RECORDS
        "${TYPE_COVERAGE_OUTPUT}")
    list(LENGTH TYPE_COVERAGE_DETAIL_RECORDS TYPE_COVERAGE_DETAIL_COUNT)
    if(NOT TYPE_COVERAGE_DETAIL_COUNT EQUAL TYPE_COVERAGE_EXPECTED_RECORDS)
        message(FATAL_ERROR
            "type coverage detailed record count mismatch: expected=${TYPE_COVERAGE_EXPECTED_RECORDS} actual=${TYPE_COVERAGE_DETAIL_COUNT}")
    endif()

    string(REPLACE "\r\n" "\n" TYPE_COVERAGE_NORMALIZED_OUTPUT "${TYPE_COVERAGE_OUTPUT}")
    string(REPLACE "\r" "\n" TYPE_COVERAGE_NORMALIZED_OUTPUT "${TYPE_COVERAGE_NORMALIZED_OUTPUT}")
    # CMake 列表同样使用分号分隔；先保护参数字段中的分号，再按换行逐条校验。
    string(REPLACE ";" "__TYPE_COVERAGE_SEMICOLON__" TYPE_COVERAGE_NORMALIZED_OUTPUT "${TYPE_COVERAGE_NORMALIZED_OUTPUT}")
    string(REPLACE "\n" ";" TYPE_COVERAGE_OUTPUT_LINES "${TYPE_COVERAGE_NORMALIZED_OUTPUT}")
    set(TYPE_COVERAGE_REQUIRED_DETAIL_FIELDS
        "case=" "InputT=" "ComputeT=" "LibraryT=" "OutputT="
        "input_shape=" "output_shape=" "params=" "seed="
        "cpu=" "gpu=" "backend=" "compared=" "metric=" "metric_value="
        "tolerance=" "full_element_review=" "alternate_case_review="
        "mutation_review=" "high_precision_review=" "minimum_legal="
        "typical=" "boundary=" "invalid=" "zero_review_pass="
        "failed_index=" "commit=" "test_time=" "pass=")
    foreach(TYPE_COVERAGE_LINE IN LISTS TYPE_COVERAGE_OUTPUT_LINES)
        if(TYPE_COVERAGE_LINE MATCHES "^\\[type coverage-DETAIL\\]")
            foreach(TYPE_COVERAGE_FIELD IN LISTS TYPE_COVERAGE_REQUIRED_DETAIL_FIELDS)
                string(FIND "${TYPE_COVERAGE_LINE}" "${TYPE_COVERAGE_FIELD}" TYPE_COVERAGE_FIELD_POSITION)
                if(TYPE_COVERAGE_FIELD_POSITION EQUAL -1)
                    message(FATAL_ERROR
                        "type coverage detailed record missing field '${TYPE_COVERAGE_FIELD}': ${TYPE_COVERAGE_LINE}")
                endif()
            endforeach()
            if(TYPE_COVERAGE_LINE MATCHES "(full_element_review|alternate_case_review|mutation_review|high_precision_review|minimum_legal|typical|boundary|invalid|zero_review_pass|pass)=0")
                message(FATAL_ERROR "type coverage detailed review failed: ${TYPE_COVERAGE_LINE}")
            endif()
            if(TYPE_COVERAGE_LINE MATCHES "commit=unbound" OR TYPE_COVERAGE_LINE MATCHES "test_time=unbound")
                message(FATAL_ERROR "type coverage detailed record is not bound to source/time: ${TYPE_COVERAGE_LINE}")
            endif()
        endif()
    endforeach()
endif()

message("[TYPE_COVERAGE][GROUP-PASS] records=${TYPE_COVERAGE_RECORD_COUNT} executable=${TYPE_COVERAGE_EXECUTABLE}")
