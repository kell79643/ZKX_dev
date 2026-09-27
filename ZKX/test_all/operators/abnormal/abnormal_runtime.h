#pragma once

#include <cusignal/runtime/errors.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace zkx::abnormal_input {

struct CaseSpec {
    std::string timestamp;
    std::string run_id;
    std::string backend;
    std::string module_name;
    std::string operator_name;
    std::string case_id;
    std::string normal_run_id;
    std::string normal_case_id;
    std::string abnormal_type;
    std::string input_summary;
    cusignal::ErrorCode expected_code{cusignal::ErrorCode::INTERNAL_ERROR};
};

struct CaseObservation {
    CaseSpec spec;
    cusignal::ErrorCode captured_code{cusignal::ErrorCode::SUCCESS};
    cusignal::Error returned_error;
    bool action_completed{};
    bool output_consumed{};
    bool passed{};
};

inline CaseObservation execute_case(
    CaseSpec spec, const std::function<void(bool&)>& action)
{
    CaseObservation result;
    result.spec = std::move(spec);
    result.returned_error = {cusignal::ErrorCode::SUCCESS,
        "异常输入未触发错误", {}, false};
    try {
        action(result.output_consumed);
        result.action_completed = true;
    } catch (const cusignal::ProjectError& caught) {
        result.captured_code = caught.detail().code;
        result.returned_error = caught.detail();
    } catch (const std::exception& caught) {
        result.captured_code = cusignal::ErrorCode::INTERNAL_ERROR;
        result.returned_error = {cusignal::ErrorCode::INTERNAL_ERROR,
            std::string("未映射的标准异常：") + caught.what(), {}, true};
    } catch (...) {
        result.captured_code = cusignal::ErrorCode::INTERNAL_ERROR;
        result.returned_error = {cusignal::ErrorCode::INTERNAL_ERROR,
            "未映射的非标准异常", {}, true};
    }
    result.passed = !result.action_completed && !result.output_consumed &&
        result.captured_code == result.spec.expected_code &&
        result.returned_error.code == result.spec.expected_code &&
        !result.returned_error.message.empty() && result.returned_error.safe_exit;
    return result;
}

inline std::string csv_escape(const std::string& value)
{
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string output{"\""};
    for (char character : value) {
        if (character == '"') output += "\"\"";
        else output += character;
    }
    return output + '"';
}

inline void write_results(const std::filesystem::path& output,
                          const std::vector<CaseObservation>& results)
{
    if (std::filesystem::exists(output))
        throw std::runtime_error("拒绝覆盖已有异常结果：" + output.string());
    std::filesystem::create_directories(output.parent_path());
    std::ofstream csv(output);
    if (!csv) throw std::runtime_error("无法创建异常结果CSV：" + output.string());
    csv << "timestamp,run_id,backend,test_object,task_name,step_name,module_name,operator_name,case_id,normal_run_id,normal_case_id,abnormal_type,input_summary,expected_code,captured_code,returned_code,actual_code,error_message,backend_error_code,backend_error_name,backend_error_message,safe_exit,output_consumed,status\n";
    for (const auto& result : results) {
        const auto& spec = result.spec;
        const std::vector<std::string> row = {spec.timestamp, spec.run_id, spec.backend,
            "operator", "NA", "NA", spec.module_name, spec.operator_name, spec.case_id,
            spec.normal_run_id, spec.normal_case_id, spec.abnormal_type, spec.input_summary,
            cusignal::error_code_name(spec.expected_code),
            cusignal::error_code_name(result.captured_code),
            cusignal::error_code_name(result.returned_error.code),
            cusignal::error_code_name(result.returned_error.code),
            result.returned_error.message, result.returned_error.backend.code,
            result.returned_error.backend.name, result.returned_error.backend.message,
            result.returned_error.safe_exit ? "true" : "false",
            result.output_consumed ? "true" : "false", result.passed ? "PASS" : "FAIL"};
        for (std::size_t index = 0; index < row.size(); ++index) {
            if (index != 0) csv << ',';
            csv << csv_escape(row[index]);
        }
        csv << '\n';
    }
}

inline void print_table(const std::vector<CaseObservation>& results)
{
    std::cout << std::left
              << std::setw(34) << "module/operator"
              << std::setw(24) << "abnormal_type"
              << std::setw(25) << "expected_code"
              << std::setw(25) << "captured_code"
              << std::setw(25) << "returned_code"
              << std::setw(11) << "safe_exit"
              << std::setw(8) << "status" << '\n';
    std::cout << std::string(152, '-') << '\n';
    for (const auto& result : results) {
        std::cout << std::left
                  << std::setw(34) << (result.spec.module_name + "/" + result.spec.operator_name)
                  << std::setw(24) << result.spec.abnormal_type
                  << std::setw(25) << cusignal::error_code_name(result.spec.expected_code)
                  << std::setw(25) << cusignal::error_code_name(result.captured_code)
                  << std::setw(25) << cusignal::error_code_name(result.returned_error.code)
                  << std::setw(11) << (result.returned_error.safe_exit ? "true" : "false")
                  << std::setw(8) << (result.passed ? "PASS" : "FAIL") << '\n';
        std::cout << "  case_id: " << result.spec.case_id << '\n'
                  << "  abnormal_input: " << result.spec.input_summary << '\n'
                  << "  error_message: " << result.returned_error.message << '\n'
                  << "  backend_error: code=" << result.returned_error.backend.code
                  << " name=" << result.returned_error.backend.name
                  << " message=" << result.returned_error.backend.message << '\n';
    }
    std::size_t passed = 0;
    for (const auto& result : results) passed += result.passed ? 1U : 0U;
    std::cout << "[ABNORMAL_INPUT][SUMMARY] total=" << results.size()
              << " pass=" << passed << " fail=" << (results.size() - passed)
              << " status=" << (passed == results.size() ? "PASS" : "FAIL") << '\n';
}

}  // namespace zkx::abnormal_input
