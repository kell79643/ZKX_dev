#include <task1/task1_abnormal_validation.h>
#include <task1/task1_config.h>

#include <filesystem>
#include <fstream>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace fs = std::filesystem;
using Row = std::map<std::string, std::string>;

std::string compiled_backend()
{
#if defined(USE_DLFFT)
    return "dlfft";
#elif defined(USE_THRUST)
    return "fft_thrust";
#else
    return "unbound";
#endif
}

std::vector<std::string> parse_csv_line(const std::string& line)
{
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t index = 0; index < line.size(); ++index) {
        const char character = line[index];
        if (quoted) {
            if (character == '"' && index + 1 < line.size() && line[index + 1] == '"') {
                field += '"';
                ++index;
            } else if (character == '"') quoted = false;
            else field += character;
        } else if (character == '"') quoted = true;
        else if (character == ',') {
            fields.push_back(std::move(field));
            field.clear();
        } else field += character;
    }
    if (quoted) throw std::runtime_error("CSV存在未闭合引号");
    fields.push_back(std::move(field));
    return fields;
}

std::vector<Row> read_csv(const fs::path& path)
{
    std::ifstream input(path);
    if (!input) throw std::runtime_error("无法打开CSV：" + path.string());
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("CSV为空：" + path.string());
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto header = parse_csv_line(line);
    std::vector<Row> rows;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const auto fields = parse_csv_line(line);
        if (fields.size() != header.size())
            throw std::runtime_error("CSV列数不一致：" + path.string());
        Row row;
        for (std::size_t index = 0; index < header.size(); ++index)
            row.emplace(header[index], fields[index]);
        rows.push_back(std::move(row));
    }
    return rows;
}

const std::string& required(const Row& row, const char* key)
{
    const auto found = row.find(key);
    if (found == row.end() || found->second.empty())
        throw std::runtime_error(std::string("CSV缺少必填字段：") + key);
    return found->second;
}

std::string csv_escape(const std::string& value)
{
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string output = "\"";
    for (const char character : value) {
        if (character == '"') output += '"';
        output += character;
    }
    return output + '"';
}

std::string safe_name(const std::string& value)
{
    std::string output;
    for (const char character : value)
        output += (std::isalnum(static_cast<unsigned char>(character)) ||
                   character == '_' || character == '-') ? character : '_';
    return output;
}

struct Cli {
    fs::path catalog;
    fs::path fixture_root;
    fs::path output_dir;
    std::string timestamp;
    std::string run_id;
    std::string backend;
    std::string case_id;
};

Cli parse_cli(int argc, char** argv)
{
    std::map<std::string, std::string> values;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc || std::string(argv[index]).rfind("--", 0) != 0)
            throw std::invalid_argument("参数必须使用--key value成对提供");
        values.emplace(std::string(argv[index]).substr(2), argv[index + 1]);
    }
    auto get = [&](const char* key) -> const std::string& {
        const auto found = values.find(key);
        if (found == values.end() || found->second.empty())
            throw std::invalid_argument(std::string("缺少参数：--") + key);
        return found->second;
    };
    return {get("catalog"), get("fixture-root"), get("output-dir"),
        get("timestamp"), get("run-id"), get("backend"), get("case-id")};
}

struct Observation {
    Row row;
    cusignal::ErrorCode expected{cusignal::ErrorCode::INTERNAL_ERROR};
    cusignal::Error actual;
    bool action_completed{};
    bool output_consumed{};
    bool passed{};
};

Observation execute(const Cli& cli, Row row)
{
    Observation result;
    result.expected = cusignal::parse_error_code(required(row, "expected_code"));
    try {
        const std::string action = required(row, "action");
        if (action == "load_config") {
            (void)task1::TaskConfig::load(cli.fixture_root / required(row, "fixture_path"));
        } else if (action == "missing_upstream") {
            task1::require_task1_upstream(false, required(row, "step_name"), "Step1.echo");
        } else if (action == "validate_dtype") {
            task1::validate_task1_dtype(required(row, "requested_dtype"));
        } else if (action == "check_resource") {
            (void)task1::checked_task1_input_elements(
                std::stoi(required(row, "num_pulses")),
                std::stoi(required(row, "samples_per_pulse")));
        } else if (action == "validate_backend") {
            task1::validate_task1_backend(required(row, "requested_backend"), cli.backend);
        } else {
            cusignal::throw_error(cusignal::ErrorCode::CONFIG_ERROR,
                "未知Task1异常fixture action：" + action);
        }
        result.action_completed = true;
        result.actual = {cusignal::ErrorCode::SUCCESS,
            "异常fixture意外执行成功", {}, false};
    } catch (const cusignal::ProjectError& error) {
        result.actual = error.detail();
    } catch (const std::exception& error) {
        result.actual = {cusignal::ErrorCode::INTERNAL_ERROR,
            "Task1异常runner捕获非项目异常：" + std::string(error.what()), {}, true};
    }
    result.passed = !result.action_completed && !result.output_consumed &&
        result.actual.code == result.expected && result.actual.code != cusignal::ErrorCode::SUCCESS &&
        !result.actual.message.empty() && result.actual.safe_exit;
    result.row = std::move(row);
    return result;
}

void write_results(const Cli& cli, const std::vector<Observation>& results)
{
    fs::create_directories(cli.output_dir);
    const fs::path output = cli.output_dir /
        ("task_abnormal_cases_" + safe_name(cli.backend) + ".csv");
    if (fs::exists(output)) throw std::runtime_error("拒绝覆盖已有结果：" + output.string());
    std::ofstream csv(output);
    if (!csv) throw std::runtime_error("无法创建结果：" + output.string());
    csv << "timestamp,run_id,backend,test_object,task_name,step_name,case_id,abnormal_type,input_summary,expected_code,captured_code,returned_code,actual_code,error_message,backend_error_code,backend_error_name,backend_error_message,safe_exit,output_consumed,status\n";
    for (const auto& result : results) {
        const auto& row = result.row;
        const std::vector<std::string> values{
            cli.timestamp, cli.run_id, cli.backend, "task", "Task1",
            required(row, "step_name"), required(row, "case_id"),
            required(row, "abnormal_type"), required(row, "input_summary"),
            cusignal::error_code_name(result.expected),
            cusignal::error_code_name(result.actual.code),
            cusignal::error_code_name(result.actual.code),
            cusignal::error_code_name(result.actual.code), result.actual.message,
            result.actual.backend.code, result.actual.backend.name,
            result.actual.backend.message,
            result.actual.safe_exit ? "true" : "false",
            result.output_consumed ? "true" : "false",
            result.passed ? "PASS" : "FAIL"};
        for (std::size_t index = 0; index < values.size(); ++index) {
            if (index) csv << ',';
            csv << csv_escape(values[index]);
        }
        csv << '\n';
    }
    std::cout << "[TASK1][ABNORMAL][CSV] path=" << output.string()
              << " rows=" << results.size() << '\n';
}

void print_results(const std::vector<Observation>& results)
{
    std::cout << std::left << std::setw(18) << "task/step"
              << std::setw(24) << "abnormal_type"
              << std::setw(24) << "expected_code"
              << std::setw(24) << "captured_code"
              << std::setw(24) << "returned_code"
              << std::setw(11) << "safe_exit" << "status\n"
              << std::string(133, '-') << '\n';
    int passed = 0;
    for (const auto& result : results) {
        const auto code = cusignal::error_code_name(result.actual.code);
        std::cout << std::setw(18) << ("Task1/" + required(result.row, "step_name"))
                  << std::setw(24) << required(result.row, "abnormal_type")
                  << std::setw(24) << cusignal::error_code_name(result.expected)
                  << std::setw(24) << code << std::setw(24) << code
                  << std::setw(11) << (result.actual.safe_exit ? "true" : "false")
                  << (result.passed ? "PASS" : "FAIL") << '\n'
                  << "  case_id: " << required(result.row, "case_id") << '\n'
                  << "  abnormal_input: " << required(result.row, "input_summary") << '\n'
                  << "  error_message: " << result.actual.message << '\n'
                  << "  backend_error: code=" << result.actual.backend.code
                  << " name=" << result.actual.backend.name
                  << " message=" << result.actual.backend.message << '\n'
                  << "  output_consumed: " << (result.output_consumed ? "true" : "false") << '\n';
        if (result.passed) ++passed;
    }
    std::cout << "[TASK1][ABNORMAL][SUMMARY] total=" << results.size()
              << " pass=" << passed << " fail=" << results.size() - passed
              << " status=" << (passed == static_cast<int>(results.size()) ? "PASS" : "FAIL")
              << '\n';
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Cli cli = parse_cli(argc, argv);
        if (cli.backend != compiled_backend())
            throw std::runtime_error("--backend与编译后端不一致：requested=" +
                cli.backend + " compiled=" + compiled_backend());
        std::vector<Observation> results;
        for (auto row : read_csv(cli.catalog)) {
            if (cli.case_id != "all" && required(row, "case_id") != cli.case_id) continue;
            results.push_back(execute(cli, std::move(row)));
        }
        if (results.empty()) throw std::runtime_error("case选择器没有匹配fixture");
        write_results(cli, results);
        print_results(results);
        for (const auto& result : results) if (!result.passed) return 1;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Task1异常runner配置/执行失败：" << error.what() << '\n';
        return 2;
    }
}
