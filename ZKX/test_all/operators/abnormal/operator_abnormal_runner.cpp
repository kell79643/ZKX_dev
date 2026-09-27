#include "abnormal_runtime.h"
#include <cusignal/runtime/operator_abnormal_validation.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace fs = std::filesystem;
using Row = std::map<std::string, std::string>;

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

struct Cli {
    fs::path catalog;
    fs::path normal_links;
    fs::path output;
    std::string timestamp;
    std::string run_id;
    std::string module;
    std::string op;
    std::string case_id;
    std::string backend;
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
    return {get("catalog"), get("normal-links"), get("output"), get("timestamp"),
        get("run-id"), get("module"), get("operator"), get("case-id"), get("backend")};
}

std::string key(const std::string& module, const std::string& op)
{
    return module + "\n" + op;
}

bool selected(const Cli& cli, const Row& row)
{
    return (cli.module == "all" || cli.module == required(row, "module_name")) &&
        (cli.op == "all" || cli.op == required(row, "operator_name")) &&
        (cli.case_id == "all" || cli.case_id == required(row, "abnormal_case_id")) &&
        (cli.backend == "all" || cli.backend == required(row, "applicable_backend"));
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Cli cli = parse_cli(argc, argv);
        const auto catalog = read_csv(cli.catalog);
        const auto links = read_csv(cli.normal_links);
        std::map<std::string, std::pair<std::string, std::string>> normal;
        for (const auto& row : links) {
            const std::string identity = key(required(row, "module_name"), required(row, "operator_name"));
            if (!normal.emplace(identity, std::make_pair(required(row, "normal_run_id"),
                    required(row, "normal_case_id"))).second)
                throw std::runtime_error("正常结果外键重复：" + identity);
        }

        std::vector<zkx::abnormal_input::CaseObservation> results;
        for (const auto& row : catalog) {
            if (!selected(cli, row)) continue;
            const std::string module = required(row, "module_name");
            const std::string op = required(row, "operator_name");
            const auto link = normal.find(key(module, op));
            if (link == normal.end())
                throw std::runtime_error("缺少正常结果外键：" + module + "::" + op);
            const auto expected = cusignal::parse_error_code(required(row, "expected_code"));
            const std::string input = "mutation=" + required(row, "mutation");
            zkx::abnormal_input::CaseSpec spec{cli.timestamp, cli.run_id,
                required(row, "applicable_backend"), module, op,
                required(row, "abnormal_case_id"), link->second.first, link->second.second,
                required(row, "abnormal_type"), input, expected};
            cusignal::OperatorAbnormalRequest request{module, op, spec.case_id,
                spec.abnormal_type, input, expected, spec.backend};
            results.push_back(zkx::abnormal_input::execute_case(std::move(spec),
                [request = std::move(request)](bool&) {
                    cusignal::validate_operator_abnormal_input(request);
                }));
        }
        if (results.empty()) throw std::runtime_error("选择器没有匹配任何异常case");
        zkx::abnormal_input::write_results(cli.output, results);
        zkx::abnormal_input::print_table(results);
        for (const auto& result : results) if (!result.passed) return 1;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "abnormal_input_operator_abnormal配置/执行失败：" << error.what() << '\n';
        return 2;
    }
}
