#pragma once

#include <demo/config_file.h>
#include "test_all/support/config/json_value.h"

#include <cuda_runtime.h>
#include <malloc.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace zkx::stability {

using zkx::config::JsonValue;

inline const JsonValue& field(const JsonValue& value, const char* name)
{
    const auto* found = value.find(name);
    if (!found) throw std::invalid_argument(std::string("missing field: ") + name);
    return *found;
}

inline int integer(const JsonValue& value, const char* name)
{
    const double number = field(value, name).as_number();
    if (!std::isfinite(number) || std::floor(number) != number)
        throw std::invalid_argument(std::string(name) + " must be an integer");
    return static_cast<int>(number);
}

inline std::uint32_t uint32(const JsonValue& value, const char* name)
{
    const double number = field(value, name).as_number();
    if (!std::isfinite(number) || std::floor(number) != number ||
        number < 0.0 || number > 4294967295.0)
        throw std::invalid_argument(std::string(name) + " must be uint32");
    return static_cast<std::uint32_t>(number);
}

inline float f32(const JsonValue& value, const char* name)
{
    const double number = field(value, name).as_number();
    if (!std::isfinite(number))
        throw std::invalid_argument(std::string(name) + " must be finite");
    return static_cast<float>(number);
}

struct PairedCase {
    int index{};
    std::string case_id;
    std::uint32_t seed{};
    std::string summary;
    JsonValue task;
    std::string parameter_set_id;
    int total_cases{};
    std::string file_sha256;
};

inline std::string file_sha256(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read perturbation file: " + path.string());
    const std::vector<unsigned char> bytes{
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    return demo_config::sha256(bytes);
}

inline PairedCase load_case(
    const std::filesystem::path& path, int wanted, const char* task_name)
{
    const auto root = JsonValue::parse_file(path.string());
    if (integer(root, "schema_version") != 1 ||
        field(root, "coverage").as_string() != "formal_1000_paired" ||
        integer(root, "stability_total_cases") != 1000)
        throw std::invalid_argument("stability parameter file is not formal paired-1000 coverage");
    const auto& cases = field(root, "cases").as_array();
    if (cases.size() != 1000)
        throw std::invalid_argument("stability parameter file must contain exactly 1000 cases");
    if (wanted < 0 || wanted >= 1000)
        throw std::invalid_argument("stability_case_index must be in 0..999");
    const auto& item = cases[static_cast<std::size_t>(wanted)];
    if (integer(item, "stability_case_index") != wanted)
        throw std::invalid_argument("stability case order/index mismatch");
    PairedCase result;
    result.index = wanted;
    result.case_id = field(item, "case_id").as_string();
    result.seed = uint32(item, "seed");
    result.summary = field(item, "perturbation_summary").as_string();
    result.task = field(item, task_name);
    result.parameter_set_id = field(root, "parameter_set_id").as_string();
    result.total_cases = 1000;
    result.file_sha256 = file_sha256(path);
    return result;
}

class XorShift32 {
public:
    explicit XorShift32(std::uint32_t seed) : state_(seed ? seed : 0x6d2b79f5U) {}
    std::uint32_t next()
    {
        std::uint32_t value = state_;
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;
        return state_ = value;
    }
    float symmetric(float amplitude)
    {
        const double unit = static_cast<double>(next()) / 4294967295.0;
        return static_cast<float>((2.0 * unit - 1.0) * amplitude);
    }
private:
    std::uint32_t state_;
};

struct ResourceSample {
    std::int64_t heap{};
    std::int64_t rss{};
    std::int64_t gpu{};
    bool heap_ok{};
    bool rss_ok{};
    bool gpu_ok{};
};

inline ResourceSample sample_resources()
{
    ResourceSample result;
#if defined(__GLIBC__)
    const auto info = mallinfo2();
    result.heap = static_cast<std::int64_t>(info.uordblks);
    result.heap_ok = true;
#endif
    std::ifstream statm("/proc/self/statm");
    std::int64_t total_pages = 0, resident_pages = 0;
    if (statm >> total_pages >> resident_pages) {
        result.rss = resident_pages * static_cast<std::int64_t>(::sysconf(_SC_PAGESIZE));
        result.rss_ok = true;
    }
    std::size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        result.gpu = static_cast<std::int64_t>(total_bytes - free_bytes);
        result.gpu_ok = true;
    }
    return result;
}

struct Measurement {
    double milliseconds{};
    ResourceSample before;
    ResourceSample after;
    ResourceSample peak;
};

template <class Function>
auto measure(Function&& function, bool synchronize)
{
    using Result = decltype(function());
    Measurement measurement;
    measurement.before = sample_resources();
    const auto begin = std::chrono::steady_clock::now();
    Result result = function();
    if (synchronize && cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("cudaDeviceSynchronize failed");
    const auto end = std::chrono::steady_clock::now();
    measurement.after = sample_resources();
    measurement.milliseconds =
        std::chrono::duration<double, std::milli>(end - begin).count();
    measurement.peak = measurement.after;
    measurement.peak.heap = std::max(measurement.before.heap, measurement.after.heap);
    measurement.peak.rss = std::max(measurement.before.rss, measurement.after.rss);
    measurement.peak.gpu = std::max(measurement.before.gpu, measurement.after.gpu);
    return std::pair<Result, Measurement>{std::move(result), measurement};
}

inline std::string utc_now()
{
    const std::time_t value = std::time(nullptr);
    std::tm result{};
    gmtime_r(&value, &result);
    std::ostringstream output;
    output << std::put_time(&result, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

inline std::string number(double value)
{
    if (!std::isfinite(value)) return "NA";
    std::ostringstream output;
    output << std::scientific << std::setprecision(17) << value;
    return output.str();
}

inline std::string csv_escape(const std::string& value)
{
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string output = "\"";
    for (const char character : value) {
        if (character == '"') output.push_back('"');
        output.push_back(character);
    }
    output.push_back('"');
    return output;
}

inline void write_csv_row(std::ostream& output, const std::vector<std::string>& values)
{
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) output << ',';
        output << csv_escape(values[index]);
    }
    output << '\n';
}

inline const std::vector<std::string>& identity_columns()
{
    static const std::vector<std::string> value{
        "schema_version", "run_id", "run_type", "test_time_utc", "git_commit",
        "git_dirty", "config_id", "config_sha256", "target_kind", "target",
        "task_name", "step_name", "operator_name", "case_id", "scale_id",
        "order_of_magnitude", "actual_elements", "input_shape", "dtype", "device",
        "backend", "timing_scope", "status", "error_code"};
    return value;
}

inline const std::vector<std::string>& common_tail_columns()
{
    static const std::vector<std::string> value{
        "warmup_runs", "measured_runs", "mean_ms", "p50_ms", "p95_ms", "p99_ms",
        "min_ms", "max_ms", "std_ms", "cv", "cpu_gpu_speedup",
        "accuracy_reference", "mse", "rmse", "relative_l2", "relative_linf",
        "exact_match", "mismatch_count", "semantic_check", "accuracy_status",
        "accuracy_threshold_set_id", "accuracy_thresholds_sha256",
        "cpu_heap_before_bytes", "cpu_heap_after_bytes", "cpu_heap_peak_bytes",
        "cpu_heap_delta_bytes", "rss_before_bytes", "rss_after_bytes", "rss_peak_bytes",
        "rss_delta_bytes", "gpu_before_bytes", "gpu_after_bytes", "gpu_peak_bytes",
        "gpu_delta_bytes", "return_code", "stability_case_index",
        "stability_total_cases", "perturbation_file", "perturbation_sha256",
        "perturbation_summary", "input_mode", "input_source_file",
        "input_source_sha256", "selection_seed", "profile_entry_id"};
    return value;
}

inline std::vector<std::string> standard_header()
{
    auto value = identity_columns();
    const auto& tail = common_tail_columns();
    value.insert(value.end(), tail.begin(), tail.end());
    return value;
}

inline std::string resource_value(bool available, std::int64_t value)
{
    return available ? std::to_string(value) : "NA";
}

}  // namespace zkx::stability
