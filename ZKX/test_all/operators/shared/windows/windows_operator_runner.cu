#include "accuracy.h"
#include "csv_writer.h"
#include <cusignal/runtime/cuda_utils.h>
#include "identity.h"
#include "json_value.h"
#include "memory.h"
#include "sha256.h"
#include "statistics.h"
#include "terminal_table.h"
#include <cusignal/operators/windows/windows_typed.h>

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef ZKX_REMAINING_OPERATOR_CASE_WINDOW_OPERATOR
#error "ZKX_REMAINING_OPERATOR_CASE_WINDOW_OPERATOR must name exactly one Remaining operator case windows operator"
#endif
#ifndef ZKX_REMAINING_OPERATOR_CASE_WINDOW_BATCH
#define ZKX_REMAINING_OPERATOR_CASE_WINDOW_BATCH "b01"
#endif

namespace {

using zkx::config::JsonValue;
namespace fs = std::filesystem;

constexpr const char* kOperator = ZKX_REMAINING_OPERATOR_CASE_WINDOW_OPERATOR;
constexpr const char* kBatch = ZKX_REMAINING_OPERATOR_CASE_WINDOW_BATCH;
constexpr const char* kModule = "windows";
#ifdef ZKX_REMAINING_OPERATOR_CASE_WINDOW_FFT_BACKEND
#ifndef ZKX_SELECTED_FFT_BACKEND_NAME
#error "ZKX_SELECTED_FFT_BACKEND_NAME is required for FFT-backed windows operators"
#endif
constexpr const char* kBackend = ZKX_SELECTED_FFT_BACKEND_NAME;
#else
constexpr const char* kBackend = "not_applicable";
#endif
constexpr const char* kTimingScope = "compatibility/end-to-end";

struct Cli {
    fs::path config_file;
    fs::path thresholds_file;
    fs::path results_root;
    std::string scale_id;
    std::string run_type;
    std::string git_commit;
    std::string git_dirty;
    std::string test_time_utc;
    std::string random_hex8;
    bool symmetric = true;
};

struct InputSpec {
    std::string name;
    std::vector<int> shape;
    std::size_t elements = 0;
    std::string dtype;
    std::string role;
};

struct ScaleConfig {
    std::string scale_id;
    std::string order_of_magnitude;
    std::size_t actual_elements = 0;
    int length = 0;
    std::string dtype;
    std::vector<InputSpec> inputs;
    std::map<std::string, double> number_parameters;
    std::map<std::string, std::vector<double>> array_parameters;
    std::map<std::string, bool> bool_parameters;
    std::map<std::string, std::string> string_parameters;
    std::vector<bool> sym_values;
    std::string mapping_reason;
};

struct OperatorConfig {
    std::string config_id;
    std::string config_sha256;
    std::vector<ScaleConfig> scales;
};

struct Thresholds {
    double mse_max = 0.0;
    double rmse_max = 0.0;
    double relative_l2_max = 0.0;
    double relative_linf_max = 0.0;
    double relative_floor = 0.0;
    std::string semantic_rule;
};

struct PhaseSample {
    std::string phase;
    int phase_index = 0;
    zkx::common::ResourceSample cpu_heap;
    zkx::common::ResourceSample rss;
    zkx::common::ResourceSample gpu;
};

struct ResourceSummary {
    bool available = false;
    std::int64_t before = 0;
    std::int64_t after = 0;
    std::int64_t peak = 0;
    std::int64_t delta = 0;
};

std::string read_file(const fs::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open file: " + path.string());
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

const JsonValue& require_member(const JsonValue& object, const std::string& key)
{
    const JsonValue* value = object.find(key);
    if (value == nullptr) throw std::invalid_argument("missing JSON member: " + key);
    return *value;
}

void require_keys(const JsonValue& value, const std::set<std::string>& allowed,
                  const std::string& context)
{
    if (!value.is_object()) throw std::invalid_argument(context + " must be object");
    for (const auto& item : value.as_object()) {
        if (allowed.count(item.first) == 0)
            throw std::invalid_argument(context + " has unknown member: " + item.first);
    }
}

std::size_t exact_size(const JsonValue& value, const std::string& label)
{
    if (!value.is_number()) throw std::invalid_argument(label + " must be number");
    const double number = value.as_number();
    if (!std::isfinite(number) || number < 0.0 || std::floor(number) != number)
        throw std::invalid_argument(label + " must be nonnegative integer");
    return static_cast<std::size_t>(number);
}

std::string string_value(const JsonValue& value, const std::string& label)
{
    if (!value.is_string()) throw std::invalid_argument(label + " must be string");
    return value.as_string();
}

std::size_t shape_elements(const std::vector<int>& shape)
{
    std::size_t result = 1;
    for (int extent : shape) {
        if (extent < 1) throw std::invalid_argument("input shape extent must be positive");
        result *= static_cast<std::size_t>(extent);
    }
    return result;
}

std::vector<int> parse_shape(const JsonValue& value)
{
    if (!value.is_array() || value.as_array().empty())
        throw std::invalid_argument("input shape must be nonempty array");
    std::vector<int> shape;
    for (const auto& extent : value.as_array())
        shape.push_back(static_cast<int>(exact_size(extent, "shape extent")));
    return shape;
}

std::vector<bool> parse_bool_array(const JsonValue& value, const std::string& label)
{
    if (!value.is_array()) throw std::invalid_argument(label + " must be array");
    std::vector<bool> result;
    for (const auto& item : value.as_array()) {
        if (!item.is_boolean()) throw std::invalid_argument(label + " must contain booleans");
        result.push_back(item.as_boolean());
    }
    return result;
}

std::vector<double> parse_number_array(const JsonValue& value, const std::string& label)
{
    if (!value.is_array()) throw std::invalid_argument(label + " must be array");
    std::vector<double> result;
    for (const auto& item : value.as_array()) {
        if (!item.is_number() || !std::isfinite(item.as_number()))
            throw std::invalid_argument(label + " must contain finite numbers");
        result.push_back(item.as_number());
    }
    return result;
}

InputSpec parse_input(const JsonValue& value)
{
    require_keys(value, {"input_name","shape","element_count","dtype","role"}, "input");
    InputSpec input;
    input.name = string_value(require_member(value, "input_name"), "input_name");
    input.shape = parse_shape(require_member(value, "shape"));
    input.elements = exact_size(require_member(value, "element_count"), "element_count");
    input.dtype = string_value(require_member(value, "dtype"), "dtype");
    input.role = string_value(require_member(value, "role"), "role");
    if (shape_elements(input.shape) != input.elements)
        throw std::invalid_argument("input element_count does not equal shape product");
    return input;
}

ScaleConfig parse_scale(const JsonValue& value)
{
    require_keys(value, {"scale_id","order_of_magnitude","actual_elements","inputs","parameters","mapping_reason"}, "scale");
    ScaleConfig scale;
    scale.scale_id = string_value(require_member(value, "scale_id"), "scale_id");
    scale.order_of_magnitude = string_value(require_member(value, "order_of_magnitude"), "order_of_magnitude");
    scale.actual_elements = exact_size(require_member(value, "actual_elements"), "actual_elements");
    scale.mapping_reason = string_value(require_member(value, "mapping_reason"), "mapping_reason");
    if (scale.mapping_reason.size() < 10) throw std::invalid_argument("mapping_reason too short");

    const JsonValue& inputs = require_member(value, "inputs");
    if (!inputs.is_array() || inputs.as_array().empty())
        throw std::invalid_argument("inputs must be nonempty array");
    std::size_t total_elements = 0;
    for (const auto& item : inputs.as_array()) {
        scale.inputs.push_back(parse_input(item));
        total_elements += scale.inputs.back().elements;
    }
    if (total_elements != scale.actual_elements)
        throw std::invalid_argument("actual_elements does not equal input element sum");
    scale.dtype = scale.inputs.front().dtype;
    for (const auto& input : scale.inputs)
        if (input.dtype != scale.dtype) throw std::invalid_argument("all Remaining operator case window inputs must share dtype");
    scale.length = scale.inputs.front().shape.front();

    const JsonValue& parameters = require_member(value, "parameters");
    if (!parameters.is_object()) throw std::invalid_argument("parameters must be object");
    for (const auto& item : parameters.as_object()) {
        if (item.first == "sym_values") {
            scale.sym_values = parse_bool_array(item.second, "sym_values");
        } else if (item.second.is_number()) {
            if (!std::isfinite(item.second.as_number()))
                throw std::invalid_argument("parameter must be finite: " + item.first);
            scale.number_parameters.emplace(item.first, item.second.as_number());
        } else if (item.second.is_array()) {
            scale.array_parameters.emplace(item.first,
                parse_number_array(item.second, item.first));
        } else if (item.second.is_boolean()) {
            scale.bool_parameters.emplace(item.first, item.second.as_boolean());
        } else if (item.second.is_string()) {
            scale.string_parameters.emplace(item.first, item.second.as_string());
        } else {
            throw std::invalid_argument("unsupported parameter type: " + item.first);
        }
    }
    if (scale.sym_values.size() != 2 || !scale.sym_values[0] || scale.sym_values[1])
        throw std::invalid_argument("sym_values must be exactly [true,false]");
    return scale;
}

OperatorConfig load_operator_config(const fs::path& path)
{
    const std::string text = read_file(path);
    const JsonValue root = JsonValue::parse(text);
    require_keys(root, {"schema_version","scale_set_id","coverage","operators"}, "root");
    if (exact_size(require_member(root, "schema_version"), "schema_version") != 1)
        throw std::invalid_argument("schema_version must be 1");
    if (string_value(require_member(root, "coverage"), "coverage") != "minimal_example_not_full_53")
        throw std::invalid_argument("Remaining operator case standalone config must not claim formal_full_53");
    OperatorConfig config;
    config.config_id = string_value(require_member(root, "scale_set_id"), "scale_set_id");
    config.config_sha256 = zkx::remaining_operator_case::sha256_hex(text);

    const JsonValue& operators = require_member(root, "operators");
    if (!operators.is_array() || operators.as_array().size() != 1)
        throw std::invalid_argument("standalone config must contain exactly one operator");
    const JsonValue& op = operators.as_array().front();
    require_keys(op, {"operator_name","module_name","scales"}, "operator");
    if (string_value(require_member(op, "operator_name"), "operator_name") != kOperator ||
        string_value(require_member(op, "module_name"), "module_name") != kModule)
        throw std::invalid_argument("config operator/module does not match executable");
    const JsonValue& scales = require_member(op, "scales");
    if (!scales.is_array() || scales.as_array().empty())
        throw std::invalid_argument("scales must be nonempty array");
    std::set<std::string> scale_ids;
    std::set<std::string> dtype_orders;
    for (const auto& item : scales.as_array()) {
        ScaleConfig scale = parse_scale(item);
        if (!scale_ids.insert(scale.scale_id).second)
            throw std::invalid_argument("duplicate scale_id: " + scale.scale_id);
        dtype_orders.insert(scale.dtype + ":" + scale.order_of_magnitude);
        config.scales.push_back(std::move(scale));
    }
    const std::array<const char*,5> dtypes{"FP32","FP16","INT32","INT16","INT8"};
    const std::array<const char*,3> orders{"10^2","10^3","10^4"};
    for (const char* dtype : dtypes)
        for (const char* order : orders)
            if (dtype_orders.count(std::string(dtype) + ":" + order) != 1)
                throw std::invalid_argument("config lacks dtype/order pair: " + std::string(dtype) + "/" + order);
    return config;
}

const ScaleConfig& select_scale(const OperatorConfig& config, const std::string& scale_id)
{
    const auto found = std::find_if(config.scales.begin(), config.scales.end(),
        [&](const ScaleConfig& value) { return value.scale_id == scale_id; });
    if (found == config.scales.end()) throw std::invalid_argument("unknown scale_id: " + scale_id);
    return *found;
}

Thresholds load_thresholds(const fs::path& path, const std::string& dtype)
{
    const JsonValue root = JsonValue::parse(read_file(path));
    require_keys(root, {"schema_version","threshold_set_id","entries"}, "threshold root");
    if (exact_size(require_member(root, "schema_version"), "schema_version") != 1)
        throw std::invalid_argument("threshold schema_version must be 1");
    const JsonValue& entries = require_member(root, "entries");
    if (!entries.is_array()) throw std::invalid_argument("threshold entries must be array");
    std::optional<Thresholds> match;
    for (const auto& entry : entries.as_array()) {
        require_keys(entry, {"target","dtype","output_name","output_kind","mse_max","rmse_max","relative_l2_max","relative_linf_max","relative_floor","require_exact_match","max_mismatch_count","semantic_rule"}, "threshold entry");
        if (string_value(require_member(entry,"target"),"target") != kOperator ||
            string_value(require_member(entry,"dtype"),"dtype") != dtype ||
            string_value(require_member(entry,"output_name"),"output_name") != "window") continue;
        Thresholds value;
        value.mse_max = require_member(entry,"mse_max").as_number();
        value.rmse_max = require_member(entry,"rmse_max").as_number();
        value.relative_l2_max = require_member(entry,"relative_l2_max").as_number();
        value.relative_linf_max = require_member(entry,"relative_linf_max").as_number();
        value.relative_floor = require_member(entry,"relative_floor").as_number();
        value.semantic_rule = string_value(require_member(entry,"semantic_rule"),"semantic_rule");
        if (string_value(require_member(entry,"output_kind"),"output_kind") != "floating" ||
            !require_member(entry,"require_exact_match").is_boolean() ||
            require_member(entry,"require_exact_match").as_boolean() ||
            !require_member(entry,"max_mismatch_count").is_null())
            throw std::invalid_argument("window threshold must describe non-exact floating output");
        if (value.mse_max < 0.0 || value.rmse_max < 0.0 ||
            value.relative_l2_max < 0.0 || value.relative_linf_max < 0.0 ||
            value.relative_floor <= 0.0)
            throw std::invalid_argument("window threshold values are invalid");
        if (match.has_value())
            throw std::invalid_argument("duplicate exact threshold for operator/dtype/window");
        match = value;
    }
    if (!match.has_value()) throw std::invalid_argument("no exact threshold for operator/dtype/window");
    return *match;
}

Cli parse_cli(int argc, char** argv)
{
    Cli cli;
    std::map<std::string,std::string> values;
    for (int index=1; index<argc; ++index) {
        const std::string key=argv[index];
        if (index+1>=argc || key.rfind("--",0)!=0) throw std::invalid_argument("invalid CLI argument");
        values[key]=argv[++index];
    }
    const auto take=[&](const char* key)->std::string {
        const auto found=values.find(key);
        if(found==values.end() || found->second.empty()) throw std::invalid_argument(std::string("missing ")+key);
        return found->second;
    };
    cli.config_file=take("--config"); cli.thresholds_file=take("--thresholds");
    cli.results_root=take("--results-root"); cli.scale_id=take("--scale");
    cli.run_type=take("--run-type"); cli.git_commit=take("--git-commit");
    cli.git_dirty=take("--git-dirty"); cli.test_time_utc=take("--test-time-utc");
    cli.random_hex8=take("--random");
    const std::string sym=take("--sym");
    if(sym=="true") cli.symmetric=true; else if(sym=="false") cli.symmetric=false;
    else throw std::invalid_argument("--sym must be true or false");
    if(cli.run_type!="smoke" && cli.run_type!="formal") throw std::invalid_argument("run-type must be smoke or formal");
    if(cli.git_dirty!="true" && cli.git_dirty!="false") throw std::invalid_argument("git-dirty must be true or false");
    return cli;
}

double parameter(const ScaleConfig& scale, const std::string& name)
{
    const auto found=scale.number_parameters.find(name);
    if(found==scale.number_parameters.end()) throw std::invalid_argument("missing parameter: "+name);
    return found->second;
}

int integer_parameter(const ScaleConfig& scale, const std::string& name)
{
    const double value = parameter(scale, name);
    if (std::floor(value) != value)
        throw std::invalid_argument("parameter must be integer: " + name);
    return static_cast<int>(value);
}

bool bool_parameter(const ScaleConfig& scale, const std::string& name)
{
    const auto found = scale.bool_parameters.find(name);
    if (found == scale.bool_parameters.end())
        throw std::invalid_argument("missing boolean parameter: " + name);
    return found->second;
}

const std::vector<double>& array_parameter(const ScaleConfig& scale, const std::string& name)
{
    const auto found=scale.array_parameters.find(name);
    if(found==scale.array_parameters.end()) throw std::invalid_argument("missing array parameter: "+name);
    return found->second;
}

const std::string& string_parameter(const ScaleConfig& scale, const std::string& name)
{
    const auto found=scale.string_parameters.find(name);
    if(found==scale.string_parameters.end()) throw std::invalid_argument("missing parameter: "+name);
    return found->second;
}

template<class T> T typed_value(double value) { return static_cast<T>(value); }
template<> __half typed_value<__half>(double value) { return __float2half_rn(static_cast<float>(value)); }

template<class T>
std::vector<T> typed_coefficients(const ScaleConfig& scale)
{
    std::vector<T> result;
    for(double value:array_parameter(scale,"coefficient_values")) result.push_back(typed_value<T>(value));
    return result;
}

template<class T>
std::vector<double> cpu_once(const ScaleConfig& scale, bool symmetric)
{
    if(std::string(kOperator)=="chebwin")
        return cusignal::chebwin_typed_cpu(scale.length,typed_value<T>(parameter(scale,"attenuation_db")),symmetric);
    if(std::string(kOperator)=="general_cosine")
        return cusignal::general_cosine_typed_cpu(scale.length,typed_coefficients<T>(scale),symmetric);
    if(std::string(kOperator)=="general_gaussian")
        return cusignal::general_gaussian_typed_cpu(scale.length,typed_value<T>(parameter(scale,"power")),typed_value<T>(parameter(scale,"width")),symmetric);
    if(std::string(kOperator)=="kaiser")
        return cusignal::kaiser_typed_cpu(scale.length,typed_value<T>(parameter(scale,"beta")),symmetric);
    if(std::string(kOperator)=="parzen")
        return cusignal::parzen_typed_cpu<T>(scale.length,symmetric);
    if(std::string(kOperator)=="taylor")
        return cusignal::taylor_typed_cpu(scale.length,integer_parameter(scale,"nbar"),
            typed_value<T>(parameter(scale,"sidelobe_level_db")),
            bool_parameter(scale,"normalize"),symmetric);
    if(std::string(kOperator)=="triang")
        return cusignal::triang_typed_cpu<T>(scale.length,symmetric);
    throw std::logic_error("unsupported compiled operator");
}

template<class T>
std::vector<double> gpu_once(const ScaleConfig& scale, bool symmetric)
{
    cusignal::DeviceArray<double> output(static_cast<std::size_t>(scale.length));
    if(std::string(kOperator)=="chebwin")
        cusignal::chebwin_device(scale.length,typed_value<T>(parameter(scale,"attenuation_db")),output,symmetric);
    else if(std::string(kOperator)=="general_cosine") {
        auto coefficients=cusignal::DeviceArray<T>::from_host(typed_coefficients<T>(scale));
        cusignal::general_cosine_device(scale.length,coefficients,output,symmetric);
    } else if(std::string(kOperator)=="general_gaussian")
        cusignal::general_gaussian_device(scale.length,typed_value<T>(parameter(scale,"power")),typed_value<T>(parameter(scale,"width")),output,symmetric);
    else if(std::string(kOperator)=="kaiser")
        cusignal::kaiser_device(scale.length,typed_value<T>(parameter(scale,"beta")),output,symmetric);
    else if(std::string(kOperator)=="parzen")
        cusignal::parzen_device<T>(scale.length,output,symmetric);
    else if(std::string(kOperator)=="taylor")
        cusignal::taylor_device(scale.length,integer_parameter(scale,"nbar"),
            typed_value<T>(parameter(scale,"sidelobe_level_db")),output,
            bool_parameter(scale,"normalize"),symmetric);
    else if(std::string(kOperator)=="triang")
        cusignal::triang_device<T>(scale.length,output,symmetric);
    else throw std::logic_error("unsupported compiled operator");
    return output.to_host();
}

zkx::common::ResourceSample gpu_sample()
{
    return zkx::common::sample_gpu_used([] {
        const auto info=cusignal::cuda_utils::current_memory_info();
        return static_cast<std::int64_t>(info.total_bytes-info.free_bytes);
    },"cudaMemGetInfo.total_minus_free");
}

PhaseSample capture_phase(const std::string& phase,int index,bool include_gpu)
{
    PhaseSample sample{phase,index,zkx::common::sample_cpu_heap(),zkx::common::sample_cpu_rss(),{}};
    sample.gpu=include_gpu?gpu_sample():zkx::common::ResourceSample{false,0,"NA","CPU path"};
    return sample;
}

template<class T>
std::vector<PhaseSample> cpu_trace(const ScaleConfig& scale,bool symmetric)
{
    std::vector<PhaseSample> trace;
    trace.push_back(capture_phase("before",0,false));
    std::vector<T> coefficients;
    if(std::string(kOperator)=="general_cosine") coefficients=typed_coefficients<T>(scale);
    trace.push_back(capture_phase("allocate",1,false));
    trace.push_back(capture_phase("h2d",2,false));
    std::vector<double> output=cpu_once<T>(scale,symmetric);
    trace.push_back(capture_phase("execute_sync",3,false));
    trace.push_back(capture_phase("d2h",4,false));
    output.clear(); output.shrink_to_fit(); coefficients.clear(); coefficients.shrink_to_fit();
    trace.push_back(capture_phase("release_sync",5,false));
    trace.push_back(capture_phase("after",6,false));
    return trace;
}

template<class T>
std::vector<PhaseSample> gpu_trace(const ScaleConfig& scale,bool symmetric)
{
    std::vector<PhaseSample> trace;
    cusignal::cuda_utils::synchronize_stream();
    trace.push_back(capture_phase("before",0,true));
    std::optional<cusignal::DeviceArray<double>> output;
    std::optional<cusignal::DeviceArray<T>> coefficients;
    output.emplace(static_cast<std::size_t>(scale.length));
    trace.push_back(capture_phase("allocate",1,true));
    if(std::string(kOperator)=="general_cosine") coefficients.emplace(cusignal::DeviceArray<T>::from_host(typed_coefficients<T>(scale)));
    trace.push_back(capture_phase("h2d",2,true));
    if(std::string(kOperator)=="chebwin")
        cusignal::chebwin_device(scale.length,typed_value<T>(parameter(scale,"attenuation_db")),*output,symmetric);
    else if(std::string(kOperator)=="general_cosine")
        cusignal::general_cosine_device(scale.length,*coefficients,*output,symmetric);
    else if(std::string(kOperator)=="general_gaussian")
        cusignal::general_gaussian_device(scale.length,typed_value<T>(parameter(scale,"power")),typed_value<T>(parameter(scale,"width")),*output,symmetric);
    else if(std::string(kOperator)=="kaiser")
        cusignal::kaiser_device(scale.length,typed_value<T>(parameter(scale,"beta")),*output,symmetric);
    else if(std::string(kOperator)=="parzen")
        cusignal::parzen_device<T>(scale.length,*output,symmetric);
    else if(std::string(kOperator)=="taylor")
        cusignal::taylor_device(scale.length,integer_parameter(scale,"nbar"),
            typed_value<T>(parameter(scale,"sidelobe_level_db")),*output,
            bool_parameter(scale,"normalize"),symmetric);
    else if(std::string(kOperator)=="triang")
        cusignal::triang_device<T>(scale.length,*output,symmetric);
    else throw std::logic_error("unsupported compiled operator");
    cusignal::cuda_utils::synchronize_stream();
    trace.push_back(capture_phase("execute_sync",3,true));
    std::vector<double> host=output->to_host();
    trace.push_back(capture_phase("d2h",4,true));
    host.clear(); host.shrink_to_fit(); coefficients.reset(); output.reset();
    cusignal::cuda_utils::synchronize_stream();
    trace.push_back(capture_phase("release_sync",5,true));
    trace.push_back(capture_phase("after",6,true));
    return trace;
}

ResourceSummary summarize_resource(const std::vector<PhaseSample>& trace,
                                   const std::string& kind)
{
    ResourceSummary result;
    auto get=[&](const PhaseSample& sample)->const zkx::common::ResourceSample& {
        if(kind=="heap") return sample.cpu_heap;
        if(kind=="rss") return sample.rss;
        return sample.gpu;
    };
    result.available=std::all_of(trace.begin(),trace.end(),[&](const PhaseSample& value){return get(value).available;});
    if(!result.available) return result;
    result.before=get(trace.front()).bytes; result.after=get(trace.back()).bytes;
    result.peak=result.before;
    for(const auto& sample:trace) result.peak=std::max(result.peak,get(sample).bytes);
    result.delta=result.after-result.before;
    return result;
}

std::string number(double value)
{
    std::ostringstream out; out<<std::setprecision(17)<<value; return out.str();
}

std::string integer(std::int64_t value) { return std::to_string(value); }

struct InputContent {
    std::string name;
    std::string shape;
    std::string dtype;
    std::size_t elements=0;
    std::string generator;
    std::string preview;
};

std::string json_quote(const std::string& value)
{
    std::ostringstream out; out<<'"';
    for(const unsigned char c:value) {
        if(c=='"' || c=='\\') out<<'\\'<<static_cast<char>(c);
        else if(c=='\n') out<<"\\n";
        else if(c=='\r') out<<"\\r";
        else if(c=='\t') out<<"\\t";
        else out<<static_cast<char>(c);
    }
    out<<'"'; return out.str();
}

std::vector<std::size_t> preview_indices(std::size_t count)
{
    if(count<=6) { std::vector<std::size_t> result; for(std::size_t i=0;i<count;++i) result.push_back(i); return result; }
    return {0,1,2,3,count-2,count-1};
}

std::string index_preview(std::size_t count)
{
    std::ostringstream out;
    for(const std::size_t index:preview_indices(count)) {
        if(out.tellp()>0) out<<';'; out<<"i="<<index<<":position="<<index;
    }
    return out.str();
}

std::string value_preview(const std::vector<double>& values)
{
    std::ostringstream out;
    for(const std::size_t index:preview_indices(values.size())) {
        if(out.tellp()>0) out<<';'; out<<"i="<<index<<':'<<number(values[index]);
    }
    return out.str();
}

std::vector<InputContent> input_content(const ScaleConfig& scale)
{
    std::vector<InputContent> result{{"window_positions","["+std::to_string(scale.length)+"]",
        "INDEX_INT64",static_cast<std::size_t>(scale.length),
        "implicit_window_index_domain(no_materialized_array)",index_preview(scale.length)}};
    const auto scalar=[&](const std::string& name,const std::string& dtype,double value) {
        result.push_back({name,"scalar",dtype,1,"exact_config_scalar","value="+number(value)});
    };
    if(std::string(kOperator)=="general_cosine") {
        const auto& values=array_parameter(scale,"coefficient_values");
        result.push_back({"coefficient_values","["+std::to_string(values.size())+"]",scale.dtype,
            values.size(),"exact_config_array",value_preview(values)});
    } else if(std::string(kOperator)=="general_gaussian") {
        scalar("power",scale.dtype,parameter(scale,"power"));
        scalar("width",scale.dtype,parameter(scale,"width"));
    } else if(std::string(kOperator)=="chebwin") {
        scalar("attenuation_db",scale.dtype,parameter(scale,"attenuation_db"));
    } else if(std::string(kOperator)=="kaiser") {
        scalar("beta",scale.dtype,parameter(scale,"beta"));
    } else if(std::string(kOperator)=="taylor") {
        scalar("nbar","INT32",parameter(scale,"nbar"));
        scalar("sidelobe_level_db",scale.dtype,parameter(scale,"sidelobe_level_db"));
        result.push_back({"normalize","scalar","BOOL",1,"exact_config_scalar",
            std::string("value=")+(bool_parameter(scale,"normalize")?"true":"false")});
    }
    return result;
}

std::string input_content_json(const std::vector<InputContent>& content)
{
    std::ostringstream out; out<<'[';
    for(std::size_t i=0;i<content.size();++i) {
        if(i) out<<',';
        const auto& row=content[i];
        out<<"{\"input_name\":"<<json_quote(row.name)<<",\"shape\":"<<json_quote(row.shape)
           <<",\"dtype\":"<<json_quote(row.dtype)<<",\"element_count\":"<<json_quote(std::to_string(row.elements))
           <<",\"generator\":"<<json_quote(row.generator)<<",\"preview_heacuda_api_tail2\":"<<json_quote(row.preview)<<'}';
    }
    out<<']'; return out.str();
}

std::string case_parameters_json(const ScaleConfig& scale,bool symmetric)
{
    std::ostringstream out; out<<"{\"length\":"<<scale.length<<",\"requested_dtype\":"<<json_quote(scale.dtype)
        <<",\"sym\":"<<(symmetric?"true":"false");
    for(const auto& item:scale.number_parameters) out<<','<<json_quote(item.first)<<':'<<number(item.second);
    for(const auto& item:scale.bool_parameters) out<<','<<json_quote(item.first)<<':'<<(item.second?"true":"false");
    for(const auto& item:scale.string_parameters) out<<','<<json_quote(item.first)<<':'<<json_quote(item.second);
    for(const auto& item:scale.array_parameters) {
        out<<','<<json_quote(item.first)<<":[";
        for(std::size_t i=0;i<item.second.size();++i) { if(i) out<<','; out<<number(item.second[i]); }
        out<<']';
    }
    out<<'}'; return out.str();
}

std::string nullable(const ResourceSummary& value,std::int64_t ResourceSummary::*field)
{
    return value.available?integer(value.*field):"NA";
}

std::string serialize_output(const std::vector<double>& output)
{
    std::ostringstream text; text<<std::setprecision(17);
    for(double value:output) text<<value<<';';
    return text.str();
}

template<class T>
std::string object_bytes_hex(const T& value)
{
    const auto* bytes=reinterpret_cast<const unsigned char*>(&value);
    std::ostringstream out; out<<std::hex<<std::setfill('0');
    for(std::size_t index=0;index<sizeof(T);++index) out<<std::setw(2)<<static_cast<unsigned int>(bytes[index]);
    return out.str();
}

struct DtypeEvidence {
    std::string semantics;
    std::size_t typed_input_count=0;
    std::string manifest;
    std::string digest;
};

template<class T>
DtypeEvidence dtype_evidence(const ScaleConfig& scale,bool symmetric)
{
    DtypeEvidence result;
    std::ostringstream manifest;
    manifest<<"length:INT32="<<scale.length<<"|sym:BOOL="<<(symmetric?"true":"false");
    const auto scalar=[&](const char* name) {
        const T value=typed_value<T>(parameter(scale,name));
        manifest<<'|'<<name<<':'<<scale.dtype<<":bytes="<<object_bytes_hex(value);
        ++result.typed_input_count;
    };
    if(std::string(kOperator)=="general_cosine") {
        const auto values=typed_coefficients<T>(scale);
        manifest<<"|coefficient_values:"<<scale.dtype<<":bytes=";
        for(const T& value:values) manifest<<object_bytes_hex(value)<<';';
        result.typed_input_count=values.size();
        result.semantics="actual_typed_array_input";
    } else if(std::string(kOperator)=="general_gaussian") {
        scalar("power"); scalar("width"); result.semantics="actual_typed_scalar_inputs";
    } else if(std::string(kOperator)=="chebwin") {
        scalar("attenuation_db"); result.semantics="actual_typed_scalar_input";
    } else if(std::string(kOperator)=="kaiser") {
        scalar("beta"); result.semantics="actual_typed_scalar_input";
    } else if(std::string(kOperator)=="taylor") {
        scalar("sidelobe_level_db"); result.semantics="actual_typed_scalar_input";
    } else if(std::string(kOperator)=="parzen" || std::string(kOperator)=="triang") {
        result.semantics="request_variant_no_typed_data_input";
    } else {
        throw std::logic_error("unsupported dtype evidence operator");
    }
    result.manifest=manifest.str();
    result.digest=zkx::remaining_operator_case::sha256_hex(result.manifest);
    return result;
}

std::string input_shape(const ScaleConfig& scale)
{
    std::ostringstream out;
    for(std::size_t i=0;i<scale.inputs.size();++i) {
        if(i) out<<';'; out<<scale.inputs[i].name<<"=[";
        for(std::size_t j=0;j<scale.inputs[i].shape.size();++j) out<<(j?"x":"")<<scale.inputs[i].shape[j];
        out<<']';
    }
    return out.str();
}

std::vector<std::string> identity_columns()
{
    return {"schema_version","run_id","run_type","test_time_utc","git_commit","git_dirty","config_id","config_sha256","target_kind","target","task_name","step_name","operator_name","case_id","scale_id","order_of_magnitude","actual_elements","input_shape","dtype","device","backend","timing_scope","status","error_code"};
}

std::vector<std::string> identity_values(const Cli& cli,const OperatorConfig& config,
    const ScaleConfig& scale,const std::string& run_id,const std::string& case_id,
    const std::string& device,const std::string& status,const std::string& error_code)
{
    return {"1",run_id,cli.run_type,cli.test_time_utc,cli.git_commit,cli.git_dirty,
        config.config_id,config.config_sha256,"operator",kOperator,"NA","NA",kOperator,
        case_id,scale.scale_id,scale.order_of_magnitude,std::to_string(scale.actual_elements),
        input_shape(scale),scale.dtype,device,kBackend,kTimingScope,status,error_code};
}

void append_all(std::vector<std::string>& target,const std::vector<std::string>& values)
{ target.insert(target.end(),values.begin(),values.end()); }

void write_text_new(const fs::path& path,const std::string& contents)
{
    if(fs::exists(path)) throw std::invalid_argument("output path exists: "+path.string());
    std::ofstream out(path,std::ios::binary); if(!out) throw std::runtime_error("cannot create "+path.string());
    out<<contents; if(!out) throw std::runtime_error("cannot write "+path.string());
}

using CsvRow=std::map<std::string,std::string>;

std::vector<std::string> parse_csv_line(const std::string& line)
{
    std::vector<std::string> fields; std::string field; bool quoted=false;
    for(std::size_t i=0;i<line.size();++i) {
        const char c=line[i];
        if(c=='"') {
            if(quoted&&i+1<line.size()&&line[i+1]=='"') { field+='"'; ++i; }
            else quoted=!quoted;
        } else if(c==','&&!quoted) { fields.push_back(field); field.clear(); }
        else field+=c;
    }
    if(quoted) throw std::runtime_error("unterminated quoted CSV field");
    fields.push_back(field); return fields;
}

std::vector<CsvRow> read_csv_rows(const fs::path& path)
{
    std::ifstream input(path,std::ios::binary); if(!input) throw std::runtime_error("cannot read "+path.string());
    std::string line; if(!std::getline(input,line)) throw std::runtime_error("CSV header missing");
    if(!line.empty()&&line.back()=='\r') line.pop_back();
    const auto header=parse_csv_line(line); std::vector<CsvRow> rows;
    while(std::getline(input,line)) {
        if(!line.empty()&&line.back()=='\r') line.pop_back();
        const auto values=parse_csv_line(line);
        if(values.size()!=header.size()) throw std::runtime_error("CSV row width mismatch: "+path.string());
        CsvRow row; for(std::size_t i=0;i<header.size();++i) row.emplace(header[i],values[i]);
        rows.push_back(std::move(row));
    }
    return rows;
}

std::string csv_value(const CsvRow& row,const std::string& key)
{
    const auto found=row.find(key); return found==row.end()?"NA":found->second;
}

const CsvRow& device_row(const std::vector<CsvRow>& rows,const std::string& device)
{
    const auto found=std::find_if(rows.begin(),rows.end(),[&](const CsvRow& row){return csv_value(row,"device")==device;});
    if(found==rows.end()) throw std::runtime_error("CSV device row missing: "+device);
    return *found;
}

template<class T>
int execute(const Cli& cli,const OperatorConfig& config,const ScaleConfig& scale,
            const Thresholds& thresholds)
{
    if(std::find(scale.sym_values.begin(),scale.sym_values.end(),cli.symmetric)==scale.sym_values.end())
        throw std::invalid_argument("selected sym value is not configured");
    const std::size_t warmup=cli.run_type=="formal"?20U:1U;
    const std::size_t measured=cli.run_type=="formal"?100U:1U;
    const std::string profile=std::string("remaining_operator_case_")+kBatch+"_"+cli.run_type;
    const std::string run_id=zkx::common::make_run_id(cli.test_time_utc,cli.git_commit,profile,cli.random_hex8);
    const DtypeEvidence dtype_record=dtype_evidence<T>(scale,cli.symmetric);
    if(std::string(kBatch)=="b02") {
        if(string_parameter(scale,"dtype_semantics")!=dtype_record.semantics)
            throw std::invalid_argument("configured dtype_semantics does not match actual operator signature");
        if(string_parameter(scale,"typed_input_names")!=(dtype_record.typed_input_count==0?"none":"sidelobe_level_db"))
            throw std::invalid_argument("configured typed_input_names does not match actual Batch02 invocation");
    }
    const std::string input_digest=dtype_record.digest;
    const auto input_records=input_content(scale);
    const std::string input_json=input_content_json(input_records);
    const std::string parameters_json=case_parameters_json(scale,cli.symmetric);
    const std::string case_id=zkx::common::make_case_id(kOperator,scale.scale_id,input_digest)
        +(cli.symmetric?"__sym":"__periodic");
    const fs::path output_dir=cli.results_root/cli.run_type/run_id;
    if(fs::exists(output_dir)) throw std::invalid_argument("run_id output directory already exists");
    fs::create_directories(output_dir);

    std::vector<double> cpu_output,gpu_output;
    const auto cpu_samples=zkx::common::collect_ms(warmup,measured,[&]{cpu_output=cpu_once<T>(scale,cli.symmetric);},[]{});
    const auto gpu_samples=zkx::common::collect_ms(warmup,measured,[&]{gpu_output=gpu_once<T>(scale,cli.symmetric);},[]{cusignal::cuda_utils::synchronize_stream();});
    const auto cpu_stats=zkx::common::summarize_ms(cpu_samples);
    const auto gpu_stats=zkx::common::summarize_ms(gpu_samples);
    const double paired_speedup=zkx::common::speedup(cpu_stats,gpu_stats,kTimingScope,kTimingScope);
    const auto accuracy=zkx::common::compare(cpu_output,gpu_output,thresholds.relative_floor);
    const bool accuracy_pass=accuracy.mse<=thresholds.mse_max && accuracy.rmse<=thresholds.rmse_max
        && accuracy.relative_l2<=thresholds.relative_l2_max && accuracy.relative_linf<=thresholds.relative_linf_max;

    const auto auxiliary=zkx::common::measure_cuda_event_auxiliary([&]{
        cusignal::cuda_utils::EventTimer timer; timer.record_start();
        gpu_output=gpu_once<T>(scale,cli.symmetric); timer.record_stop(); return static_cast<double>(timer.elapsed_ms());
    });
    const auto cpu_phases=cpu_trace<T>(scale,cli.symmetric);
    const auto gpu_phases=gpu_trace<T>(scale,cli.symmetric);
    const auto cpu_heap=summarize_resource(cpu_phases,"heap");
    const auto cpu_rss=summarize_resource(cpu_phases,"rss");
    const auto gpu_heap=summarize_resource(gpu_phases,"heap");
    const auto gpu_rss=summarize_resource(gpu_phases,"rss");
    const auto gpu_memory=summarize_resource(gpu_phases,"gpu");
    const bool resource_pass=cpu_heap.available&&cpu_rss.available&&gpu_heap.available&&gpu_rss.available&&gpu_memory.available;
    const bool passed=accuracy_pass&&resource_pass;
    const std::string status=passed?"PASS":"FAIL";
    const std::string error_code=accuracy_pass?(resource_pass?"OK":"RESOURCE_FAILED"):"ACCURACY_FAILED";
    const std::string cpu_output_digest=
        zkx::remaining_operator_case::sha256_hex(serialize_output(cpu_output));
    const std::string gpu_output_digest=
        zkx::remaining_operator_case::sha256_hex(serialize_output(gpu_output));

    auto main_columns=identity_columns();
    append_all(main_columns,{"warmup_runs","measured_runs","mean_ms","p50_ms","p95_ms","p99_ms","min_ms","max_ms","std_ms","cv","cpu_gpu_speedup","accuracy_reference","mse","rmse","relative_l2","relative_linf","exact_match","mismatch_count","semantic_check","accuracy_status","cpu_heap_before_bytes","cpu_heap_after_bytes","cpu_heap_peak_bytes","cpu_heap_delta_bytes","rss_before_bytes","rss_after_bytes","rss_peak_bytes","rss_delta_bytes","gpu_before_bytes","gpu_after_bytes","gpu_peak_bytes","gpu_delta_bytes","stability_case_index","stability_total_cases","perturbation_file","perturbation_sha256","perturbation_summary","input_mode","input_source_file","input_source_sha256","selection_seed","profile_entry_id"});
    zkx::common::CsvWriter main_writer(output_dir/(std::string("operator_main_results_")+kOperator+"_"+kBackend+"_"+run_id+".csv"),main_columns);
    const auto emit_main=[&](const std::string& device,const zkx::common::Statistics& stats,
        const ResourceSummary& heap,const ResourceSummary& rss,const ResourceSummary* gpu,
        const zkx::common::Accuracy& metrics,const std::string& speedup_value) {
        auto row=identity_values(cli,config,scale,run_id,case_id,device,status,error_code);
        append_all(row,{std::to_string(warmup),std::to_string(measured),number(stats.mean_ms),number(stats.p50_ms),number(stats.p95_ms),number(stats.p99_ms),number(stats.min_ms),number(stats.max_ms),number(stats.stddev_ms),number(stats.cv),speedup_value,"cpu_typed_reference",number(metrics.mse),number(metrics.rmse),number(metrics.relative_l2),number(metrics.relative_linf),"NA","NA",thresholds.semantic_rule+std::string(":shape=[")+std::to_string(scale.length)+"];finite=true;sym="+(cli.symmetric?"true":"false"),accuracy_pass?"PASS":"FAIL",nullable(heap,&ResourceSummary::before),nullable(heap,&ResourceSummary::after),nullable(heap,&ResourceSummary::peak),nullable(heap,&ResourceSummary::delta),nullable(rss,&ResourceSummary::before),nullable(rss,&ResourceSummary::after),nullable(rss,&ResourceSummary::peak),nullable(rss,&ResourceSummary::delta),gpu?nullable(*gpu,&ResourceSummary::before):"NA",gpu?nullable(*gpu,&ResourceSummary::after):"NA",gpu?nullable(*gpu,&ResourceSummary::peak):"NA",gpu?nullable(*gpu,&ResourceSummary::delta):"NA","NA","NA","NA","NA","NA","NA","NA","NA","NA","NA"});
        main_writer.append(row);
    };
    const zkx::common::Accuracy zero{};
    emit_main("CPU",cpu_stats,cpu_heap,cpu_rss,nullptr,zero,"NA");
    emit_main("GPU",gpu_stats,gpu_heap,gpu_rss,&gpu_memory,accuracy,
        number(paired_speedup));

    auto timing_columns=identity_columns();
    append_all(timing_columns,{"warmup_runs","measured_runs","sample_index","latency_ms","synchronized","input_digest","output_digest"});
    zkx::common::CsvWriter timing_writer(output_dir/(std::string("operator_timing_samples_")+kOperator+"_"+kBackend+"_"+run_id+".csv"),timing_columns);
    const auto emit_samples=[&](const char* device,const std::vector<double>& samples,
                                const std::string& output_digest) {
        for(std::size_t index=0;index<samples.size();++index) {
            auto row=identity_values(cli,config,scale,run_id,case_id,device,status,error_code);
            append_all(row,{std::to_string(warmup),std::to_string(measured),std::to_string(index),number(samples[index]),"true",input_digest,output_digest});
            timing_writer.append(row);
        }
    };
    emit_samples("CPU",cpu_samples,cpu_output_digest);
    emit_samples("GPU",gpu_samples,gpu_output_digest);

    auto memory_columns=identity_columns();
    append_all(memory_columns,{"sample_index","trace_phase","phase_index","cpu_live_heap_bytes","rss_bytes","gpu_used_bytes"});
    zkx::common::CsvWriter memory_writer(output_dir/(std::string("operator_windows_")+kOperator+"_memory_trace_"+kBackend+".csv"),memory_columns);
    const auto emit_trace=[&](const char* device,const std::vector<PhaseSample>& trace) {
        for(const auto& sample:trace) {
            auto row=identity_values(cli,config,scale,run_id,case_id,device,status,error_code);
            append_all(row,{"0",sample.phase,std::to_string(sample.phase_index),
                sample.cpu_heap.available?integer(sample.cpu_heap.bytes):"NA",
                sample.rss.available?integer(sample.rss.bytes):"NA",
                sample.gpu.available?integer(sample.gpu.bytes):"NA"});
            memory_writer.append(row);
        }
    };
    emit_trace("CPU",cpu_phases); emit_trace("GPU",gpu_phases);

    zkx::common::CsvWriter input_writer(output_dir/(std::string("operator_input_evidence_")+kOperator+"_"+kBackend+"_"+run_id+".csv"),
        {"run_id","case_id","operator_name","input_name","shape","dtype","element_count","generator","preview_heacuda_api_tail2","status","error_code"});
    for(const auto& row:input_records) input_writer.append({run_id,case_id,kOperator,row.name,row.shape,row.dtype,
        std::to_string(row.elements),row.generator,row.preview,status,error_code});

    zkx::common::CsvWriter dtype_writer(output_dir/(std::string("operator_dtype_evidence_")+kOperator+"_"+kBackend+"_"+run_id+".csv"),
        {"run_id","case_id","operator_name","requested_dtype","config_input_dtype","dtype_semantics","typed_input_count","typed_input_manifest","input_content_json","case_parameters_json","actual_input_digest","output_dtype","status","error_code"});
    dtype_writer.append({run_id,case_id,kOperator,scale.dtype,scale.dtype,dtype_record.semantics,
        std::to_string(dtype_record.typed_input_count),dtype_record.manifest,input_json,parameters_json,
        dtype_record.digest,"FP64",status,error_code});

    const std::string batch_number = std::string(kBatch).substr(1);
    zkx::common::CsvWriter module_writer(output_dir/(std::string("windows_batch")+batch_number+"_module_summary_"+kBackend+"_"+run_id+".csv"),
        {"run_id","case_id","operator_name","scale_id","dtype","sym","backend","warmup_runs","measured_runs","cpu_mean_ms","gpu_mean_ms","cpu_gpu_speedup","mse","rmse","relative_l2","relative_linf","cuda_event_ms","cuda_event_source","status","error_code"});
    module_writer.append({run_id,case_id,kOperator,scale.scale_id,scale.dtype,cli.symmetric?"true":"false",kBackend,
        std::to_string(warmup),std::to_string(measured),number(cpu_stats.mean_ms),number(gpu_stats.mean_ms),number(paired_speedup),number(accuracy.mse),number(accuracy.rmse),number(accuracy.relative_l2),number(accuracy.relative_linf),auxiliary.available?number(auxiliary.cuda_event_ms):"NA",auxiliary.source,status,error_code});

    std::ostringstream summary;
    summary<<"status="<<status<<"\noperator="<<kOperator<<"\nrun_id="<<run_id<<"\ncase_id="<<case_id
        <<"\nscale_id="<<scale.scale_id<<"\ndtype="<<scale.dtype<<"\nsym="<<(cli.symmetric?"true":"false")
        <<"\ndtype_semantics="<<dtype_record.semantics<<"\ntyped_input_count="<<dtype_record.typed_input_count
        <<"\ntyped_input_manifest="<<dtype_record.manifest
        <<"\ninput_content_json="<<input_json<<"\ncase_parameters_json="<<parameters_json
        <<"\nbackend="<<kBackend<<"\nwarmup_runs="<<warmup<<"\nmeasured_runs="<<measured
        <<"\ninput_digest="<<input_digest<<"\ncpu_output_digest="<<cpu_output_digest
        <<"\ngpu_output_digest="<<gpu_output_digest<<"\nerror_code="<<error_code<<'\n';
    const fs::path summary_path=output_dir/(std::string("operator_")+kOperator+"_summary.txt");
    write_text_new(summary_path,summary.str());

    std::ostringstream json;
    json<<"{\n  \"schema_version\": 1,\n  \"run_id\": \""<<run_id<<"\",\n  \"case_id\": \""<<case_id
        <<"\",\n  \"operator_name\": \""<<kOperator<<"\",\n  \"scale_id\": \""<<scale.scale_id
        <<"\",\n  \"dtype\": \""<<scale.dtype<<"\",\n  \"sym\": "<<(cli.symmetric?"true":"false")
        <<",\n  \"dtype_semantics\": \""<<dtype_record.semantics
        <<"\",\n  \"typed_input_count\": "<<dtype_record.typed_input_count
        <<",\n  \"typed_input_manifest\": \""<<dtype_record.manifest<<"\""
        <<",\n  \"input_content\": "<<input_json
        <<",\n  \"case_parameters\": "<<parameters_json
        <<",\n  \"backend\": \""<<kBackend<<"\",\n  \"warmup_runs\": "<<warmup
        <<",\n  \"measured_runs\": "<<measured<<",\n  \"input_digest\": \""<<input_digest
        <<"\",\n  \"cuda_event_auxiliary_ms\": "<<(auxiliary.available?number(auxiliary.cuda_event_ms):"null")
        <<",\n  \"cuda_event_source\": \""<<auxiliary.source<<"\",\n  \"mse\": "<<number(accuracy.mse)
        <<",\n  \"rmse\": "<<number(accuracy.rmse)<<",\n  \"relative_l2\": "<<number(accuracy.relative_l2)
        <<",\n  \"relative_linf\": "<<number(accuracy.relative_linf)<<",\n  \"status\": \""<<status
        <<"\",\n  \"error_code\": \""<<error_code<<"\"\n}\n";
    const fs::path report_path=output_dir/(std::string("operator_")+kOperator+"_report.json");
    write_text_new(report_path,json.str());

    const fs::path log_path=output_dir/(std::string("operator_")+kOperator+"_full.log");
    const std::string log_prefix = std::string("[remaining_operator_case-") + kBatch + "]";
    std::ostringstream log;
    log<<log_prefix<<" operator="<<kOperator<<" entry=cpu:"<<kOperator<<"_typed_cpu gpu:"<<kOperator
        <<"_device\n"<<log_prefix<<" config="<<cli.config_file.string()<<" config_sha256="<<config.config_sha256
        <<"\n"<<log_prefix<<" run_type="<<cli.run_type<<" scale="<<scale.scale_id<<" dtype="<<scale.dtype
        <<" sym="<<(cli.symmetric?"true":"false")<<" backend="<<kBackend
        <<"\n"<<log_prefix<<" dtype_semantics="<<dtype_record.semantics
        <<" typed_input_count="<<dtype_record.typed_input_count<<" actual_input_digest="<<dtype_record.digest
        <<"\n"<<log_prefix<<" timing_scope="<<kTimingScope<<" warmup="<<warmup<<" measured="<<measured
        <<"\n"<<log_prefix<<" cpu_mean_ms="<<number(cpu_stats.mean_ms)<<" gpu_mean_ms="<<number(gpu_stats.mean_ms)
        <<" speedup="<<number(paired_speedup)<<"\n"<<log_prefix<<" mse="<<number(accuracy.mse)<<" rmse="<<number(accuracy.rmse)
        <<" relative_l2="<<number(accuracy.relative_l2)<<" relative_linf="<<number(accuracy.relative_linf)
        <<"\n"<<log_prefix<<" resources="<<(resource_pass?"available":"unavailable")
        <<" cuda_event_ms="<<(auxiliary.available?number(auxiliary.cuda_event_ms):"NA")
        <<"\n"<<log_prefix<<" status="<<status<<" error_code="<<error_code<<'\n';
    write_text_new(log_path,log.str());

    // 所有CSV落盘后重新读取，终端与完整日志只消费CSV字段，不使用内存统计量。
    const auto main_rows=read_csv_rows(main_writer.path());
    const auto timing_rows=read_csv_rows(timing_writer.path());
    const auto input_rows=read_csv_rows(input_writer.path());
    const auto dtype_rows=read_csv_rows(dtype_writer.path());
    const CsvRow& cpu= device_row(main_rows,"CPU");
    const CsvRow& gpu= device_row(main_rows,"GPU");
    const CsvRow& cpu_timing=device_row(timing_rows,"CPU");
    const CsvRow& gpu_timing=device_row(timing_rows,"GPU");
    if(dtype_rows.size()!=1) throw std::runtime_error("dtype evidence CSV row count mismatch");
    const CsvRow& dtype=dtype_rows.front();
    const auto identity=[&](const CsvRow& row){return std::vector<std::string>{
        csv_value(row,"target"),csv_value(row,"scale_id"),csv_value(row,"device"),
        csv_value(row,"backend"),csv_value(row,"dtype"),csv_value(row,"warmup_runs"),
        csv_value(row,"measured_runs"),csv_value(row,"timing_scope"),csv_value(row,"status"),
        csv_value(row,"error_code")};};
    const auto timing=[&](const CsvRow& row){return std::vector<std::string>{
        csv_value(row,"device"),csv_value(row,"mean_ms"),csv_value(row,"p50_ms"),
        csv_value(row,"p95_ms"),csv_value(row,"p99_ms"),csv_value(row,"min_ms"),
        csv_value(row,"max_ms"),csv_value(row,"std_ms"),csv_value(row,"cv"),
        csv_value(row,"cpu_gpu_speedup")};};
    std::ostringstream terminal;
    terminal<<"[remaining_operator_case] identity and status\n"<<zkx::common::render_table(
        {"target","scale_id","device","backend","dtype","warmup","measured","timing_scope","status","error_code"},
        {identity(cpu),identity(gpu)});
    terminal<<"\n[remaining_operator_case] timing detail\n"<<zkx::common::render_table(
        {"device","mean_ms","p50_ms","p95_ms","p99_ms","min_ms","max_ms","std_ms","cv","speedup"},
        {timing(cpu),timing(gpu)});
    terminal<<"\n[remaining_operator_case] accuracy detail\n"<<zkx::common::render_table(
        {"accuracy_reference","mse","rmse","relative_l2","relative_linf","exact_match","mismatch_count","semantic_check","status"},
        {{csv_value(gpu,"accuracy_reference"),csv_value(gpu,"mse"),csv_value(gpu,"rmse"),
          csv_value(gpu,"relative_l2"),csv_value(gpu,"relative_linf"),csv_value(gpu,"exact_match"),
          csv_value(gpu,"mismatch_count"),csv_value(gpu,"semantic_check"),csv_value(gpu,"accuracy_status")}});
    std::vector<std::vector<std::string>> resource_rows;
    for(const auto* row:{&cpu,&gpu}) for(const auto& metric:std::vector<std::pair<std::string,std::string>>{
        {"cpu_heap","cpu_heap"},{"rss","rss"},{"gpu","gpu_memory"}}) {
        if(csv_value(*row,metric.first+"_before_bytes")!="NA") resource_rows.push_back({
            csv_value(*row,"device"),metric.second,csv_value(*row,metric.first+"_before_bytes"),
            csv_value(*row,metric.first+"_after_bytes"),csv_value(*row,metric.first+"_peak_bytes"),
            csv_value(*row,metric.first+"_delta_bytes")});
    }
    terminal<<"\n[remaining_operator_case] resource detail\n"<<zkx::common::render_table(
        {"device","metric","before_bytes","after_bytes","peak_bytes","delta_bytes"},resource_rows);
    terminal<<"\n[remaining_operator_case] input/output identity\n"<<zkx::common::render_table(
        {"device","input_digest","output_digest"},
        {{"CPU",csv_value(cpu_timing,"input_digest"),csv_value(cpu_timing,"output_digest")},
         {"GPU",csv_value(gpu_timing,"input_digest"),csv_value(gpu_timing,"output_digest")}});
    std::vector<std::vector<std::string>> input_table_rows;
    for(const auto& row:input_rows) input_table_rows.push_back({csv_value(row,"input_name"),csv_value(row,"shape"),
        csv_value(row,"dtype"),csv_value(row,"element_count"),csv_value(row,"generator"),
        csv_value(row,"preview_heacuda_api_tail2")});
    terminal<<"\n[remaining_operator_case] input content\n"<<zkx::common::render_table(
        {"input_name","shape","dtype","elements","generator","values(head4/tail2)"},input_table_rows);
    terminal<<"\n[remaining_operator_case] case parameters\n"<<zkx::common::render_table(
        {"parameters"},{{csv_value(dtype,"case_parameters_json")}});
    terminal<<"\n[remaining_operator_case] dtype evidence\n"<<zkx::common::render_table(
        {"requested_dtype","config_input_dtype","output_dtype","actual_input_digest","status"},
        {{csv_value(dtype,"requested_dtype"),csv_value(dtype,"config_input_dtype"),
          csv_value(dtype,"output_dtype"),csv_value(dtype,"actual_input_digest"),csv_value(dtype,"status")}});
    terminal<<"\n[remaining_operator_case] evidence files\n"<<zkx::common::render_table(
        {"kind","path"},
        {{"main_csv",main_writer.path().string()},{"timing_csv",timing_writer.path().string()},
         {"memory_csv",memory_writer.path().string()},{"input_csv",input_writer.path().string()},
         {"dtype_csv",dtype_writer.path().string()},
         {"module_csv",module_writer.path().string()},{"report_json",report_path.string()},
         {"summary_txt",summary_path.string()},{"full_log",log_path.string()},
         {"results_dir",output_dir.string()}});
    { std::ofstream append(log_path,std::ios::app|std::ios::binary);
      if(!append) throw std::runtime_error("cannot append "+log_path.string()); append<<terminal.str(); }
    std::cout<<log.str()<<terminal.str()<<log_prefix<<" results="<<output_dir.string()<<'\n';
    return passed?0:1;
}

int dispatch(const Cli& cli,const OperatorConfig& config,const ScaleConfig& scale,const Thresholds& thresholds)
{
    if(scale.dtype=="FP32") return execute<float>(cli,config,scale,thresholds);
    if(scale.dtype=="FP16") return execute<__half>(cli,config,scale,thresholds);
    if(scale.dtype=="INT32") return execute<std::int32_t>(cli,config,scale,thresholds);
    if(scale.dtype=="INT16") return execute<std::int16_t>(cli,config,scale,thresholds);
    if(scale.dtype=="INT8") return execute<std::int8_t>(cli,config,scale,thresholds);
    throw std::invalid_argument("unsupported dtype in selected scale: "+scale.dtype);
}

}  // namespace

int main(int argc,char** argv)
{
    try {
        const Cli cli=parse_cli(argc,argv);
        const OperatorConfig config=load_operator_config(cli.config_file);
        const ScaleConfig& scale=select_scale(config,cli.scale_id);
        const Thresholds thresholds=load_thresholds(cli.thresholds_file,scale.dtype);
        return dispatch(cli,config,scale,thresholds);
    } catch(const std::invalid_argument& error) {
        std::cerr<<"[remaining_operator_case-"<<kBatch<<"][INVALID_ARGUMENT] "<<error.what()<<'\n'; return 2;
    } catch(const std::exception& error) {
        std::cerr<<"[remaining_operator_case-"<<kBatch<<"][EXECUTION_FAILED] "<<error.what()<<'\n'; return 1;
    }
}
