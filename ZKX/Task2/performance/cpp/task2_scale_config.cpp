#include "task2_scale_config.h"

#include <demo/config_file.h>
#include "test_all/support/config/json_value.h"

#include <cmath>
#include <fstream>
#include <iterator>
#include <sstream>
#include <vector>

namespace task2 {
namespace {

using zkx::config::JsonValue;

const JsonValue& field(const JsonValue& value, const char* name)
{
    const auto* result = value.find(name);
    if (!result) throw std::invalid_argument(std::string("missing field: ") + name);
    return *result;
}

int integer(const JsonValue& value, const char* name)
{
    const double result = field(value, name).as_number();
    if (!std::isfinite(result) || std::floor(result) != result)
        throw std::invalid_argument(std::string(name) + " must be integer");
    return static_cast<int>(result);
}

std::uint32_t uint32(const JsonValue& value, const char* name)
{
    const double result = field(value, name).as_number();
    if (!std::isfinite(result) || std::floor(result) != result ||
        result < 0 || result > 4294967295.0)
        throw std::invalid_argument(std::string(name) + " must be uint32");
    return static_cast<std::uint32_t>(result);
}

float f32(const JsonValue& value, const char* name)
{
    const double result = field(value, name).as_number();
    if (!std::isfinite(result))
        throw std::invalid_argument(std::string(name) + " must be finite");
    return static_cast<float>(result);
}

std::string file_sha(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::invalid_argument("cannot open task scales: " + path.string());
    const std::vector<unsigned char> bytes{
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    return demo_config::sha256(bytes);
}

}  // namespace

Task2ScaleConfig load_task2_scale(
    const std::filesystem::path& requested, const std::string& wanted)
{
    const auto path = std::filesystem::absolute(requested);
    const auto root = JsonValue::parse_file(path.string());
    const std::string scale_set_id = field(root, "scale_set_id").as_string();
    const auto& scales = field(field(field(root, "tasks"), "Task2"), "scales").as_array();
    for (const auto& scale : scales) {
        if (field(scale, "scale_id").as_string() != wanted) continue;
        const auto& shape = field(scale, "input_shape").as_array();
        if (shape.size() != 1) throw std::invalid_argument("Task2 shape must have 1 axis");
        const auto& parameters = field(scale, "parameters");
        const std::string sha = file_sha(path);
        TaskConfig config{path, sha,
            integer(parameters, "samples"), f32(parameters, "sample_rate_hz"),
            integer(parameters, "target_delay_samples"), uint32(parameters, "noise_seed"),
            f32(parameters, "target_weight"), f32(parameters, "gaussian_weight"),
            f32(parameters, "sawtooth_weight"), f32(parameters, "square_weight"),
            f32(parameters, "noise_amplitude"), integer(parameters, "filter_taps"),
            f32(parameters, "filter_cutoff_hz"), integer(parameters, "step3_crop_each"),
            f32(parameters, "fusion_weight_fm"),
            f32(parameters, "fusion_weight_correlation"),
            f32(parameters, "fusion_weight_spectral"),
            f32(parameters, "fusion_weight_wavelet"),
            integer(parameters, "argrelextrema_order"),
            integer(parameters, "kalman_observation_count"),
            f32(parameters, "kalman_F"), f32(parameters, "kalman_Q"),
            f32(parameters, "kalman_H"), f32(parameters, "kalman_R"),
            f32(parameters, "kalman_measurement_noise_variant")};
        config.validate();
        const auto elements = integer(scale, "actual_elements");
        if (elements != config.samples)
            throw std::invalid_argument("actual_elements mismatch");
        if (shape[0].as_number() != config.samples)
            throw std::invalid_argument("input_shape does not match Task2 parameters");
        std::ostringstream identity;
        identity << wanted << '|' << config.samples << '|' << config.sample_rate_hz
                 << '|' << config.target_delay_samples << '|' << config.noise_seed
                 << '|' << config.target_weight << '|' << config.gaussian_weight
                 << '|' << config.sawtooth_weight << '|' << config.square_weight
                 << '|' << config.noise_amplitude << '|' << config.filter_taps
                 << '|' << config.filter_cutoff_hz << '|' << config.step3_crop_each
                 << '|' << config.fusion_weight_fm << '|'
                 << config.fusion_weight_correlation << '|'
                 << config.fusion_weight_spectral << '|'
                 << config.fusion_weight_wavelet << '|'
                 << config.argrelextrema_order << '|'
                 << config.kalman_observation_count << '|' << config.kalman_f
                 << '|' << config.kalman_q << '|' << config.kalman_h << '|'
                 << config.kalman_r << '|' << config.kalman_measurement_noise_variant;
        const auto text = identity.str();
        const std::vector<unsigned char> bytes(text.begin(), text.end());
        return {wanted, scale_set_id,
            field(scale, "order_of_magnitude").as_string(),
            "[" + std::to_string(config.samples) + "]",
            demo_config::sha256(bytes), elements, std::move(config)};
    }
    throw std::invalid_argument("Task2 scale not found: " + wanted);
}

}  // namespace task2
