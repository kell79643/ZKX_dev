#pragma once
#include <task1/task1_config.h>
#include <filesystem>
#include <string>
namespace task1 {
struct Task1ScaleConfig {
    std::string scale_id, scale_set_id, order_of_magnitude, input_shape, input_digest;
    std::int64_t actual_elements{};
    TaskConfig config;
};
Task1ScaleConfig load_task1_scale(const std::filesystem::path& path, const std::string& scale_id);
}  // namespace task1
