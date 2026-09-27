#pragma once

#include <task2/task2_config.h>

#include <filesystem>
#include <string>

namespace task2 {

struct Task2ScaleConfig {
    std::string scale_id;
    std::string scale_set_id;
    std::string order_of_magnitude;
    std::string input_shape;
    std::string input_digest;
    std::int64_t actual_elements{};
    TaskConfig config;
};

Task2ScaleConfig load_task2_scale(
    const std::filesystem::path& path, const std::string& scale_id);

}  // namespace task2
