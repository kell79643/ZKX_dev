#pragma once

#include <task2/task2_config.h>
#include <task2/task2_runner.h>

#include <iostream>
#include <stdexcept>
#include <string>

namespace task2 {

inline int run_configured_entry(int argc, char** argv, int stop_after)
{
    try {
        std::string config_path;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--config" && index + 1 < argc) config_path = argv[++index];
            else throw std::invalid_argument("unknown/incomplete Task2 argument: " + argument);
        }
        if (config_path.empty())
            throw std::invalid_argument("--config is required; no compiled fallback exists");
        return run_task2_until(stop_after, TaskConfig::load(config_path));
    } catch (const std::exception& error) {
        std::cerr << "[TASK2][ENTRY][FAIL] " << error.what() << '\n';
        return 2;
    }
}

}  // namespace task2
