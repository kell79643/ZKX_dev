#include <task2/task2_runner.h>
#include <task2/task2_config.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int parse_positive(const char* value, const char* name)
{
    const int parsed = std::stoi(value);
    if (parsed < 1) throw std::invalid_argument(std::string(name) + " must be >= 1");
    return parsed;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        std::filesystem::path output_root;
        std::string config_path;
        int warmup = 5;
        int repeats = 20;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--benchmark-output-root" && index + 1 < argc)
                output_root = argv[++index];
            else if (argument == "--config" && index + 1 < argc)
                config_path = argv[++index];
            else if (argument == "--warmup" && index + 1 < argc)
                warmup = parse_positive(argv[++index], "warmup");
            else if (argument == "--repeats" && index + 1 < argc)
                repeats = parse_positive(argv[++index], "repeats");
            else
                throw std::invalid_argument("unknown/incomplete Task2 benchmark argument: " + argument);
        }
        if (config_path.empty())
            throw std::invalid_argument("--config is required; no compiled fallback exists");
        const auto config = task2::TaskConfig::load(config_path);
        if (output_root.empty()) return task2::run_task2_until(6, config);
        output_root = std::filesystem::absolute(output_root);
        std::filesystem::create_directories(output_root);
        const auto original_directory = std::filesystem::current_path();
        auto run_group = [&](const char* group, int count) {
            for (int run = 1; run <= count; ++run) {
                const auto directory = output_root / group / ("run_" + std::to_string(run));
                std::filesystem::create_directories(directory);
                std::filesystem::current_path(directory);
                const int status = task2::run_task2_until(6, config);
                std::filesystem::current_path(original_directory);
                if (status != 0) return status;
            }
            return 0;
        };
        if (const int status = run_group("warmup", warmup); status != 0) return status;
        if (const int status = run_group("measured", repeats); status != 0) return status;
        std::cout << "[TASK2][BENCHMARK_BATCH] warmup=" << warmup
                  << " measured=" << repeats << " output_root=" << output_root.string()
                  << " status=pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[TASK2][BENCHMARK_BATCH][FAIL] " << error.what() << '\n';
        return 2;
    }
}
