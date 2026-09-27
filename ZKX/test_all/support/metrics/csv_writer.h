#pragma once
#include <filesystem>
#include <string>
#include <vector>
#include "identity.h"
namespace zkx::common {
std::string csv_escape(const std::string& value);
std::string nullable_csv(bool available, const std::string& value);
std::filesystem::path backend_csv_path(const std::filesystem::path& directory,
                                       const std::string& stem, Backend backend);
class CsvWriter {
public:
    CsvWriter(std::filesystem::path path, std::vector<std::string> columns);
    void append(const std::vector<std::string>& values);
    const std::filesystem::path& path() const { return path_; }
private:
    std::filesystem::path path_;
    std::vector<std::string> columns_;
};
}  // namespace zkx::common
