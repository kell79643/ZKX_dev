#pragma once
#include <string>
#include <vector>
namespace zkx::common {
std::string render_table(const std::vector<std::string>& headers,
                         const std::vector<std::vector<std::string>>& rows);
}  // namespace zkx::common
