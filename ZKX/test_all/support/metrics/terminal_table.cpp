#include "terminal_table.h"
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>
namespace zkx::common {
std::string render_table(const std::vector<std::string>& headers,
                         const std::vector<std::vector<std::string>>& rows)
{
    if (headers.empty()) throw std::invalid_argument("table headers empty");
    std::vector<std::size_t> widths(headers.size());
    for (std::size_t i = 0; i < headers.size(); ++i) widths[i] = headers[i].size();
    for (const auto& row : rows) {
        if (row.size() != headers.size()) throw std::invalid_argument("table column mismatch");
        for (std::size_t i = 0; i < row.size(); ++i) widths[i] = std::max(widths[i], row[i].size());
    }
    std::ostringstream out;
    auto emit = [&](const auto& row) {
        for (std::size_t i = 0; i < row.size(); ++i)
            out << (i ? " | " : "") << std::left << std::setw(static_cast<int>(widths[i])) << row[i];
        out << '\n';
    };
    emit(headers);
    for (const auto& row : rows) emit(row);
    return out.str();
}
}  // namespace zkx::common
