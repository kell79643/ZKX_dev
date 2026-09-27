#include "csv_writer.h"
#include <fstream>
#include <stdexcept>
namespace zkx::common {
std::string csv_escape(const std::string& value)
{
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string out = "\"";
    for (char c : value) out += c == '"' ? "\"\"" : std::string(1, c);
    return out + '"';
}
std::string nullable_csv(bool available, const std::string& value) { return available ? value : "NA"; }
std::filesystem::path backend_csv_path(const std::filesystem::path& directory,
                                       const std::string& stem, Backend backend)
{
    if (!valid_scale_id(stem)) throw std::invalid_argument("invalid CSV stem");
    return directory / (stem + "_" + backend_name(backend) + ".csv");
}
static void write_row(std::ofstream& stream, const std::vector<std::string>& values)
{
    for (std::size_t i = 0; i < values.size(); ++i) stream << (i ? "," : "") << csv_escape(values[i]);
    stream << '\n';
    if (!stream) throw std::runtime_error("CSV write failed");
}
CsvWriter::CsvWriter(std::filesystem::path path, std::vector<std::string> columns)
    : path_(std::move(path)), columns_(std::move(columns))
{
    if (columns_.empty() || std::filesystem::exists(path_))
        throw std::invalid_argument("CSV path exists or columns empty");
    std::filesystem::create_directories(path_.parent_path());
    std::ofstream stream(path_, std::ios::out | std::ios::binary);
    if (!stream) throw std::runtime_error("CSV create failed");
    write_row(stream, columns_);
}
void CsvWriter::append(const std::vector<std::string>& values)
{
    if (values.size() != columns_.size()) throw std::invalid_argument("CSV column count mismatch");
    std::ofstream stream(path_, std::ios::app | std::ios::binary);
    if (!stream) throw std::runtime_error("CSV append failed");
    write_row(stream, values);
}
}  // namespace zkx::common
