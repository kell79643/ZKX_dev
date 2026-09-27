#include "identity.h"
#include <regex>
#include <stdexcept>

namespace zkx::common {
std::string backend_name(Backend value)
{
    if (value == Backend::fft_thrust) return "fft_thrust";
    if (value == Backend::dlfft) return "dlfft";
    return "not_applicable";
}
Backend parse_backend(const std::string& value)
{
    if (value == "fft_thrust") return Backend::fft_thrust;
    if (value == "dlfft") return Backend::dlfft;
    if (value == "not_applicable") return Backend::not_applicable;
    throw std::invalid_argument("unsupported backend: " + value);
}
bool valid_scale_id(const std::string& value)
{
    static const std::regex pattern("^[a-z0-9][a-z0-9._-]{2,63}$");
    return std::regex_match(value, pattern);
}
std::string make_run_id(const std::string& utc, const std::string& git,
                        const std::string& profile, const std::string& random)
{
    if (!std::regex_match(utc, std::regex("^[0-9]{8}T[0-9]{6}Z$")) || git.size() < 12 ||
        !valid_scale_id(profile) || !std::regex_match(random, std::regex("^[0-9a-f]{8}$")))
        throw std::invalid_argument("invalid run_id component");
    return utc + "_" + git.substr(0, 12) + "_" + profile + "_" + random;
}
std::string make_case_id(const std::string& target, const std::string& scale,
                         const std::string& digest)
{
    if (!valid_scale_id(target) || !valid_scale_id(scale) ||
        !std::regex_match(digest, std::regex("^[0-9a-f]{64}$")))
        throw std::invalid_argument("invalid case_id component");
    return target + "__" + scale + "__" + digest.substr(0, 16);
}
}  // namespace zkx::common
