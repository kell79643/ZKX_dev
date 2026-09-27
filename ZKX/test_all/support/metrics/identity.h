#pragma once
#include <string>

namespace zkx::common {
enum class Backend { fft_thrust, dlfft, not_applicable };
std::string backend_name(Backend value);
Backend parse_backend(const std::string& value);
bool valid_scale_id(const std::string& value);
std::string make_run_id(const std::string& utc_compact, const std::string& git_commit,
                        const std::string& profile, const std::string& random_hex8);
std::string make_case_id(const std::string& target_slug, const std::string& scale_id,
                         const std::string& input_sha256);
}  // namespace zkx::common
