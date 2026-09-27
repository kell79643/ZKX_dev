#pragma once
#include <complex>
#include <cstdint>
#include <vector>
namespace zkx::common {
struct Accuracy { double mse{}, rmse{}, relative_l2{}, linf{}, relative_linf{}; };
Accuracy compare(const std::vector<double>& reference, const std::vector<double>& actual, double floor = 1e-30);
Accuracy compare(const std::vector<std::complex<double>>& reference,
                 const std::vector<std::complex<double>>& actual, double floor = 1e-30);
struct DiscreteAccuracy { bool exact{}; std::uint64_t mismatch_count{}; };
DiscreteAccuracy compare_discrete(const std::vector<std::int64_t>& reference,
                                  const std::vector<std::int64_t>& actual);
}  // namespace zkx::common
