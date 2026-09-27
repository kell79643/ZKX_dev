#include "accuracy.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace zkx::common {
template <class T>
Accuracy compare_impl(const std::vector<T>& reference, const std::vector<T>& actual, double floor)
{
    if (reference.empty() || reference.size() != actual.size() || floor <= 0)
        throw std::invalid_argument("accuracy inputs invalid");
    double squared = 0, reference_squared = 0, linf = 0, reference_linf = 0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double error = std::abs(actual[i] - reference[i]);
        squared += error * error;
        const double reference_magnitude = std::abs(reference[i]);
        reference_squared += reference_magnitude * reference_magnitude;
        reference_linf = std::max(reference_linf, reference_magnitude);
        linf = std::max(linf, error);
    }
    const double mse = squared / reference.size();
    return {mse, std::sqrt(mse), std::sqrt(squared) / std::max(std::sqrt(reference_squared), floor),
            linf, linf / std::max(reference_linf, floor)};
}
Accuracy compare(const std::vector<double>& reference, const std::vector<double>& actual, double floor)
{ return compare_impl(reference, actual, floor); }
Accuracy compare(const std::vector<std::complex<double>>& reference,
                 const std::vector<std::complex<double>>& actual, double floor)
{ return compare_impl(reference, actual, floor); }
DiscreteAccuracy compare_discrete(const std::vector<std::int64_t>& reference,
                                  const std::vector<std::int64_t>& actual)
{
    if (reference.size() != actual.size()) throw std::invalid_argument("discrete size mismatch");
    std::uint64_t mismatch = 0;
    for (std::size_t i = 0; i < reference.size(); ++i) mismatch += reference[i] != actual[i];
    return {mismatch == 0, mismatch};
}
}  // namespace zkx::common
