#include <task2/task2_common.h>

#include <algorithm>
#include <numeric>

namespace task2 {

double milliseconds(Clock::time_point begin, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

double rms(const std::vector<float>& values)
{
    if (values.empty()) return 0.0;
    double sum = 0.0;
    for (float value : values) sum += static_cast<double>(value) * value;
    return std::sqrt(sum / static_cast<double>(values.size()));
}

double roughness(const std::vector<float>& values)
{
    if (values.size() < 3) return 0.0;
    double sum = 0.0;
    for (std::size_t index = 1; index + 1 < values.size(); ++index) {
        const double second = values[index - 1] - 2.0 * values[index] + values[index + 1];
        sum += second * second;
    }
    return std::sqrt(sum / static_cast<double>(values.size() - 2));
}

std::vector<float> normalize_abs(const std::vector<float>& input)
{
    std::vector<float> output(input.size());
    float maximum = 0.0F;
    for (float value : input) maximum = std::max(maximum, std::abs(value));
    if (maximum == 0.0F) return output;
    for (std::size_t index = 0; index < input.size(); ++index)
        output[index] = std::abs(input[index]) / maximum;
    return output;
}

std::vector<float> resample_linear(
    const std::vector<float>& input, std::size_t count)
{
    std::vector<float> output(count, 0.0F);
    if (input.empty() || count == 0) return output;
    if (input.size() == 1 || count == 1) {
        std::fill(output.begin(), output.end(), input.front());
        return output;
    }
    for (std::size_t index = 0; index < count; ++index) {
        const double position = static_cast<double>(index) * (input.size() - 1) / (count - 1);
        const std::size_t left = static_cast<std::size_t>(position);
        const std::size_t right = std::min(left + 1, input.size() - 1);
        const float fraction = static_cast<float>(position - left);
        output[index] = input[left] + fraction * (input[right] - input[left]);
    }
    return output;
}

}  // namespace task2
