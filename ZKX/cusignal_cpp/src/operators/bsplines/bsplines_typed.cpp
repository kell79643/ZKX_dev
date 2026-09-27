#include <cusignal/operators/bsplines/bsplines_typed.h>
#include <cusignal/runtime/errors.h>

#include <type_traits>

namespace cusignal {
namespace {

template <typename T, typename Evaluator>
std::vector<T> evaluate_cpu(const std::vector<T>& x, Evaluator evaluator)
{
    static_assert(detail::is_bspline_input_v<T>, "unsupported B-spline business input dtype");
    std::vector<T> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] = evaluator(x[i]);
    }
    return y;
}

}  // namespace

template <typename T>
std::vector<T> cubic_cpu(const std::vector<T>& x)
{
    return evaluate_cpu(x, [](T value) { return detail::cubic_value(value); });
}

template <typename T>
std::vector<T> cubic(const std::vector<T>& x)
{
    return cubic_cpu(x);
}

template <typename T>
std::vector<T> gauss_spline_cpu(const std::vector<T>& x, int n)
{
    if (n < 0) {
        throw_error(ErrorCode::INVALID_ARGUMENT,
            "gauss_spline参数无效：n必须非负，实际为" + std::to_string(n));
    }
    return evaluate_cpu(x, [n](T value) { return detail::gauss_spline_value(value, n); });
}

template <typename T>
std::vector<T> gauss_spline(const std::vector<T>& x, int n)
{
    return gauss_spline_cpu(x, n);
}

template <typename T>
std::vector<T> quadratic_cpu(const std::vector<T>& x)
{
    return evaluate_cpu(x, [](T value) { return detail::quadratic_value(value); });
}

template <typename T>
std::vector<T> quadratic(const std::vector<T>& x)
{
    return quadratic_cpu(x);
}

#define INSTANTIATE_BSPLINE_CPU(T) \
    template std::vector<T> cubic_cpu(const std::vector<T>&); \
    template std::vector<T> cubic(const std::vector<T>&); \
    template std::vector<T> gauss_spline_cpu(const std::vector<T>&, int); \
    template std::vector<T> gauss_spline(const std::vector<T>&, int); \
    template std::vector<T> quadratic_cpu(const std::vector<T>&); \
    template std::vector<T> quadratic(const std::vector<T>&)

INSTANTIATE_BSPLINE_CPU(float);
INSTANTIATE_BSPLINE_CPU(__half);
INSTANTIATE_BSPLINE_CPU(std::int32_t);
INSTANTIATE_BSPLINE_CPU(std::int16_t);
INSTANTIATE_BSPLINE_CPU(std::int8_t);

#undef INSTANTIATE_BSPLINE_CPU

}  // namespace cusignal
