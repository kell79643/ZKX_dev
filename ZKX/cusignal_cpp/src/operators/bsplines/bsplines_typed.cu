#include <cusignal/operators/bsplines/bsplines_typed.h>
#include "bsplines_kernels.cuh"

#include <cusignal/runtime/errors.h>
#include <cusignal/runtime/cuda_utils.h>

#include <string>
#include <type_traits>

namespace cusignal {
namespace {

template <typename T>
void require_compatible_buffers(const DeviceArray<T>& x, const DeviceArray<T>& y, const char* name)
{
    static_assert(detail::is_bspline_input_v<T>, "unsupported B-spline business input dtype");
    if (x.size() != y.size()) {
        throw_error(ErrorCode::INVALID_SHAPE,
            std::string(name) + "输入输出shape不一致：input_size=" +
                std::to_string(x.size()) + "，output_size=" + std::to_string(y.size()));
    }
}

}  // namespace

template <typename T>
void cubic_device(const DeviceArray<T>& x, DeviceArray<T>& y)
{
    require_compatible_buffers(x, y, "cubic_device");
    if (!x.empty()) {
        cuda_utils::launch_1d_kernel(
            bsplines_detail::cubic_kernel<T>, x.size(), x.data(), y.data(), x.size());
    }
}

template <typename T>
void gauss_spline_device(const DeviceArray<T>& x, DeviceArray<T>& y, int n)
{
    require_compatible_buffers(x, y, "gauss_spline_device");
    if (n < 0) {
        throw_error(ErrorCode::INVALID_ARGUMENT,
            "gauss_spline参数无效：n必须非负，实际为" + std::to_string(n));
    }
    if (!x.empty()) {
        const float signsq = static_cast<float>(n + 1) / 12.0F;
        constexpr float pi = 3.14159265358979323846F;
        const float r_signsq = 0.5F / signsq;
        const float norm = 1.0F / sqrtf(2.0F * pi * signsq);
        cuda_utils::launch_1d_kernel(
            bsplines_detail::gauss_spline_kernel<T>,
            x.size(), x.data(), y.data(), x.size(), r_signsq, norm);
    }
}

template <typename T>
void quadratic_device(const DeviceArray<T>& x, DeviceArray<T>& y)
{
    require_compatible_buffers(x, y, "quadratic_device");
    if (!x.empty()) {
        cuda_utils::launch_1d_kernel(
            bsplines_detail::quadratic_kernel<T>, x.size(), x.data(), y.data(), x.size());
    }
}

#define INSTANTIATE_BSPLINE_GPU(T) \
    template void cubic_device(const DeviceArray<T>&, DeviceArray<T>&); \
    template void gauss_spline_device(const DeviceArray<T>&, DeviceArray<T>&, int); \
    template void quadratic_device(const DeviceArray<T>&, DeviceArray<T>&)

INSTANTIATE_BSPLINE_GPU(float);
INSTANTIATE_BSPLINE_GPU(__half);
INSTANTIATE_BSPLINE_GPU(std::int32_t);
INSTANTIATE_BSPLINE_GPU(std::int16_t);
INSTANTIATE_BSPLINE_GPU(std::int8_t);

#undef INSTANTIATE_BSPLINE_GPU

}  // namespace cusignal
