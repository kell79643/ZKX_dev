#include <cusignal/operators/demod/demod_typed.h>
#include "demod_kernels.cuh"

#include <cusignal/runtime/cuda_utils.h>

#include <limits>
#include <stdexcept>

namespace cusignal {
namespace {

bool same_workspace(
    const FmDemodWorkspace& left,
    const FmDemodWorkspace& right)
{
    return left.input_shape == right.input_shape &&
           left.output_shape == right.output_shape &&
           left.axis == right.axis &&
           left.axis_length == right.axis_length &&
           left.output_axis_length == right.output_axis_length &&
           left.inner_size == right.inner_size &&
           left.outer_size == right.outer_size &&
           left.input_count == right.input_count &&
           left.output_count == right.output_count;
}

FmDemodWorkspace validated_workspace(const FmDemodWorkspace& workspace)
{
    FmDemodOptions options;
    options.shape = workspace.input_shape;
    options.axis = workspace.axis;
    FmDemodWorkspace validated(options);
    if (!same_workspace(workspace, validated)) {
        throw std::invalid_argument("fm_demod workspace mismatch");
    }
    return validated;
}

FmDemodWorkspace one_dimensional_workspace(std::size_t count)
{
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("fm_demod input exceeds 32-bit indexing");
    }
    FmDemodOptions options;
    options.shape = {static_cast<int>(count)};
    return FmDemodWorkspace(options);
}

template <typename Complex>
void launch_fm_demod(
    const DeviceArray<Complex>& x,
    DeviceArray<float>& y,
    const FmDemodWorkspace& workspace)
{
    const FmDemodWorkspace layout = validated_workspace(workspace);
    if (x.size() != layout.input_size()) {
        throw std::invalid_argument("fm_demod_device input shape mismatch");
    }
    if (y.size() != layout.output_size()) {
        throw std::invalid_argument("fm_demod_device output size mismatch");
    }
    if (!y.empty() && reinterpret_cast<const void*>(x.data()) ==
                          reinterpret_cast<const void*>(y.data())) {
        throw std::invalid_argument("fm_demod_device does not permit input/output aliasing");
    }
    if (y.empty()) return;

    cuda_utils::launch_1d_kernel(
        demod_detail::fm_demod_nd_kernel<Complex>,
        y.size(),
        x.data(),
        y.data(),
        static_cast<int>(layout.output_size()),
        layout.axis_length,
        layout.output_axis_length,
        layout.inner_size);
}

}  // namespace

template <typename T>
void fm_demod_device(
    const DeviceArray<DemodComplex<T>>& x,
    DeviceArray<float>& y,
    const FmDemodWorkspace& workspace)
{
    static_assert(detail::is_demod_input_v<T>, "unsupported fm_demod business input dtype");
    launch_fm_demod(x, y, workspace);
}

void fm_demod_device(
    const DeviceArray<ComplexFloat>& x,
    DeviceArray<float>& y,
    const FmDemodWorkspace& workspace)
{
    launch_fm_demod(x, y, workspace);
}

template <typename T>
void fm_demod_device(
    const DeviceArray<DemodComplex<T>>& x,
    DeviceArray<float>& y)
{
    static_assert(detail::is_demod_input_v<T>, "unsupported fm_demod business input dtype");
    fm_demod_device(x, y, one_dimensional_workspace(x.size()));
}

void fm_demod_device(
    const DeviceArray<ComplexFloat>& x,
    DeviceArray<float>& y)
{
    fm_demod_device(x, y, one_dimensional_workspace(x.size()));
}

template <typename T>
void fm_demod_device(
    const DeviceArray<DemodComplex<T>>& x,
    DeviceArray<float>& y,
    int rows,
    int cols,
    int axis)
{
    static_assert(detail::is_demod_input_v<T>, "unsupported fm_demod business input dtype");
    FmDemodOptions options;
    options.shape = {rows, cols};
    options.axis = axis;
    fm_demod_device(x, y, FmDemodWorkspace(options));
}

#define INSTANTIATE_DEMOD_GPU(T) \
    template void fm_demod_device(const DeviceArray<DemodComplex<T>>&, DeviceArray<float>&, const FmDemodWorkspace&); \
    template void fm_demod_device(const DeviceArray<DemodComplex<T>>&, DeviceArray<float>&); \
    template void fm_demod_device(const DeviceArray<DemodComplex<T>>&, DeviceArray<float>&, int, int, int)

INSTANTIATE_DEMOD_GPU(float);
INSTANTIATE_DEMOD_GPU(__half);
INSTANTIATE_DEMOD_GPU(std::int32_t);
INSTANTIATE_DEMOD_GPU(std::int16_t);
INSTANTIATE_DEMOD_GPU(std::int8_t);

#undef INSTANTIATE_DEMOD_GPU

}  // namespace cusignal
