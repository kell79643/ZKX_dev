#include <cusignal/operators/convolution/convolution_typed.h>
#include "convolution_kernels.cuh"

#include <cusignal/runtime/cuda_utils.h>

#include <algorithm>
#include <stdexcept>

namespace cusignal {
namespace {

std::size_t typed_output_start(std::size_t len1, std::size_t len2, std::size_t out_len)
{
    return (len1 + len2 - 1 - out_len) / 2;
}

int next_power_of_two(int value)
{
    int result = 1;
    while (result < value) {
        if (result > std::numeric_limits<int>::max() / 2) {
            throw std::invalid_argument("typed convolution FFT length exceeds 32-bit plan capacity");
        }
        result *= 2;
    }
    return result;
}

int checked_nd_elements(const std::vector<int>& shape, const char* label)
{
    long long total = 1;
    for (int dimension : shape) {
        if (dimension < 0) {
            throw std::invalid_argument(
                std::string("typed correlate negative ") + label + " dimension");
        }
        if (dimension == 0) return 0;
        total *= dimension;
        if (total > std::numeric_limits<int>::max()) {
            throw std::invalid_argument(
                std::string("typed correlate ") + label + " exceeds 32-bit indexing");
        }
    }
    return static_cast<int>(total);
}

std::vector<int> nd_strides(const std::vector<int>& shape)
{
    std::vector<int> strides(shape.size(), 1);
    for (std::size_t axis = shape.size(); axis-- > 1;) {
        const long long stride = static_cast<long long>(strides[axis]) * shape[axis];
        if (stride > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("typed correlate stride exceeds 32-bit indexing");
        }
        strides[axis - 1] = static_cast<int>(stride);
    }
    return strides;
}

void transform_nd(
    DeviceArray<ComplexFloat>& row_major,
    DeviceArray<ComplexFloat>& packed,
    DeviceArray<ComplexFloat>& transformed,
    CorrelateNDWorkspace& workspace,
    bool inverse)
{
    for (int axis = workspace.rank - 1; axis >= 0; --axis) {
        const int axis_length = workspace.fft_shape[axis];
        const int inner_size = workspace.fft_strides[axis];
        const int batch = workspace.fft_elements / axis_length;
        cuda_utils::launch_1d_kernel(
            convolution_detail::pack_nd_axis_kernel,
            static_cast<std::size_t>(workspace.fft_elements),
            row_major.data(), axis_length, inner_size, workspace.fft_elements,
            packed.data());
        if (inverse) {
            workspace.axis_fft[axis]->ifft_batch_device(
                packed, transformed, batch);
        } else {
            workspace.axis_fft[axis]->fft_batch_device(
                packed, transformed, batch);
        }
        cuda_utils::launch_1d_kernel(
            convolution_detail::unpack_nd_axis_kernel,
            static_cast<std::size_t>(workspace.fft_elements),
            transformed.data(), axis_length, inner_size, workspace.fft_elements,
            row_major.data());
    }
}

template <typename T>
void validate_1d(
    const DeviceArray<T>& in1,
    const DeviceArray<T>& in2,
    const DeviceArray<T>& out,
    const std::string& mode)
{
    static_assert(detail::is_convolution_input_v<T>, "unsupported correlate business input dtype");
    const std::size_t expected = correlate_output_size(in1.size(), in2.size(), mode);
    constexpr std::size_t max_index = static_cast<std::size_t>(std::numeric_limits<int>::max());
    const bool full_exceeds_index = !in1.empty() && !in2.empty() &&
        (in1.size() > max_index || in2.size() > max_index ||
         in1.size() > max_index - in2.size() + 1);
    if (in1.size() > max_index || in2.size() > max_index || expected > max_index ||
        full_exceeds_index) {
        throw std::invalid_argument("typed correlate shape exceeds 32-bit device indexing");
    }
    if (out.size() != expected) {
        throw std::invalid_argument("typed correlate output size mismatch");
    }
    if (!out.empty() && (out.data() == in1.data() || out.data() == in2.data())) {
        throw std::invalid_argument("typed correlate does not permit input/output aliasing");
    }
}

template <typename T>
T correlate2d_workspace_fill(const Correlate2DWorkspace& workspace)
{
    if constexpr (std::is_same_v<T, __half>) {
        return workspace.fill_fp16;
    } else if constexpr (std::is_same_v<T, float>) {
        return workspace.fill_fp32;
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        return workspace.fill_int32;
    } else if constexpr (std::is_same_v<T, std::int16_t>) {
        return workspace.fill_int16;
    } else {
        return workspace.fill_int8;
    }
}

}  // namespace

void CorrelateDeviceWorkspace::reset(const CorrelateDeviceParams& params)
{
    if (params.len1 < 0 || params.len2 < 0) {
        throw std::invalid_argument("CorrelateDeviceWorkspace: lengths must be non-negative");
    }
    if (params.mode != "full" && params.mode != "same" && params.mode != "valid") {
        throw std::invalid_argument("CorrelateDeviceWorkspace: mode must be full, same, or valid");
    }

    len1 = params.len1;
    len2 = params.len2;
    mode = params.mode;
    if (len1 > 0 && len2 > 0 && len1 > std::numeric_limits<int>::max() - len2 + 1) {
        throw std::invalid_argument("CorrelateDeviceWorkspace: full length exceeds 32-bit capacity");
    }
    full_len = (len1 == 0 || len2 == 0) ? 0 : len1 + len2 - 1;
    fft_len = full_len > 0 ? next_power_of_two(full_len) : 0;
    out_len = full_len == 0
        ? 0
        : static_cast<int>(correlate_output_size(
              static_cast<std::size_t>(len1), static_cast<std::size_t>(len2), mode));

    const std::size_t count = fft_size();
    in1_time.reset(count);
    in2_time.reset(count);
    in1_freq.reset(count);
    in2_freq.reset(count);
    product_freq.reset(count);
    ifft_time.reset(count);
    fft = fft_len == 0 ? nullptr : std::make_unique<FFTInterface>(fft_len);
}

void CorrelateNDWorkspace::reset(const CorrelateNDOptions& options)
{
    const CorrelateNDOutputShape output = correlate_nd_output_shape(options);
    shape1 = options.shape1;
    shape2 = options.shape2;
    output_shape = output.dimensions;
    requested_method = options.method;
    mode = options.mode;
    rank = static_cast<int>(shape1.size());
    input1_elements = checked_nd_elements(shape1, "in1");
    input2_elements = checked_nd_elements(shape2, "in2");
    output_elements = static_cast<int>(output.size());
    effective_method = requested_method == CorrelateMethod::auto_select
        ? (rank <= 1 ? CorrelateMethod::direct : CorrelateMethod::fft)
        : requested_method;

    full_shape.clear();
    fft_shape.clear();
    crop_start.clear();
    strides1 = nd_strides(shape1);
    strides2 = nd_strides(shape2);
    output_strides = output_elements == 0 ? std::vector<int>{} : nd_strides(output_shape);
    fft_strides.clear();
    axis_fft.clear();
    fft_elements = 0;

    d_shape1.reset();
    d_shape2.reset();
    d_strides1.reset();
    d_strides2.reset();
    d_output_strides.reset();
    d_fft_strides.reset();
    d_crop_start.reset();
    in1_time.reset();
    in2_time.reset();
    in1_freq.reset();
    in2_freq.reset();
    product_freq.reset();
    transform_scratch.reset();

    if (rank == 0 || output_elements == 0 || effective_method == CorrelateMethod::direct) {
        return;
    }

    full_shape.resize(rank);
    fft_shape.resize(rank);
    crop_start.resize(rank);
    for (int axis = 0; axis < rank; ++axis) {
        if (shape1[axis] > std::numeric_limits<int>::max() - shape2[axis] + 1) {
            throw std::invalid_argument("typed correlate full shape exceeds 32-bit indexing");
        }
        full_shape[axis] = shape1[axis] + shape2[axis] - 1;
        fft_shape[axis] = next_power_of_two(full_shape[axis]);
        crop_start[axis] = (full_shape[axis] - output_shape[axis]) / 2;
    }
    fft_strides = nd_strides(fft_shape);
    fft_elements = checked_nd_elements(fft_shape, "FFT workspace");

    d_shape1 = DeviceArray<int>::from_host(shape1);
    d_shape2 = DeviceArray<int>::from_host(shape2);
    d_strides1 = DeviceArray<int>::from_host(strides1);
    d_strides2 = DeviceArray<int>::from_host(strides2);
    d_output_strides = DeviceArray<int>::from_host(output_strides);
    d_fft_strides = DeviceArray<int>::from_host(fft_strides);
    d_crop_start = DeviceArray<int>::from_host(crop_start);

    const std::size_t count = static_cast<std::size_t>(fft_elements);
    in1_time.reset(count);
    in2_time.reset(count);
    in1_freq.reset(count);
    in2_freq.reset(count);
    product_freq.reset(count);
    transform_scratch.reset(count);
    axis_fft.reserve(rank);
    for (int axis = 0; axis < rank; ++axis) {
        axis_fft.push_back(std::make_unique<FFTInterface>(fft_shape[axis]));
    }
}

template <typename T>
void correlate_device(
    const DeviceArray<T>& in1,
    const DeviceArray<T>& in2,
    DeviceArray<T>& out,
    const std::string& mode)
{
    validate_1d(in1, in2, out, mode);
    if (in1.empty() || in2.empty()) return;
    cuda_utils::launch_1d_kernel(
        convolution_detail::correlate_real_direct_kernel<T>,
        out.size(),
        in1.data(),
        static_cast<int>(in1.size()),
        in2.data(),
        static_cast<int>(in2.size()),
        out.data(),
        static_cast<int>(out.size()),
        static_cast<int>(typed_output_start(in1.size(), in2.size(), out.size())));
}

template <typename T>
void correlate_device(
    const DeviceArray<T>& in1,
    const DeviceArray<T>& in2,
    DeviceArray<T>& out,
    CorrelateNDWorkspace& workspace)
{
    static_assert(detail::is_convolution_input_v<T>, "unsupported correlate business input dtype");
    CorrelateNDOptions options;
    options.shape1 = workspace.shape1;
    options.shape2 = workspace.shape2;
    options.mode = workspace.mode;
    options.method = workspace.requested_method;
    const CorrelateNDOutputShape expected = correlate_nd_output_shape(options);
    if (in1.size() != static_cast<std::size_t>(workspace.input1_elements) ||
        in2.size() != static_cast<std::size_t>(workspace.input2_elements) ||
        out.size() != expected.size() ||
        workspace.output_elements != static_cast<int>(expected.size()) ||
        workspace.output_shape != expected.dimensions) {
        throw std::invalid_argument("typed correlate N-D workspace mismatch");
    }
    if (!out.empty() && (out.data() == in1.data() || out.data() == in2.data())) {
        throw std::invalid_argument("typed correlate does not permit input/output aliasing");
    }
    if (out.empty()) return;
    if (workspace.rank == 0) {
        cuda_utils::launch_1d_kernel(
            convolution_detail::correlate_scalar_kernel<T>, 1,
            in1.data(), in2.data(), out.data());
        return;
    }
    if (workspace.effective_method == CorrelateMethod::direct) {
        correlate_device(in1, in2, out, workspace.mode);
        return;
    }
    if (workspace.effective_method != CorrelateMethod::fft ||
        workspace.fft_elements <= 0 ||
        static_cast<int>(workspace.axis_fft.size()) != workspace.rank) {
        throw std::invalid_argument("typed correlate N-D FFT workspace is incomplete");
    }

    CUDA_CHECK(cudaMemset(
        workspace.in1_time.data(), 0,
        static_cast<std::size_t>(workspace.fft_elements) * sizeof(ComplexFloat)));
    CUDA_CHECK(cudaMemset(
        workspace.in2_time.data(), 0,
        static_cast<std::size_t>(workspace.fft_elements) * sizeof(ComplexFloat)));
    cuda_utils::launch_1d_kernel(
        convolution_detail::pack_nd_real_kernel<T>,
        static_cast<std::size_t>(workspace.input1_elements),
        in1.data(), workspace.input1_elements,
        workspace.d_shape1.data(), workspace.d_strides1.data(),
        workspace.d_fft_strides.data(), workspace.rank, false,
        workspace.in1_time.data());
    cuda_utils::launch_1d_kernel(
        convolution_detail::pack_nd_real_kernel<T>,
        static_cast<std::size_t>(workspace.input2_elements),
        in2.data(), workspace.input2_elements,
        workspace.d_shape2.data(), workspace.d_strides2.data(),
        workspace.d_fft_strides.data(), workspace.rank, true,
        workspace.in2_time.data());

    transform_nd(
        workspace.in1_time, workspace.transform_scratch,
        workspace.in1_freq, workspace, false);
    transform_nd(
        workspace.in2_time, workspace.transform_scratch,
        workspace.in2_freq, workspace, false);
    cuda_utils::launch_1d_kernel(
        convolution_detail::multiply_kernel<>,
        static_cast<std::size_t>(workspace.fft_elements),
        workspace.in1_time.data(), workspace.in2_time.data(),
        workspace.product_freq.data(), workspace.fft_elements);
    transform_nd(
        workspace.product_freq, workspace.transform_scratch,
        workspace.in1_freq, workspace, true);
    cuda_utils::launch_1d_kernel(
        convolution_detail::crop_nd_kernel<T>, expected.size(),
        workspace.product_freq.data(), out.data(), workspace.output_elements,
        workspace.d_output_strides.data(), workspace.d_fft_strides.data(),
        workspace.d_crop_start.data(), workspace.rank);
}

template <typename T>
void correlate_device(
    const DeviceArray<T>& in1,
    const DeviceArray<T>& in2,
    DeviceArray<T>& out,
    CorrelateDeviceWorkspace& workspace)
{
    validate_1d(in1, in2, out, workspace.mode);
    if (in1.size() != static_cast<std::size_t>(workspace.len1) ||
        in2.size() != static_cast<std::size_t>(workspace.len2) ||
        out.size() != workspace.output_size()) {
        throw std::invalid_argument("typed correlate workspace mismatch");
    }
    if (in1.empty() || in2.empty()) return;
    if (!workspace.fft) throw std::invalid_argument("typed correlate FFT plan missing");

    cuda_utils::launch_1d_kernel(
        convolution_detail::pack_1d_kernel<T>, workspace.fft_size(), in1.data(), workspace.len1,
        workspace.in1_time.data(), workspace.fft_len, false);
    cuda_utils::launch_1d_kernel(
        convolution_detail::pack_1d_kernel<T>, workspace.fft_size(), in2.data(), workspace.len2,
        workspace.in2_time.data(), workspace.fft_len, true);
    workspace.fft->fft_device(workspace.in1_time, workspace.in1_freq);
    workspace.fft->fft_device(workspace.in2_time, workspace.in2_freq);
    cuda_utils::launch_1d_kernel(
        convolution_detail::multiply_kernel<>, workspace.fft_size(), workspace.in1_freq.data(),
        workspace.in2_freq.data(), workspace.product_freq.data(), static_cast<int>(workspace.fft_size()));
    workspace.fft->ifft_device(workspace.product_freq, workspace.ifft_time);
    cuda_utils::launch_1d_kernel(
        convolution_detail::crop_1d_kernel<T>, workspace.output_size(), workspace.ifft_time.data(), out.data(),
        workspace.out_len, static_cast<int>(typed_output_start(in1.size(), in2.size(), out.size())));
}

template <typename T>
void correlate2d_device(
    const DeviceArray<T>& in1,
    const DeviceArray<T>& in2,
    DeviceArray<T>& out,
    Correlate2DWorkspace& workspace)
{
    static_assert(detail::is_convolution_input_v<T>, "unsupported correlate2d business input dtype");
    Correlate2DOptions options;
    options.rows1 = workspace.rows1;
    options.cols1 = workspace.cols1;
    options.rows2 = workspace.rows2;
    options.cols2 = workspace.cols2;
    options.mode = workspace.mode;
    options.boundary = workspace.boundary;
    options.fillvalue = workspace.fillvalue;
    const Correlate2DOutputShape expected = correlate2d_output_shape(options);
    Correlate2DWorkspace validated(options);
    if (workspace.out_rows != validated.out_rows || workspace.out_cols != validated.out_cols ||
        workspace.row_start != validated.row_start || workspace.col_start != validated.col_start ||
        workspace.boundary_code != validated.boundary_code ||
        in1.size() != static_cast<std::size_t>(workspace.rows1) * workspace.cols1 ||
        in2.size() != static_cast<std::size_t>(workspace.rows2) * workspace.cols2 ||
        out.size() != expected.size()) {
        throw std::invalid_argument("typed correlate2d workspace mismatch");
    }
    if constexpr (std::is_integral_v<T>) {
        if (!validated.integral_fill_valid) {
            throw std::invalid_argument("typed correlate2d integral fillvalue must be finite");
        }
    }
    if (!out.empty() && (out.data() == in1.data() || out.data() == in2.data())) {
        throw std::invalid_argument("typed correlate2d does not permit input/output aliasing");
    }
    if (out.empty()) return;
    cuda_utils::launch_1d_kernel(
        convolution_detail::correlate2d_boundary_direct_kernel<T>,
        expected.size(),
        in1.data(), validated.rows1, validated.cols1,
        in2.data(), validated.rows2, validated.cols2,
        out.data(), validated.out_rows, validated.out_cols,
        validated.row_start, validated.col_start, validated.boundary_code,
        correlate2d_workspace_fill<T>(validated));
}

#define INSTANTIATE_CONV_GPU(T) \
    template void correlate_device(const DeviceArray<T>&, const DeviceArray<T>&, DeviceArray<T>&, const std::string&); \
    template void correlate_device(const DeviceArray<T>&, const DeviceArray<T>&, DeviceArray<T>&, CorrelateDeviceWorkspace&); \
    template void correlate_device(const DeviceArray<T>&, const DeviceArray<T>&, DeviceArray<T>&, CorrelateNDWorkspace&); \
    template void correlate2d_device(const DeviceArray<T>&, const DeviceArray<T>&, DeviceArray<T>&, Correlate2DWorkspace&)

INSTANTIATE_CONV_GPU(float);
INSTANTIATE_CONV_GPU(__half);
INSTANTIATE_CONV_GPU(std::int32_t);
INSTANTIATE_CONV_GPU(std::int16_t);
INSTANTIATE_CONV_GPU(std::int8_t);

#undef INSTANTIATE_CONV_GPU

}  // namespace cusignal
