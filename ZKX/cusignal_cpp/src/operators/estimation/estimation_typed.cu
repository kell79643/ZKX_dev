#include <cusignal/operators/estimation/estimation_typed.h>
#include "estimation_kernels.cuh"

#include <cusignal/runtime/cuda_utils.h>

#include <stdexcept>
#include <vector>

namespace cusignal {
namespace {

bool same_layout(const KalmanLayout& left, const KalmanLayout& right)
{
    return left.dim_x == right.dim_x && left.dim_z == right.dim_z &&
           left.points == right.points && left.x_count == right.x_count &&
           left.p_count == right.p_count && left.h_count == right.h_count &&
           left.r_count == right.r_count && left.z_count == right.z_count;
}

KalmanLayout validated_layout(const KalmanLayout& layout)
{
    KalmanLayout validated(layout.dim_x, layout.dim_z, layout.points);
    if (!same_layout(layout, validated)) {
        throw std::invalid_argument("Kalman layout was mutated");
    }
    return validated;
}

void validate_state(const KalmanDeviceState& state)
{
    const KalmanLayout layout = validated_layout(state.layout);
    if (state.x.size() != layout.x_count || state.p.size() != layout.p_count) {
        throw std::invalid_argument("Kalman device state shape mismatch");
    }
}

void validate_workspace(const KalmanDeviceWorkspace& workspace, const KalmanLayout& expected)
{
    const KalmanLayout layout = validated_layout(workspace.layout);
    if (!same_layout(layout, expected) || workspace.f.size() != layout.p_count ||
        workspace.q.size() != layout.p_count ||
        workspace.alpha_sq.size() != static_cast<std::size_t>(layout.points) ||
        workspace.h.size() != layout.h_count || workspace.r.size() != layout.r_count ||
        workspace.z.size() != layout.z_count || workspace.next_x.size() != layout.x_count ||
        workspace.next_p.size() != layout.p_count || workspace.fp.size() != layout.p_count ||
        workspace.pht.size() != layout.h_count ||
        workspace.innovation.size() != layout.z_count || workspace.s.size() != layout.r_count ||
        workspace.augmented.size() % 2 != 0 ||
        workspace.augmented.size() / 2 != layout.r_count ||
        workspace.gain.size() != layout.h_count || workspace.i_kh.size() != layout.p_count ||
        workspace.left_covariance.size() != layout.p_count) {
        throw std::invalid_argument("Kalman device workspace shape mismatch");
    }
}

template <typename T>
void convert_to_float(const DeviceArray<T>& input, DeviceArray<float>& output)
{
    if (input.size() != output.size()) {
        throw std::invalid_argument("Kalman conversion shape mismatch");
    }
    if (!input.empty()) {
        cuda_utils::launch_1d_kernel(
            estimation_detail::convert_to_float_kernel<T>, input.size(),
            input.data(), output.data(), input.size());
    }
}

}  // namespace

template <typename T>
void initialize_kalman_device_state(
    const DeviceArray<T>& x,
    const DeviceArray<T>& p,
    KalmanDeviceState& state)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported Kalman business dtype");
    validate_state(state);
    if (x.size() != state.layout.x_count || p.size() != state.layout.p_count) {
        throw std::invalid_argument("initialize_kalman_device_state shape mismatch");
    }
    convert_to_float(x, state.x);
    convert_to_float(p, state.p);
}

template <typename T>
void kalman_predict_device(
    KalmanDeviceState& state,
    const DeviceArray<T>& f,
    const DeviceArray<T>& q,
    const DeviceArray<T>& alpha_sq,
    KalmanDeviceWorkspace& workspace,
    const KalmanPredictOptions& options)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported Kalman business dtype");
    if (options.control_provided) {
        throw std::logic_error("Kalman control input is not implemented by fixed cuSignal");
    }
    validate_state(state);
    validate_workspace(workspace, state.layout);
    const KalmanLayout& layout = state.layout;
    if (f.size() != layout.p_count ||
        (!options.process_noise_scalar && q.size() != layout.p_count) ||
        alpha_sq.size() != static_cast<std::size_t>(layout.points)) {
        throw std::invalid_argument("kalman_predict_device shape mismatch");
    }
    if (layout.points == 0) return;

    convert_to_float(f, workspace.f);
    if (!options.process_noise_scalar) convert_to_float(q, workspace.q);
    convert_to_float(alpha_sq, workspace.alpha_sq);
    cuda_utils::launch_1d_kernel(
        estimation_detail::kalman_predict_x_kernel, layout.x_count,
        state.x.data(), workspace.f.data(), workspace.next_x.data(),
        layout.x_count, layout.dim_x);
    cuda_utils::launch_1d_kernel(
        estimation_detail::kalman_predict_fp_kernel, layout.p_count,
        workspace.f.data(), state.p.data(), workspace.fp.data(),
        layout.p_count, layout.dim_x);
    cuda_utils::launch_1d_kernel(
        estimation_detail::kalman_predict_p_kernel, layout.p_count,
        workspace.fp.data(), workspace.f.data(), workspace.q.data(),
        workspace.alpha_sq.data(), workspace.next_p.data(), layout.p_count,
        layout.dim_x, options.process_noise_scalar, options.q_scalar);
    cuda_utils::copy_device_to_device(
        state.x.data(), workspace.next_x.data(), layout.x_count);
    cuda_utils::copy_device_to_device(
        state.p.data(), workspace.next_p.data(), layout.p_count);
}

template <typename T>
void kalman_update_device(
    KalmanDeviceState& state,
    const DeviceArray<T>& h,
    const DeviceArray<T>& r,
    const DeviceArray<T>& z,
    KalmanDeviceWorkspace& workspace,
    const KalmanUpdateOptions& options)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported Kalman business dtype");
    validate_state(state);
    validate_workspace(workspace, state.layout);
    if (!options.measurement_present) return;
    const KalmanLayout& layout = state.layout;
    if (h.size() != layout.h_count || z.size() != layout.z_count ||
        (!options.measurement_noise_scalar && r.size() != layout.r_count)) {
        throw std::invalid_argument("kalman_update_device shape mismatch");
    }
    if (layout.points == 0) return;

    convert_to_float(h, workspace.h);
    if (!options.measurement_noise_scalar) convert_to_float(r, workspace.r);
    convert_to_float(z, workspace.z);
    cuda_utils::launch_1d_kernel(
        estimation_detail::kalman_innovation_kernel, layout.z_count,
        state.x.data(), workspace.h.data(), workspace.z.data(),
        workspace.innovation.data(), layout.z_count, layout.dim_x, layout.dim_z);
    cuda_utils::launch_1d_kernel(
        estimation_detail::kalman_pht_kernel, layout.h_count,
        state.p.data(), workspace.h.data(), workspace.pht.data(),
        layout.h_count, layout.dim_x, layout.dim_z);
    cuda_utils::launch_1d_kernel(
        estimation_detail::kalman_s_kernel, layout.r_count,
        workspace.h.data(), workspace.pht.data(), workspace.r.data(),
        workspace.s.data(), layout.r_count, layout.dim_x, layout.dim_z,
        options.measurement_noise_scalar, options.r_scalar);
    cuda_utils::launch_1d_kernel(
        estimation_detail::kalman_augmented_init_kernel,
        workspace.augmented.size(), workspace.s.data(), workspace.augmented.data(),
        workspace.augmented.size(), layout.dim_z);
    cuda_utils::launch_1d_kernel(
        estimation_detail::kalman_inverse_kernel,
        static_cast<std::size_t>(layout.points), workspace.augmented.data(),
        layout.points, layout.dim_z);
    cuda_utils::launch_1d_kernel(
        estimation_detail::kalman_gain_kernel, layout.h_count,
        workspace.pht.data(), workspace.augmented.data(), workspace.gain.data(),
        layout.h_count, layout.dim_x, layout.dim_z);
    cuda_utils::launch_1d_kernel(
        estimation_detail::kalman_update_x_kernel, layout.x_count,
        state.x.data(), workspace.gain.data(), workspace.innovation.data(),
        workspace.next_x.data(), layout.x_count, layout.dim_x, layout.dim_z);
    cuda_utils::launch_1d_kernel(
        estimation_detail::kalman_i_kh_kernel, layout.p_count,
        workspace.gain.data(), workspace.h.data(), workspace.i_kh.data(),
        layout.p_count, layout.dim_x, layout.dim_z);
    cuda_utils::launch_1d_kernel(
        estimation_detail::kalman_left_covariance_kernel, layout.p_count,
        workspace.i_kh.data(), state.p.data(), workspace.left_covariance.data(),
        layout.p_count, layout.dim_x);
    cuda_utils::launch_1d_kernel(
        estimation_detail::kalman_joseph_p_kernel, layout.p_count,
        workspace.left_covariance.data(), workspace.i_kh.data(),
        workspace.gain.data(), workspace.r.data(), workspace.next_p.data(),
        layout.p_count, layout.dim_x, layout.dim_z,
        options.measurement_noise_scalar, options.r_scalar);
    cuda_utils::copy_device_to_device(
        state.x.data(), workspace.next_x.data(), layout.x_count);
    cuda_utils::copy_device_to_device(
        state.p.data(), workspace.next_p.data(), layout.p_count);
}

void kalman_predict_update_scalar_sequence_device(
    KalmanDeviceState& state,
    const DeviceArray<float>& f,
    const DeviceArray<float>& q,
    const DeviceArray<float>& alpha_sq,
    const DeviceArray<float>& h,
    const DeviceArray<float>& r,
    const DeviceArray<float>& observations)
{
    validate_state(state);
    const KalmanLayout& layout = state.layout;
    if (layout.dim_x != 1 || layout.dim_z != 1 || layout.points != 1 ||
        f.size() != 1 || q.size() != 1 || alpha_sq.size() != 1 ||
        h.size() != 1 || r.size() != 1) {
        throw std::invalid_argument(
            "kalman scalar sequence requires layout [points=1,dim_x=1,dim_z=1]");
    }
    if (observations.empty()) return;
    estimation_detail::kalman_scalar_sequence_kernel<<<1, 1>>>(
        state.x.data(), state.p.data(), f.data(), q.data(), alpha_sq.data(),
        h.data(), r.data(), observations.data(), observations.size());
    CUDA_CHECK(cudaGetLastError());
}

void kalman_predict_device(
    DeviceArray<float>& x,
    DeviceArray<float>& p,
    const DeviceArray<float>& q,
    const DeviceArray<float>& f,
    int dim_x,
    int points)
{
    KalmanLayout layout(dim_x, 1, points);
    KalmanDeviceState state(layout);
    KalmanDeviceWorkspace workspace(layout);
    initialize_kalman_device_state(x, p, state);
    std::vector<float> host_alpha(static_cast<std::size_t>(points), 1.0F);
    DeviceArray<float> alpha_sq = DeviceArray<float>::from_host(host_alpha);
    kalman_predict_device(state, f, q, alpha_sq, workspace);
    cuda_utils::copy_device_to_device(x.data(), state.x.data(), layout.x_count);
    cuda_utils::copy_device_to_device(p.data(), state.p.data(), layout.p_count);
    cuda_utils::synchronize_stream();
}

void kalman_update_device(
    DeviceArray<float>& x,
    DeviceArray<float>& p,
    const DeviceArray<float>& h,
    const DeviceArray<float>& r,
    const DeviceArray<float>& z,
    int dim_x,
    int dim_z,
    int points)
{
    KalmanLayout layout(dim_x, dim_z, points);
    KalmanDeviceState state(layout);
    KalmanDeviceWorkspace workspace(layout);
    initialize_kalman_device_state(x, p, state);
    kalman_update_device(state, h, r, z, workspace);
    cuda_utils::copy_device_to_device(x.data(), state.x.data(), layout.x_count);
    cuda_utils::copy_device_to_device(p.data(), state.p.data(), layout.p_count);
    cuda_utils::synchronize_stream();
}

#define INSTANTIATE_KALMAN_GPU(T) \
    template void initialize_kalman_device_state(const DeviceArray<T>&, const DeviceArray<T>&, KalmanDeviceState&); \
    template void kalman_predict_device(KalmanDeviceState&, const DeviceArray<T>&, const DeviceArray<T>&, const DeviceArray<T>&, KalmanDeviceWorkspace&, const KalmanPredictOptions&); \
    template void kalman_update_device(KalmanDeviceState&, const DeviceArray<T>&, const DeviceArray<T>&, const DeviceArray<T>&, KalmanDeviceWorkspace&, const KalmanUpdateOptions&)

INSTANTIATE_KALMAN_GPU(float);
INSTANTIATE_KALMAN_GPU(__half);
INSTANTIATE_KALMAN_GPU(std::int32_t);
INSTANTIATE_KALMAN_GPU(std::int16_t);
INSTANTIATE_KALMAN_GPU(std::int8_t);

#undef INSTANTIATE_KALMAN_GPU

}  // namespace cusignal
