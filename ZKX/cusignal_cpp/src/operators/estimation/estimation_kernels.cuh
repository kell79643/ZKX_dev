#pragma once

#include "simple_signal_typed.h"

#include <cmath>
#include <cstddef>

namespace cusignal::estimation_detail {

template <typename T>
__global__ void convert_to_float_kernel(const T* input, float* output, std::size_t count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = detail::SimpleSignalTypePolicy<T>::load(input[index]);
}

__global__ void kalman_predict_x_kernel(
    const float* x, const float* f, float* next_x,
    std::size_t count, int dim_x)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int row = static_cast<int>(index % static_cast<std::size_t>(dim_x));
    const std::size_t point = index / static_cast<std::size_t>(dim_x);
    const std::size_t x_base = point * dim_x;
    const std::size_t matrix_base = x_base * dim_x;
    float value = 0.0F;
    for (int inner = 0; inner < dim_x; ++inner) {
        value = fmaf(f[matrix_base + row * dim_x + inner], x[x_base + inner], value);
    }
    next_x[index] = value;
}

__global__ void kalman_predict_fp_kernel(
    const float* f, const float* p, float* fp,
    std::size_t count, int dim_x)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const std::size_t matrix_size = static_cast<std::size_t>(dim_x) * dim_x;
    const std::size_t matrix_base = (index / matrix_size) * matrix_size;
    const int entry = static_cast<int>(index % matrix_size);
    const int row = entry / dim_x;
    const int col = entry % dim_x;
    float value = 0.0F;
    for (int inner = 0; inner < dim_x; ++inner) {
        value = fmaf(f[matrix_base + row * dim_x + inner],
                     p[matrix_base + inner * dim_x + col], value);
    }
    fp[index] = value;
}

__global__ void kalman_predict_p_kernel(
    const float* fp, const float* f, const float* q, const float* alpha_sq,
    float* next_p, std::size_t count, int dim_x,
    bool scalar_q, float q_scalar)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const std::size_t matrix_size = static_cast<std::size_t>(dim_x) * dim_x;
    const std::size_t point = index / matrix_size;
    const std::size_t matrix_base = point * matrix_size;
    const int entry = static_cast<int>(index % matrix_size);
    const int row = entry / dim_x;
    const int col = entry % dim_x;
    float value = 0.0F;
    for (int inner = 0; inner < dim_x; ++inner) {
        value = fmaf(fp[matrix_base + row * dim_x + inner],
                     f[matrix_base + col * dim_x + inner], value);
    }
    const float noise = scalar_q ? (row == col ? q_scalar : 0.0F) : q[index];
    next_p[index] = alpha_sq[point] * value + noise;
}

__global__ void kalman_innovation_kernel(
    const float* x, const float* h, const float* z, float* innovation,
    std::size_t count, int dim_x, int dim_z)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int row = static_cast<int>(index % static_cast<std::size_t>(dim_z));
    const std::size_t point = index / static_cast<std::size_t>(dim_z);
    const std::size_t x_base = point * dim_x;
    const std::size_t h_base = point * static_cast<std::size_t>(dim_z) * dim_x;
    float value = z[index];
    for (int inner = 0; inner < dim_x; ++inner) {
        value = fmaf(-h[h_base + row * dim_x + inner], x[x_base + inner], value);
    }
    innovation[index] = value;
}

__global__ void kalman_pht_kernel(
    const float* p, const float* h, float* pht,
    std::size_t count, int dim_x, int dim_z)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const std::size_t point_matrix = static_cast<std::size_t>(dim_x) * dim_z;
    const std::size_t point = index / point_matrix;
    const int entry = static_cast<int>(index % point_matrix);
    const int row = entry / dim_z;
    const int col = entry % dim_z;
    const std::size_t p_base = point * static_cast<std::size_t>(dim_x) * dim_x;
    const std::size_t h_base = point * static_cast<std::size_t>(dim_z) * dim_x;
    float value = 0.0F;
    for (int inner = 0; inner < dim_x; ++inner) {
        value = fmaf(p[p_base + row * dim_x + inner],
                     h[h_base + col * dim_x + inner], value);
    }
    pht[index] = value;
}

__global__ void kalman_s_kernel(
    const float* h, const float* pht, const float* r, float* s,
    std::size_t count, int dim_x, int dim_z,
    bool scalar_r, float r_scalar)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const std::size_t matrix_size = static_cast<std::size_t>(dim_z) * dim_z;
    const std::size_t point = index / matrix_size;
    const int entry = static_cast<int>(index % matrix_size);
    const int row = entry / dim_z;
    const int col = entry % dim_z;
    const std::size_t h_base = point * static_cast<std::size_t>(dim_z) * dim_x;
    const std::size_t pht_base = point * static_cast<std::size_t>(dim_x) * dim_z;
    float value = scalar_r ? (row == col ? r_scalar : 0.0F) : r[index];
    for (int inner = 0; inner < dim_x; ++inner) {
        value = fmaf(h[h_base + row * dim_x + inner],
                     pht[pht_base + inner * dim_z + col], value);
    }
    s[index] = value;
}

__global__ void kalman_augmented_init_kernel(
    const float* s, float* augmented, std::size_t count, int dim_z)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const std::size_t stride = static_cast<std::size_t>(2) * dim_z;
    const std::size_t point_size = static_cast<std::size_t>(dim_z) * stride;
    const std::size_t point = index / point_size;
    const int entry = static_cast<int>(index % point_size);
    const int row = entry / static_cast<int>(stride);
    const int col = entry % static_cast<int>(stride);
    if (col < dim_z) {
        augmented[index] = s[point * static_cast<std::size_t>(dim_z) * dim_z +
                             row * dim_z + col];
    } else {
        augmented[index] = row == col - dim_z ? 1.0F : 0.0F;
    }
}

__global__ void kalman_inverse_kernel(float* augmented, int points, int dim_z)
{
    const int point = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (point >= points) return;
    const int stride = 2 * dim_z;
    const std::size_t base = static_cast<std::size_t>(point) * dim_z * stride;

    for (int row = dim_z - 1; row > 0; --row) {
        if (augmented[base + (row - 1) * stride] <
            augmented[base + row * stride]) {
            for (int col = 0; col < stride; ++col) {
                const float value = augmented[base + (row - 1) * stride + col];
                augmented[base + (row - 1) * stride + col] =
                    augmented[base + row * stride + col];
                augmented[base + row * stride + col] = value;
            }
        }
    }
    for (int pivot = 0; pivot < dim_z; ++pivot) {
        for (int row = 0; row < dim_z; ++row) {
            if (row == pivot) continue;
            const float factor = augmented[base + row * stride + pivot] /
                                 augmented[base + pivot * stride + pivot];
            for (int col = 0; col < stride; ++col) {
                augmented[base + row * stride + col] -=
                    augmented[base + pivot * stride + col] * factor;
            }
        }
    }
    for (int row = 0; row < dim_z; ++row) {
        const float diagonal = augmented[base + row * stride + row];
        for (int col = 0; col < stride; ++col) {
            augmented[base + row * stride + col] /= diagonal;
        }
    }
}

__global__ void kalman_gain_kernel(
    const float* pht, const float* augmented, float* gain,
    std::size_t count, int dim_x, int dim_z)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const std::size_t point_matrix = static_cast<std::size_t>(dim_x) * dim_z;
    const std::size_t point = index / point_matrix;
    const int entry = static_cast<int>(index % point_matrix);
    const int row = entry / dim_z;
    const int col = entry % dim_z;
    const int stride = 2 * dim_z;
    const std::size_t aug_base = point * static_cast<std::size_t>(dim_z) * stride;
    const std::size_t pht_base = point * point_matrix;
    float value = 0.0F;
    for (int inner = 0; inner < dim_z; ++inner) {
        value = fmaf(pht[pht_base + row * dim_z + inner],
                     augmented[aug_base + col * stride + dim_z + inner], value);
    }
    gain[index] = value;
}

__global__ void kalman_update_x_kernel(
    const float* x, const float* gain, const float* innovation, float* next_x,
    std::size_t count, int dim_x, int dim_z)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int row = static_cast<int>(index % static_cast<std::size_t>(dim_x));
    const std::size_t point = index / static_cast<std::size_t>(dim_x);
    const std::size_t gain_base = point * static_cast<std::size_t>(dim_x) * dim_z;
    const std::size_t z_base = point * dim_z;
    float value = x[index];
    for (int inner = 0; inner < dim_z; ++inner) {
        value = fmaf(gain[gain_base + row * dim_z + inner],
                     innovation[z_base + inner], value);
    }
    next_x[index] = value;
}

__global__ void kalman_i_kh_kernel(
    const float* gain, const float* h, float* i_kh,
    std::size_t count, int dim_x, int dim_z)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const std::size_t matrix_size = static_cast<std::size_t>(dim_x) * dim_x;
    const std::size_t point = index / matrix_size;
    const int entry = static_cast<int>(index % matrix_size);
    const int row = entry / dim_x;
    const int col = entry % dim_x;
    const std::size_t gain_base = point * static_cast<std::size_t>(dim_x) * dim_z;
    const std::size_t h_base = point * static_cast<std::size_t>(dim_z) * dim_x;
    float value = row == col ? 1.0F : 0.0F;
    for (int inner = 0; inner < dim_z; ++inner) {
        value = fmaf(-gain[gain_base + row * dim_z + inner],
                     h[h_base + inner * dim_x + col], value);
    }
    i_kh[index] = value;
}

__global__ void kalman_left_covariance_kernel(
    const float* i_kh, const float* p, float* left,
    std::size_t count, int dim_x)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const std::size_t matrix_size = static_cast<std::size_t>(dim_x) * dim_x;
    const std::size_t matrix_base = (index / matrix_size) * matrix_size;
    const int entry = static_cast<int>(index % matrix_size);
    const int row = entry / dim_x;
    const int col = entry % dim_x;
    float value = 0.0F;
    for (int inner = 0; inner < dim_x; ++inner) {
        value = fmaf(i_kh[matrix_base + row * dim_x + inner],
                     p[matrix_base + inner * dim_x + col], value);
    }
    left[index] = value;
}

__global__ void kalman_joseph_p_kernel(
    const float* left, const float* i_kh, const float* gain, const float* r,
    float* next_p, std::size_t count, int dim_x, int dim_z,
    bool scalar_r, float r_scalar)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const std::size_t matrix_size = static_cast<std::size_t>(dim_x) * dim_x;
    const std::size_t point = index / matrix_size;
    const std::size_t matrix_base = point * matrix_size;
    const int entry = static_cast<int>(index % matrix_size);
    const int row = entry / dim_x;
    const int col = entry % dim_x;
    const std::size_t gain_base = point * static_cast<std::size_t>(dim_x) * dim_z;
    const std::size_t r_base = point * static_cast<std::size_t>(dim_z) * dim_z;
    float value = 0.0F;
    for (int inner = 0; inner < dim_x; ++inner) {
        value = fmaf(left[matrix_base + row * dim_x + inner],
                     i_kh[matrix_base + col * dim_x + inner], value);
    }
    for (int left_z = 0; left_z < dim_z; ++left_z) {
        for (int right_z = 0; right_z < dim_z; ++right_z) {
            const float noise = scalar_r
                ? (left_z == right_z ? r_scalar : 0.0F)
                : r[r_base + left_z * dim_z + right_z];
            const float gain_noise =
                gain[gain_base + row * dim_z + left_z] * noise;
            value = fmaf(
                gain_noise,
                gain[gain_base + col * dim_z + right_z], value);
        }
    }
    next_p[index] = value;
}

// 单状态、单观测维度的时间序列必须按观测顺序递推。用一个设备线程完成整段
// predict/update，避免把时间维错误地展开成相互独立的 points，也避免每个观测
// 启动一组只有一个元素的 kernel。
__global__ void kalman_scalar_sequence_kernel(
    float* x, float* p, const float* f, const float* q,
    const float* alpha_sq, const float* h, const float* r,
    const float* observations, std::size_t observation_count)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float state = x[0];
    float covariance = p[0];
    const float transition = f[0];
    const float process_noise = q[0];
    const float alpha = alpha_sq[0];
    const float measurement = h[0];
    const float measurement_noise = r[0];
    for (std::size_t index = 0; index < observation_count; ++index) {
        state = transition * state;
        const float projected_covariance =
            transition * covariance * transition;
        covariance = fmaf(alpha, projected_covariance, process_noise);
        const float innovation =
            fmaf(-measurement, state, observations[index]);
        const float pht = covariance * measurement;
        const float innovation_covariance =
            fmaf(measurement, pht, measurement_noise);
        const float gain = pht / innovation_covariance;
        state = fmaf(gain, innovation, state);
        const float i_kh = fmaf(-gain, measurement, 1.0F);
        const float gain_noise_gain =
            gain * measurement_noise * gain;
        covariance = fmaf(
            i_kh * covariance, i_kh, gain_noise_gain);
    }
    x[0] = state;
    p[0] = covariance;
}

}  // namespace cusignal::estimation_detail
