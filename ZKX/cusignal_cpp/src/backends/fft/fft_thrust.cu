/**
 * @file fft_thrust.cu
 * @brief Thrust 辅助的项目 FFT 算法（不依赖 cuFFT/dlfft）
 *
 * 本文件是 fft_interface.h 中定义的项目自研 ZQ500 GPU FFT 实现。正式构建通过
 * -DUSE_DLFFT=OFF -DUSE_THRUST=ON 选择本实现；test_all 也保留独立目标用于
 * 平台可行性、正确性和内存稳定性对照。文件名中的 thrust 主要表示使用 Thrust
 * 的复数类型；FFT 计划、位反转、蝶形、Bluestein、2D 行列分解等执行流程均由
 * 本文件的 CUDA kernel 和 C++ 调度实现。
 *
 * 算法选择策略：
 *   - N 为 2 的幂：Cooley-Tukey Radix-4/末级 Radix-2 FFT
 *     （out-of-place 首级融合位反转，in-place 使用安全的独立位反转）
 *   - N 为非 2 的幂：Bluestein's Chirp-Z Transform（自动切换）
 *   - 支持任意正整数长度 N >= 1
 *
 * 支持的变换类型：
 *   - C2C: 复数到复数（支持任意长度，含 Batch 和 In-place）
 *   - R2C: 实数到复数（输出 N/2+1 个点，利用共轭对称性）
 *   - C2R: 复数到实数（输入 N/2+1 个点，自动补全负频率）
 *   - 2D FFT: 行列分解法（先逐行 FFT，再逐列 FFT）
 *
 * 性能说明：
 *   - 正式二次幂路径使用 plan 级旋转因子表和 Radix-4 蝶形；实验性的
 *     shared-memory kernel 当前不在正式选择器中
 *   - Cooley-Tukey 路径为 O(N log N)，Radix-4 减少逐级 kernel 和全局往返
 *   - Bluestein 路径：O(M log M) 复杂度，M = next_power_of_2(2N)
 *   - 尚未使用 Split-radix、mixed-radix 或 tiled 2D transpose
 *   - 主工作缓冲区和 Bluestein scratch 均在 plan_create 时预分配并复用
 *
 * 归一化约定：
 *   - 正向变换（FFT_FORWARD）：不归一化
 *   - 逆向变换（FFT_INVERSE）：结果除以 N（1/N 缩放）
 *   - 与 NumPy numpy.fft.ifft 行为一致
 *
 * @note 此实现不链接 cuFFT/dlfft，依赖 ZQ500 CUDA Runtime-compatible API、curt
 *       和 SDK 提供的 Thrust 头文件。
 *
 * @author cusignal_cpp
 * @date 2024-2026
 */

#include <cusignal/backends/fft/fft_interface.h>
#include <cusignal/runtime/device_allocation_tracker.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <cstddef>
#include <cmath>
#include <cstring>

using thrust::complex;

using cusignal::cuda_utils::tracked_cuda_free;
using cusignal::cuda_utils::tracked_cuda_malloc;

struct FFTPlan {
    int nx;                          /**< 1D FFT 长度 / 2D FFT 列数（nx × ny） */
    int ny;                          /**< 2D FFT 行数，ny=0 表示 1D 模式 */
    int batch;                       /**< 批处理数量 */
    FFTType type;                    /**< 变换类型：C2C / R2C / C2R */
    FFTDirection direction;          /**< 变换方向：FORWARD / INVERSE */
    complex<float>* d_buffer;        /**< R2C/C2R 专用缓冲区（半谱存储） */

    complex<float>* d_work_main;     /**< 主工作缓冲区（plan_create 时预分配） */
    complex<float>* d_work_aux;      /**< 辅助工作缓冲区（R2C/C2R 全谱空间） */
    complex<float>* d_twiddles;      /**< d_work_main 内的 1D 二次幂旋转因子表 */
    complex<float>* d_bluestein_a;   /**< d_work_main 内的非二次幂卷积工作区 A */
    complex<float>* d_bluestein_b;   /**< d_work_main 内的非二次幂卷积工作区 B */
    int bluestein_size;              /**< 两块 Bluestein 工作区的复数元素数 */
};

namespace {

/**
 * @brief 位反转置换 Kernel（Cooley-Tukey FFT 前置步骤）
 *
 * 将输入数据按位反转顺序重新排列，是 Radix-2 FFT 的必要预处理。
 * 使用 CUDA 内建函数 __brev 实现 32 位整数位反转，效率极高。
 * 采用 swap 避免写冲突：仅当 idx < rev 时执行交换，保证每个元素只被处理一次。
 *
 * @param[out] data  待重排的复数数组（原地修改）
 * @param[in]  n     数组长度（必须为 2 的幂）
 * @param[in]  bits  log2(n)，位反转的有效位数
 */
__global__ void bit_reverse_kernel(
    complex<float>* data,
    int n,
    int bits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int rev = __brev(idx) >> (32 - bits);
    if (idx < rev) {
        complex<float> tmp = data[idx];
        data[idx] = data[rev];
        data[rev] = tmp;
    }
}

/**
 * @brief 蝶形运算 Kernel（Cooley-Tukey FFT 核心步骤）
 *
 * 执行单级蝶形运算，是 FFT 的核心计算单元。每级处理 n/2 个蝶形单元，
 * 跨度为 half = 2^stage。旋转因子（Twiddle Factor）在运行时通过
 * 三角函数实时计算，避免预存查找表的内存开销。
 *
 * 计算公式：
 *   output[offset]       = a + W * b
 *   output[offset+half]  = a - W * b
 * 其中 W = exp(-j * π * k / half)，direction 控制正/逆变换符号
 *
 * @param[in,out] data      复数数据（原地更新）
 * @param[in]     n         数据长度（必须为 2 的幂）
 * @param[in]     stage     当前蝶形级数（0-based，共 log2(n) 级）
 * @param[in]     direction 变换方向：1=正向(FFT)，-1=逆向(IFFT)
 */
__global__ void butterfly_kernel(
    complex<float>* data,
    int n,
    int stage,
    int direction
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n / 2) return;

    int half = 1 << stage;
    int group_size = half << 1;

    int group_idx = (idx / half) * group_size;
    int butterfly_idx = idx % half;

    float angle = direction * (-M_PI) * butterfly_idx / half;
    complex<float> twiddle(cosf(angle), sinf(angle));

    int offset = group_idx + butterfly_idx;
    complex<float> a = data[offset];
    complex<float> b = data[offset + half];
    data[offset] = a + twiddle * b;
    data[offset + half] = a - twiddle * b;
}

__global__ void bit_reverse_batched_kernel(
    complex<float>* data,
    int n,
    int batch,
    int bits
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * batch;
    if (idx >= total) return;

    const int batch_idx = idx / n;
    const int local_idx = idx - batch_idx * n;
    const int rev = __brev(local_idx) >> (32 - bits);
    if (local_idx < rev) {
        complex<float>* base = data + batch_idx * n;
        complex<float> tmp = base[local_idx];
        base[local_idx] = base[rev];
        base[rev] = tmp;
    }
}

__global__ void butterfly_batched_kernel(
    complex<float>* data,
    int n,
    int batch,
    int stage,
    int direction
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int butterflies_per_batch = n / 2;
    const int total = butterflies_per_batch * batch;
    if (idx >= total) return;

    const int batch_idx = idx / butterflies_per_batch;
    const int local_idx = idx - batch_idx * butterflies_per_batch;
    const int half = 1 << stage;
    const int group_size = half << 1;
    const int group_idx = (local_idx / half) * group_size;
    const int butterfly_idx = local_idx % half;

    const float angle = direction * (-M_PI) * butterfly_idx / half;
    const complex<float> twiddle(cosf(angle), sinf(angle));

    complex<float>* base = data + batch_idx * n;
    const int offset = group_idx + butterfly_idx;
    const complex<float> a = base[offset];
    const complex<float> b = base[offset + half];
    base[offset] = a + twiddle * b;
    base[offset + half] = a - twiddle * b;
}

__global__ void initialize_twiddles_kernel(complex<float>* twiddles, int n)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < n / 2) {
        const float angle = -2.0f * static_cast<float>(M_PI) * index / n;
        twiddles[index] = complex<float>(cosf(angle), sinf(angle));
    }
}

__global__ void butterfly_batched_twiddle_kernel(
    complex<float>* data,
    const complex<float>* twiddles,
    int n,
    int batch,
    int stage,
    int direction)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int butterflies_per_batch = n / 2;
    const int total = butterflies_per_batch * batch;
    if (index >= total) return;

    const int batch_idx = index / butterflies_per_batch;
    const int local_idx = index - batch_idx * butterflies_per_batch;
    const int half = 1 << stage;
    const int group_size = half << 1;
    const int butterfly_idx = local_idx % half;
    const int offset = (local_idx / half) * group_size + butterfly_idx;
    complex<float> twiddle = twiddles[butterfly_idx * (n / group_size)];
    if (direction < 0) {
        twiddle = complex<float>(twiddle.real(), -twiddle.imag());
    }

    complex<float>* base = data + batch_idx * n;
    const complex<float> a = base[offset];
    const complex<float> b = base[offset + half];
    base[offset] = a + twiddle * b;
    base[offset + half] = a - twiddle * b;
}

/**
 * @brief 将连续两级 Radix-2 蝶形融合为一次四路全局读写。
 *
 * 输入仍采用现有二进制位反转布局，因此该 kernel 不改变 FFT 排列约定；它只把 stage
 * 和 stage+1 的中间结果保存在寄存器中，减少一次 kernel 启动及一次中间全局内存往返。
 */
__global__ void butterfly_radix4_batched_twiddle_kernel(
    complex<float>* data,
    const complex<float>* twiddles,
    int n,
    int batch,
    int stage,
    int direction)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int butterflies_per_batch = n / 4;
    const int total = butterflies_per_batch * batch;
    if (index >= total) return;

    const int batch_idx = index / butterflies_per_batch;
    const int local_idx = index - batch_idx * butterflies_per_batch;
    const int half = 1 << stage;
    const int group_size = half << 2;
    const int k = local_idx % half;
    const int base_offset = (local_idx / half) * group_size + k;

    complex<float> w1 = twiddles[k * (n / (half << 1))];
    complex<float> w2 = twiddles[k * (n / group_size)];
    complex<float> w3 = twiddles[(k + half) * (n / group_size)];
    if (direction < 0) {
        w1 = complex<float>(w1.real(), -w1.imag());
        w2 = complex<float>(w2.real(), -w2.imag());
        w3 = complex<float>(w3.real(), -w3.imag());
    }

    complex<float>* base = data + batch_idx * n;
    const complex<float> x0 = base[base_offset];
    const complex<float> x1 = base[base_offset + half];
    const complex<float> x2 = base[base_offset + 2 * half];
    const complex<float> x3 = base[base_offset + 3 * half];

    const complex<float> first_even = x0 + w1 * x1;
    const complex<float> first_odd = x0 - w1 * x1;
    const complex<float> second_even = x2 + w1 * x3;
    const complex<float> second_odd = x2 - w1 * x3;

    base[base_offset] = first_even + w2 * second_even;
    base[base_offset + 2 * half] = first_even - w2 * second_even;
    base[base_offset + half] = first_odd + w3 * second_odd;
    base[base_offset + 3 * half] = first_odd - w3 * second_odd;
}

/**
 * @brief out-of-place 首级：直接从原输入按位反转地址读取，并完成前两级蝶形。
 *
 * 该 kernel 把原来的 D2D copy、独立位反转 kernel 和 stage 0/1 合并。仅用于 input 与
 * output 不同的 C2C 路径；in-place 路径仍使用独立位反转，避免跨 block 读写覆盖。
 */
__global__ void bit_reverse_radix4_outofplace_kernel(
    complex<float>* output,
    const complex<float>* input,
    const complex<float>* twiddles,
    int n,
    int batch,
    int bits,
    int direction)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int quads_per_batch = n / 4;
    const int total = quads_per_batch * batch;
    if (index >= total) return;

    const int batch_idx = index / quads_per_batch;
    const int quad = index - batch_idx * quads_per_batch;
    const int offset = quad * 4;
    const complex<float>* batch_input = input + batch_idx * n;
    complex<float>* batch_output = output + batch_idx * n;

    const int r0 = __brev(offset) >> (32 - bits);
    const int r1 = __brev(offset + 1) >> (32 - bits);
    const int r2 = __brev(offset + 2) >> (32 - bits);
    const int r3 = __brev(offset + 3) >> (32 - bits);
    const complex<float> x0 = batch_input[r0];
    const complex<float> x1 = batch_input[r1];
    const complex<float> x2 = batch_input[r2];
    const complex<float> x3 = batch_input[r3];

    complex<float> w3 = twiddles[n / 4];
    if (direction < 0) {
        w3 = complex<float>(w3.real(), -w3.imag());
    }
    const complex<float> first_even = x0 + x1;
    const complex<float> first_odd = x0 - x1;
    const complex<float> second_even = x2 + x3;
    const complex<float> second_odd = x2 - x3;
    batch_output[offset] = first_even + second_even;
    batch_output[offset + 2] = first_even - second_even;
    batch_output[offset + 1] = first_odd + w3 * second_odd;
    batch_output[offset + 3] = first_odd - w3 * second_odd;
}

/**
 * @brief 单 block 融合的 batched Radix-2 FFT。
 *
 * 每个 block 处理一个 FFT，把位反转后的完整序列保存在 shared memory 中，所有蝶形级
 * 在同一个 kernel 内完成。与原始路径相比，它消除了逐级 kernel 启动和逐级全局内存
 * 往返；输入和输出允许指向同一设备缓冲区。
 */
__global__ void fused_shared_fft_batched_kernel(
    complex<float>* output,
    const complex<float>* input,
    int n,
    int batch,
    int ffts_per_block,
    int bits,
    int direction,
    float scale,
    const complex<float>* twiddles)
{
    // 当前选择器只对 N=64 启用本 kernel：4 个 FFT/block，共固定 256 个复数。
    // 使用静态 shared memory，避免 ZQ500 动态 shared launch 路径产生 Host 元数据增长。
    __shared__ complex<float> shared_data[256];
    const int tid = static_cast<int>(threadIdx.x);
    const int local_fft = tid / n;
    const int lane = tid - local_fft * n;
    const int batch_idx = static_cast<int>(blockIdx.x) * ffts_per_block + local_fft;
    const bool active = batch_idx < batch;
    const complex<float>* batch_input = input + batch_idx * n;
    complex<float>* batch_output = output + batch_idx * n;
    complex<float>* fft_shared = shared_data + local_fft * n;

    if (active) {
        const int reversed = bits == 0 ? 0 : (__brev(lane) >> (32 - bits));
        fft_shared[reversed] = batch_input[lane];
    }
    __syncthreads();

    for (int stage = 0; stage < bits; ++stage) {
        const int half = 1 << stage;
        const int group_size = half << 1;
        if (active && lane < n / 2) {
            const int butterfly = lane;
            const int group_idx = (butterfly / half) * group_size;
            const int butterfly_idx = butterfly % half;
            complex<float> twiddle = twiddles[butterfly_idx * (n / group_size)];
            if (direction < 0) {
                twiddle = complex<float>(twiddle.real(), -twiddle.imag());
            }
            const int offset = group_idx + butterfly_idx;
            const complex<float> a = fft_shared[offset];
            const complex<float> b = fft_shared[offset + half];
            fft_shared[offset] = a + twiddle * b;
            fft_shared[offset + half] = a - twiddle * b;
        }
        __syncthreads();
    }

    if (active) {
        batch_output[lane] = fft_shared[lane] * scale;
    }
}

void launch_fused_shared_fft(
    complex<float>* output,
    const complex<float>* input,
    int n,
    int batch,
    int direction,
    bool normalize,
    const complex<float>* twiddles)
{
    const int bits = static_cast<int>(log2f(n));
    const int ffts_per_block = 256 / n;
    const int threads = ffts_per_block * n;
    const int blocks = (batch + ffts_per_block - 1) / ffts_per_block;
    const float scale = normalize ? 1.0f / static_cast<float>(n) : 1.0f;
    fused_shared_fft_batched_kernel<<<blocks, threads>>>(
        output, input, n, batch, ffts_per_block, bits, direction, scale, twiddles);
}

__global__ void normalize_kernel(complex<float>* data, int n, float scale);

void execute_power_of_two_planned(
    complex<float>* data,
    int n,
    int batch,
    FFTDirection direction,
    const complex<float>* twiddles)
{
    const int fft_direction = direction == FFT_FORWARD ? 1 : -1;
    const int bits = static_cast<int>(log2f(n));
    const int total = n * batch;
    bit_reverse_batched_kernel<<<(total + 255) / 256, 256>>>(data, n, batch, bits);
    int stage = 0;
    for (; stage + 1 < bits; stage += 2) {
        const int butterflies = (n / 4) * batch;
        butterfly_radix4_batched_twiddle_kernel<<<(butterflies + 255) / 256, 256>>>(
            data, twiddles, n, batch, stage, fft_direction);
    }
    if (stage < bits) {
        const int butterflies = (n / 2) * batch;
        butterfly_batched_twiddle_kernel<<<(butterflies + 255) / 256, 256>>>(
            data, twiddles, n, batch, stage, fft_direction);
    }
    if (direction == FFT_INVERSE) {
        normalize_kernel<<<(total + 255) / 256, 256>>>(
            data, total, 1.0f / static_cast<float>(n));
    }
}

void execute_power_of_two_planned_outofplace(
    complex<float>* output,
    const complex<float>* input,
    int n,
    int batch,
    FFTDirection direction,
    const complex<float>* twiddles)
{
    if (n < 4) {
        cudaMemcpy(
            output,
            input,
            static_cast<std::size_t>(n) * batch * sizeof(complex<float>),
            cudaMemcpyDeviceToDevice);
        execute_power_of_two_planned(output, n, batch, direction, twiddles);
        return;
    }

    const int fft_direction = direction == FFT_FORWARD ? 1 : -1;
    const int bits = static_cast<int>(log2f(n));
    const int first_threads = (n / 4) * batch;
    bit_reverse_radix4_outofplace_kernel<<<(first_threads + 255) / 256, 256>>>(
        output, input, twiddles, n, batch, bits, fft_direction);
    int stage = 2;
    for (; stage + 1 < bits; stage += 2) {
        const int butterflies = (n / 4) * batch;
        butterfly_radix4_batched_twiddle_kernel<<<(butterflies + 255) / 256, 256>>>(
            output, twiddles, n, batch, stage, fft_direction);
    }
    if (stage < bits) {
        const int butterflies = (n / 2) * batch;
        butterfly_batched_twiddle_kernel<<<(butterflies + 255) / 256, 256>>>(
            output, twiddles, n, batch, stage, fft_direction);
    }
    if (direction == FFT_INVERSE) {
        const int total = n * batch;
        normalize_kernel<<<(total + 255) / 256, 256>>>(
            output, total, 1.0f / static_cast<float>(n));
    }
}

__global__ void normalize_kernel(
    complex<float>* data,
    int n,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

/**
 * @brief Row-major in_rows×in_cols → in_cols×in_rows transpose (one contiguous read per element).
 *
 * Replaces the previous column-FFT path that issued O(in_rows*in_cols) cudaMemcpy calls per pass.
 */
__global__ void transpose_rect_complex_kernel(
    const complex<float>* __restrict__ in,
    complex<float>* __restrict__ out,
    int in_rows,
    int in_cols)
{
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = in_rows * in_cols;
    if (idx >= total) {
        return;
    }
    const int r = idx / in_cols;
    const int c = idx % in_cols;
    out[c * in_rows + r] = in[r * in_cols + c];
}

__global__ void fft_shift_kernel(
    complex<float>* output,
    const complex<float>* input,
    int n,
    int half
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int new_idx = (idx + half) % n;
        output[idx] = input[new_idx];
    }
}

__global__ void fft_shift_2d_kernel(
    complex<float>* output,
    const complex<float>* input,
    int nx,
    int ny,
    int shift_x,
    int shift_y
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny;

    if (idx < total) {
        int x = idx % nx;
        int y = idx / nx;

        int new_x = (x + shift_x) % nx;
        int new_y = (y + shift_y) % ny;

        output[y * nx + x] = input[new_y * nx + new_x];
    }
}

__global__ void real_to_complex_full_kernel(
    complex<float>* output,
    const float* input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = complex<float>(input[idx], 0.0f);
    }
}

__global__ void conjugate_symmetry_kernel(
    complex<float>* spectrum,
    int n,
    int half_n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= 1 && idx < half_n - 1) {
        int sym_idx = n - idx;

        float real_part = spectrum[idx].real();
        float imag_part = spectrum[idx].imag();

        spectrum[sym_idx] = complex<float>(real_part, -imag_part);
    }
}

__global__ void extract_real_kernel(
    float* output,
    const complex<float>* input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx].real();
    }
}

void cooley_tukey_fft_raw(complex<float>* d_data, int n, int direction);

/**
 * @brief Chirp 信号生成 Kernel
 *
 * 生成 Bluestein 算法的 Chirp（线性调频）信号：
 *   chirp[k] = exp(-j * π * k² / N) * sign
 *
 * 用于将任意长度 N 的 DFT 转化为循环卷积问题。
 *
 * @param[out] output 输出 chirp 数组
 * @param[in]  n      原始信号长度 N（决定频率分辨率）
 * @param[in]  sign   符号因子：+1 正向变换，-1 逆向变换
 */
__global__ void chirp_kernel(
    complex<float>* output,
    int n,
    float sign
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const unsigned long long period =
            2ULL * static_cast<unsigned long long>(n);
        const unsigned long long squared =
            static_cast<unsigned long long>(idx) *
            static_cast<unsigned long long>(idx);
        const unsigned long long reduced = squared % period;
        float angle = -M_PI * static_cast<float>(reduced) / static_cast<float>(n);
        output[idx] = complex<float>(cosf(angle), sign * sinf(angle));
    }
}

/**
 * @brief Chirp 尾部逆序生成 Kernel（Bluestein 卷积核尾部）
 *
 * Bluestein 的卷积核 b 数组需要循环结构：
 *   b[0..N-1]     = chirp(0), chirp(1), ..., chirp(N-1)
 *   b[M-(N-1)..M-1] = chirp(N-1), chirp(N-2), ..., chirp(1)  ← 逆序！
 *
 * 本 Kernel 从数组末端**逆序**写入，实现负索引部分的 chirp 值。
 * 映射关系：b[M-1-k] = chirp(k+1)，k = 0..N-2
 *
 * @param[in,out] b       卷积核数组（在尾部位置写入）
 * @param[in]     n_tail  尾部元素个数 = N - 1
 * @param[in]     sign    符号因子（与主 chirp 相反）
 * @param[in]     n_orig  原始信号长度 N（用于角度计算）
 * @param[in]     m       扩展后长度 M = next_power_of_2(2N)
 */
__global__ void chirp_reverse_tail_kernel(
    complex<float>* b,
    int n_tail,
    float sign,
    int n_orig,
    int m
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_tail) {
        int k = idx + 1;
        int pos = m - 1 - idx;
        const unsigned long long period =
            2ULL * static_cast<unsigned long long>(n_orig);
        const unsigned long long squared =
            static_cast<unsigned long long>(k) *
            static_cast<unsigned long long>(k);
        const unsigned long long reduced = squared % period;
        float angle = -M_PI * static_cast<float>(reduced) /
            static_cast<float>(n_orig);
        b[pos] = complex<float>(cosf(angle), sign * sinf(angle));
    }
}

/**
 * @brief 复数逐点乘法 Kernel
 *
 * 对两个复数数组执行元素级乘法（用于频域卷积）。
 * 手动展开复数乘法公式，避免 thrust::complex 操作符的额外开销：
 *   (a+bi)(c+di) = (ac-bd) + (ad+bc)i
 *
 * @param[out] output  输出数组（c = a * b）
 * @param[in]  a       第一个输入数组
 * @param[in]  b       第二个输入数组
 * @param[in]  n       数组长度
 */
__global__ void pointwise_mul_kernel(
    complex<float>* output,
    const complex<float>* a,
    const complex<float>* b,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ar = a[idx].real(), ai = a[idx].imag();
        float br = b[idx].real(), bi = b[idx].imag();
        output[idx] = complex<float>(ar*br - ai*bi, ar*bi + ai*br);
    }
}

static bool is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

static int next_power_of_two(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

/**
 * @brief Bluestein's Chirp-Z Transform — 任意长度 FFT
 *
 * 将长度为 N 的 DFT 转化为 M 点循环卷积（M = next_power_of_2(2N)），
 * 然后通过 3 次 Cooley-Tukey FFT（2 正向 + 1 逆向）完成计算。
 *
 * 算法步骤：
 *   1. a[k] = x[k] * chirp(k, sign)，k = 0..N-1    （输入调制）
 *   2. 构建循环卷积核 b[0..M-1]                      （含尾部逆序）
 *   3. A = FFT(a), B = FFT(b)                         （频域变换）
 *   4. C = A .* B                                     （频域逐点乘 = 循环卷积）
 *   5. c = IFFT(C)，归一化 1/M                          （时域卷积结果）
 *   6. y[k] = c[k] * chirp(k, sign)，k = 0..N-1     （输出解调）
 *   7. 若 direction == INVERSE，额外归一化 1/N
 *
 * 复杂度：O(M log M)，其中 M ≤ 4N（最坏情况 N 为 Fermat 素数+1 时 M=2N）
 *
 * @param[in,out] d_data    输入/输出复数数组（GPU 设备内存，原地修改）
 * @param[in]     n         DFT 长度 N（任意正整数，N >= 1）
 * @param[in]     direction 变换方向：1=FFT_FORWARD, -1=FFT_INVERSE
 */
void bluestein_fft(
    complex<float>* d_data,
    int n,
    int direction,
    complex<float>* d_a,
    complex<float>* d_b) {
    if (is_power_of_two(n)) {
        cooley_tukey_fft_raw(d_data, n, direction);
        return;
    }

    int m = next_power_of_two(2 * n);
    float sign = (direction == FFT_FORWARD) ? 1.0f : -1.0f;
    cudaMemset(d_a, 0, m * sizeof(complex<float>));
    cudaMemset(d_b, 0, m * sizeof(complex<float>));

    chirp_kernel<<<(n + 255) / 256, 256>>>(d_a, n, sign);
    pointwise_mul_kernel<<<(n + 255) / 256, 256>>>(d_a, d_data, d_a, n);

    chirp_kernel<<<(n + 255) / 256, 256>>>(d_b, n, -sign);

    int tail_start = m - n + 1;
    if (tail_start > n) {
        chirp_reverse_tail_kernel<<<(n - 1 + 255) / 256, 256>>>(d_b, n - 1, -sign, n, m);
    }

    cooley_tukey_fft_raw(d_a, m, 1);
    cooley_tukey_fft_raw(d_b, m, 1);
    pointwise_mul_kernel<<<(m + 255) / 256, 256>>>(d_a, d_a, d_b, m);
    cooley_tukey_fft_raw(d_a, m, -1);

    float inv_m = 1.0f / m;
    normalize_kernel<<<(m + 255) / 256, 256>>>(d_a, m, inv_m);

    chirp_kernel<<<(n + 255) / 256, 256>>>(d_b, n, sign);
    pointwise_mul_kernel<<<(n + 255) / 256, 256>>>(d_data, d_a, d_b, n);

    if (direction == FFT_INVERSE) {
        float inv_n = 1.0f / n;
        normalize_kernel<<<(n + 255) / 256, 256>>>(d_data, n, inv_n);
    }

}

/**
 * @brief 带归一化的批量 Cooley-Tukey / Bluestein FFT（供 C2C 路径调用）
 *
 * 统一入口：根据 N 是否为 2 的幂自动选择算法。
 * 2 的幂路径包含位反转 + 蝶形运算 + 可选的 1/N 归一化。
 * 非 2 的幂路径委托给 bluestein_fft（内部已含归一化）。
 *
 * @param[in,out] d_data  复数数据数组（GPU 设备内存）
 * @param[in]     n       FFT 长度
 * @param[in]     batch   批处理数量
 * @param[in]     dir     变换方向
 */
void cooley_tukey_fft(
    complex<float>* d_data,
    int n,
    int batch,
    FFTDirection dir,
    complex<float>* bluestein_a = nullptr,
    complex<float>* bluestein_b = nullptr) {
    int fft_dir = (dir == FFT_FORWARD) ? 1 : -1;

    if (is_power_of_two(n)) {
        int bits = static_cast<int>(log2f(n));
        const int total = n * batch;
        bit_reverse_batched_kernel<<<(total + 255) / 256, 256>>>(d_data, n, batch, bits);
        for (int stage = 0; stage < bits; ++stage) {
            const int threads = (n / 2) * batch;
            butterfly_batched_kernel<<<(threads + 255) / 256, 256>>>(d_data, n, batch, stage, fft_dir);
        }
        if (dir == FFT_INVERSE) {
            float inv_n = 1.0f / n;
            normalize_kernel<<<(total + 255) / 256, 256>>>(d_data, total, inv_n);
        }
    } else {
        for (int b = 0; b < batch; b++) {
            bluestein_fft(d_data + b * n, n, fft_dir, bluestein_a, bluestein_b);
        }
    }
}

/**
 * @brief 无归一化的批量 FFT（供 R2C/C2R 路径调用）
 *
 * 与 cooley_tukey_fft 类似，但不执行 1/N 归一化。
 * R2C/C2R 路径由调用方自行控制归一化时机。
 *
 * @param[in,out] d_data    复数数据数组
 * @param[in]     n         FFT 长度
 * @param[in]     batch     批处理数量
 * @param[in]     direction 变换方向（1 或 -1）
 */
void cooley_tukey_fft_batch(
    complex<float>* d_data,
    int n,
    int batch,
    int direction,
    complex<float>* bluestein_a = nullptr,
    complex<float>* bluestein_b = nullptr) {
    int fft_dir = direction;

    if (is_power_of_two(n)) {
        int bits = static_cast<int>(log2f(n));
        const int total = n * batch;
        bit_reverse_batched_kernel<<<(total + 255) / 256, 256>>>(d_data, n, batch, bits);
        for (int stage = 0; stage < bits; ++stage) {
            const int threads = (n / 2) * batch;
            butterfly_batched_kernel<<<(threads + 255) / 256, 256>>>(d_data, n, batch, stage, fft_dir);
        }
    } else {
        for (int b = 0; b < batch; b++) {
            bluestein_fft(d_data + b * n, n, fft_dir, bluestein_a, bluestein_b);
        }
    }
}

/**
 * @brief 原始 FFT 核心（无归一化，无批处理）
 *
 * 最底层的 FFT 执行函数，被 bluestein_fft 递归调用。
 * 对 2 的幂长度执行标准 Cooley-Tukey（位反转 + 蝶形），
 * 对非 2 的幂长度委托给 bluestein_fft（避免无限递归）。
 *
 * @param[in,out] d_data    复数数据数组
 * @param[in]     n         FFT 长度
 * @param[in]     direction 变换方向：1=正向, -1=逆向
 */
void cooley_tukey_fft_raw(complex<float>* d_data, int n, int direction) {
    if (is_power_of_two(n)) {
        int bits = static_cast<int>(log2f(n));

        bit_reverse_kernel<<<(n + 255) / 256, 256>>>(d_data, n, bits);

        for (int stage = 0; stage < bits; ++stage) {
            int threads = n / 2;
            butterfly_kernel<<<(threads + 255) / 256, 256>>>(d_data, n, stage, direction);
        }
    } else {
        // 当前调用者只会把二次幂卷积长度传入 raw 路径。
    }
}

} // anonymous namespace

extern "C" {

FFTResult fft_plan_create_1d(
    FFTPlanHandle* plan,
    int nx,
    int batch,
    FFTType type,
    FFTDirection direction
) {
    if (!plan) return FFT_INVALID_VALUE;

    if (nx <= 0 || batch <= 0 ||
        (type != FFT_TYPE_C2C && type != FFT_TYPE_R2C && type != FFT_TYPE_C2R) ||
        (direction != FFT_FORWARD && direction != FFT_INVERSE)) {
        return FFT_INVALID_VALUE;
    }
    FFTPlan* p = new FFTPlan();
    p->nx = nx;
    p->ny = 0;
    p->batch = batch;
    p->type = type;
    p->direction = direction;
    p->d_buffer = nullptr;
    p->d_work_main = nullptr;
    p->d_work_aux = nullptr;
    p->d_twiddles = nullptr;
    p->d_bluestein_a = nullptr;
    p->d_bluestein_b = nullptr;
    p->bluestein_size = 0;

    if (type == FFT_TYPE_R2C || type == FFT_TYPE_C2R) {
        int size = (type == FFT_TYPE_R2C) ? (nx / 2 + 1) * batch : nx * batch;
        if (tracked_cuda_malloc(&p->d_buffer, size * sizeof(complex<float>)) != cudaSuccess) {
            delete p;
            return FFT_ALLOC_FAILED;
        }
    }

    if (!is_power_of_two(nx)) {
        p->bluestein_size = next_power_of_two(2 * nx);
    }
    int work_size = nx * batch;
    const int bluestein_work_size = 2 * p->bluestein_size;
    if (bluestein_work_size > work_size) {
        work_size = bluestein_work_size;
    }
    if (tracked_cuda_malloc(&p->d_work_main, work_size * sizeof(complex<float>)) != cudaSuccess) {
        if (p->d_buffer) tracked_cuda_free(p->d_buffer);
        delete p;
        return FFT_ALLOC_FAILED;
    }

    if (type == FFT_TYPE_R2C) {
        if (tracked_cuda_malloc(&p->d_work_aux, work_size * sizeof(complex<float>)) != cudaSuccess) {
            if (p->d_buffer) tracked_cuda_free(p->d_buffer);
            tracked_cuda_free(p->d_work_main);
            delete p;
            return FFT_ALLOC_FAILED;
        }
    } else if (type == FFT_TYPE_C2R) {
        if (tracked_cuda_malloc(&p->d_work_aux, work_size * sizeof(complex<float>)) != cudaSuccess) {
            if (p->d_buffer) tracked_cuda_free(p->d_buffer);
            tracked_cuda_free(p->d_work_main);
            delete p;
            return FFT_ALLOC_FAILED;
        }
    }

    if (is_power_of_two(nx) && nx > 1) {
        p->d_twiddles = p->d_work_main;
        initialize_twiddles_kernel<<<(nx / 2 + 255) / 256, 256>>>(p->d_twiddles, nx);
    } else {
        p->d_bluestein_a = p->d_work_main;
        p->d_bluestein_b = p->d_work_main + p->bluestein_size;
    }

    *plan = reinterpret_cast<FFTPlanHandle>(p);
    return FFT_SUCCESS;
}

FFTResult fft_plan_create_2d(
    FFTPlanHandle* plan,
    int nx,
    int ny,
    FFTType type,
    FFTDirection direction
) {
    if (!plan) return FFT_INVALID_VALUE;

    if (nx <= 0 || ny <= 0 ||
        (type != FFT_TYPE_C2C && type != FFT_TYPE_R2C && type != FFT_TYPE_C2R) ||
        (direction != FFT_FORWARD && direction != FFT_INVERSE)) {
        return FFT_INVALID_VALUE;
    }
    FFTPlan* p = new FFTPlan();
    p->nx = nx;
    p->ny = ny;
    p->batch = 1;
    p->type = type;
    p->direction = direction;
    p->d_buffer = nullptr;
    p->d_work_main = nullptr;
    p->d_work_aux = nullptr;
    p->d_twiddles = nullptr;
    p->d_bluestein_a = nullptr;
    p->d_bluestein_b = nullptr;
    p->bluestein_size = 0;

    if (type == FFT_TYPE_R2C || type == FFT_TYPE_C2R) {
        int size = nx * (ny / 2 + 1);
        if (tracked_cuda_malloc(&p->d_buffer, size * sizeof(complex<float>)) != cudaSuccess) {
            delete p;
            return FFT_ALLOC_FAILED;
        }
    }

    int total_size = nx * ny;
    if (!is_power_of_two(nx) || !is_power_of_two(ny)) {
        const int mx = is_power_of_two(nx) ? 0 : next_power_of_two(2 * nx);
        const int my = is_power_of_two(ny) ? 0 : next_power_of_two(2 * ny);
        p->bluestein_size = mx > my ? mx : my;
    }
    const int main_work_size = total_size + 2 * p->bluestein_size;
    if (tracked_cuda_malloc(&p->d_work_main, main_work_size * sizeof(complex<float>)) != cudaSuccess) {
        if (p->d_buffer) tracked_cuda_free(p->d_buffer);
        delete p;
        return FFT_ALLOC_FAILED;
    }
    if (tracked_cuda_malloc(&p->d_work_aux, total_size * sizeof(complex<float>)) != cudaSuccess) {
        if (p->d_buffer) tracked_cuda_free(p->d_buffer);
        tracked_cuda_free(p->d_work_main);
        delete p;
        return FFT_ALLOC_FAILED;
    }

    if (p->bluestein_size > 0) {
        p->d_bluestein_a = p->d_work_main + total_size;
        p->d_bluestein_b = p->d_bluestein_a + p->bluestein_size;
    }

    *plan = reinterpret_cast<FFTPlanHandle>(p);
    return FFT_SUCCESS;
}

FFTResult fft_execute(
    FFTPlanHandle plan,
    void* output,
    const void* input
) {
    if (!plan) return FFT_INVALID_VALUE;

    FFTPlan* p = reinterpret_cast<FFTPlan*>(plan);

    if (p->ny > 0) {
        const int total_size = p->nx * p->ny;
        complex<float>* d_data = p->d_work_main;
        complex<float>* d_tr = p->d_work_aux;

        cudaMemcpy(d_data, input, static_cast<size_t>(total_size) * sizeof(complex<float>), cudaMemcpyDeviceToDevice);

        const int fft_direction = (p->direction == FFT_FORWARD) ? 1 : -1;

        cooley_tukey_fft_batch(
            d_data, p->nx, p->ny, fft_direction, p->d_bluestein_a, p->d_bluestein_b);

        transpose_rect_complex_kernel<<<(total_size + 255) / 256, 256>>>(d_data, d_tr, p->ny, p->nx);

        cooley_tukey_fft_batch(
            d_tr, p->ny, p->nx, fft_direction, p->d_bluestein_a, p->d_bluestein_b);

        transpose_rect_complex_kernel<<<(total_size + 255) / 256, 256>>>(d_tr, d_data, p->nx, p->ny);

        if (p->direction == FFT_INVERSE) {
            float inv_total = 1.0f;
            if (is_power_of_two(p->nx)) {
                inv_total /= static_cast<float>(p->nx);
            }
            if (is_power_of_two(p->ny)) {
                inv_total /= static_cast<float>(p->ny);
            }
            if (std::abs(inv_total - 1.0f) > 1e-12f) {
                normalize_kernel<<<(total_size + 255) / 256, 256>>>(d_data, total_size, inv_total);
            }
        }

        cudaMemcpy(output, d_data, static_cast<size_t>(total_size) * sizeof(complex<float>), cudaMemcpyDeviceToDevice);

        return FFT_SUCCESS;
    }

    switch (p->type) {
        case FFT_TYPE_C2C: {
            complex<float>* d_output = reinterpret_cast<complex<float>*>(output);
            const complex<float>* d_input =
                reinterpret_cast<const complex<float>*>(input);
            if (is_power_of_two(p->nx)) {
                if (output == input) {
                    execute_power_of_two_planned(
                        d_output, p->nx, p->batch, p->direction, p->d_twiddles);
                } else {
                    execute_power_of_two_planned_outofplace(
                        d_output,
                        d_input,
                        p->nx,
                        p->batch,
                        p->direction,
                        p->d_twiddles);
                }
            } else {
                if (output != input) {
                    cudaMemcpy(
                        d_output,
                        d_input,
                        static_cast<std::size_t>(p->nx) * p->batch * sizeof(complex<float>),
                        cudaMemcpyDeviceToDevice);
                }
                cooley_tukey_fft(
                    d_output,
                    p->nx,
                    p->batch,
                    p->direction,
                    p->d_bluestein_a,
                    p->d_bluestein_b);
            }

            break;
        }
        case FFT_TYPE_R2C: {
            int half_n = p->nx / 2 + 1;
            int total_input_size = p->nx * p->batch;

            complex<float>* d_full_fft = p->d_work_aux;

            real_to_complex_full_kernel<<<(total_input_size + 255) / 256, 256>>>(
                d_full_fft,
                reinterpret_cast<const float*>(input),
                total_input_size
            );

            cooley_tukey_fft_batch(
                d_full_fft, p->nx, p->batch, 1, p->d_bluestein_a, p->d_bluestein_b);

            for (int b = 0; b < p->batch; b++) {
                cudaMemcpy(
                    reinterpret_cast<complex<float>*>(output) + b * half_n,
                    d_full_fft + b * p->nx,
                    half_n * sizeof(complex<float>),
                    cudaMemcpyDeviceToDevice
                );
            }

            break;
        }
        case FFT_TYPE_C2R: {
            int half_n = p->nx / 2 + 1;
            int total_output_size = p->nx * p->batch;

            complex<float>* d_full_spectrum = p->d_work_aux;

            cudaMemset(d_full_spectrum, 0, total_output_size * sizeof(complex<float>));

            for (int b = 0; b < p->batch; b++) {
                cudaMemcpy(
                    d_full_spectrum + b * p->nx,
                    reinterpret_cast<const complex<float>*>(input) + b * half_n,
                    half_n * sizeof(complex<float>),
                    cudaMemcpyDeviceToDevice
                );

                conjugate_symmetry_kernel<<<(half_n + 255) / 256, 256>>>(
                    d_full_spectrum + b * p->nx,
                    p->nx,
                    half_n
                );
            }

            cooley_tukey_fft_batch(
                d_full_spectrum,
                p->nx,
                p->batch,
                -1,
                p->d_bluestein_a,
                p->d_bluestein_b);

            for (int b = 0; b < p->batch; b++) {
                float inv_n = 1.0f / p->nx;
                normalize_kernel<<<(p->nx + 255) / 256, 256>>>(
                    d_full_spectrum + b * p->nx, p->nx, inv_n
                );

                extract_real_kernel<<<(p->nx + 255) / 256, 256>>>(
                    reinterpret_cast<float*>(output) + b * p->nx,
                    d_full_spectrum + b * p->nx,
                    p->nx
                );
            }

            break;
        }
        default:
            return FFT_INVALID_VALUE;
    }

    return FFT_SUCCESS;
}

FFTResult fft_execute_inplace(
    FFTPlanHandle plan,
    void* data
) {
    if (!plan) return FFT_INVALID_VALUE;

    FFTPlan* p = reinterpret_cast<FFTPlan*>(plan);

    if (p->type == FFT_TYPE_C2C) {
        if (is_power_of_two(p->nx)) {
            execute_power_of_two_planned(
                reinterpret_cast<complex<float>*>(data),
                p->nx,
                p->batch,
                p->direction,
                p->d_twiddles);
        } else {
            cooley_tukey_fft(
                reinterpret_cast<complex<float>*>(data),
                p->nx,
                p->batch,
                p->direction,
                p->d_bluestein_a,
                p->d_bluestein_b);
        }
        return FFT_SUCCESS;
    }

    return FFT_SUCCESS;
}

FFTResult fft_plan_destroy(FFTPlanHandle plan) {
    if (!plan) return FFT_INVALID_VALUE;

    FFTPlan* p = reinterpret_cast<FFTPlan*>(plan);
    cudaError_t release_status = cudaSuccess;
    if (p->d_buffer) {
        const cudaError_t status = tracked_cuda_free(p->d_buffer);
        if (status != cudaSuccess) release_status = status;
    }
    if (p->d_work_main) {
        const cudaError_t status = tracked_cuda_free(p->d_work_main);
        if (status != cudaSuccess) release_status = status;
    }
    if (p->d_work_aux) {
        const cudaError_t status = tracked_cuda_free(p->d_work_aux);
        if (status != cudaSuccess) release_status = status;
    }
    delete p;

    return release_status == cudaSuccess ? FFT_SUCCESS : FFT_INTERNAL_ERROR;
}

FFTResult fft_shift(
    void* output,
    const void* input,
    int nx,
    int batch
) {
    int half = nx / 2;

    for (int b = 0; b < batch; b++) {
        const complex<float>* in_batch =
            reinterpret_cast<const complex<float>*>(input) + b * nx;
        complex<float>* out_batch =
            reinterpret_cast<complex<float>*>(output) + b * nx;

        fft_shift_kernel<<<(nx + 255) / 256, 256>>>(
            out_batch, in_batch, nx, half
        );
    }

    return FFT_SUCCESS;
}

FFTResult fft_inverse_shift(
    void* output,
    const void* input,
    int nx,
    int batch
) {
    int half = nx / 2;

    for (int b = 0; b < batch; b++) {
        const complex<float>* in_batch =
            reinterpret_cast<const complex<float>*>(input) + b * nx;
        complex<float>* out_batch =
            reinterpret_cast<complex<float>*>(output) + b * nx;

        fft_shift_kernel<<<(nx + 255) / 256, 256>>>(
            out_batch, in_batch, nx, half
        );
    }

    return FFT_SUCCESS;
}

const char* fft_error_string(FFTResult result) {
    switch (result) {
        case FFT_SUCCESS: return "Success";
        case FFT_INVALID_VALUE: return "Invalid value";
        case FFT_ALLOC_FAILED: return "Allocation failed";
        case FFT_EXEC_FAILED: return "Execution failed";
        case FFT_INTERNAL_ERROR: return "Internal error";
        default: return "Unknown error";
    }
}

int fft_version() {
    return 1;
}

FFTResult fft_freq(float* freqs, int n, float d) {
    if (!freqs || n <= 0) return FFT_INVALID_VALUE;

    float inv_n_d = 1.0f / (n * d);
    int num_pos = (n + 1) / 2;

    for (int i = 0; i < num_pos; i++) {
        freqs[i] = i * inv_n_d;
    }

    for (int i = 0; i < n - num_pos; i++) {
        freqs[num_pos + i] = -(n - num_pos - i) * inv_n_d;
    }

    return FFT_SUCCESS;
}

FFTResult fft_rfftfreq(float* freqs, int n, float d) {
    if (!freqs || n <= 0) return FFT_INVALID_VALUE;

    int len = n / 2 + 1;
    float inv_n_d = 1.0f / (n * d);

    for (int i = 0; i < len; i++) {
        freqs[i] = i * inv_n_d;
    }

    return FFT_SUCCESS;
}

FFTResult fft_shift_2d(
    void* output,
    const void* input,
    int nx,
    int ny
) {
    if (!output || !input) return FFT_INVALID_VALUE;
    if (nx <= 0 || ny <= 0) return FFT_INVALID_VALUE;

    int shift_x = nx / 2;
    int shift_y = ny / 2;
    int total = nx * ny;

    fft_shift_2d_kernel<<<(total + 255) / 256, 256>>>(
        reinterpret_cast<complex<float>*>(output),
        reinterpret_cast<const complex<float>*>(input),
        nx, ny, shift_x, shift_y
    );

    return FFT_SUCCESS;
}

FFTResult fft_inverse_shift_2d(
    void* output,
    const void* input,
    int nx,
    int ny
) {
    if (!output || !input) return FFT_INVALID_VALUE;
    if (nx <= 0 || ny <= 0) return FFT_INVALID_VALUE;

    int shift_x = (nx + 1) / 2;
    int shift_y = (ny + 1) / 2;
    int total = nx * ny;

    fft_shift_2d_kernel<<<(total + 255) / 256, 256>>>(
        reinterpret_cast<complex<float>*>(output),
        reinterpret_cast<const complex<float>*>(input),
        nx, ny, shift_x, shift_y
    );

    return FFT_SUCCESS;
}

} // extern "C"
