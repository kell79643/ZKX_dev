/**
 * @file fft_interface.h
 * @brief GPU FFT 抽象接口
 *
 * 本文件定义项目当前 GPU FFT 抽象层。源码中有两种 FFT 实现：
 *   1. fft_thrust.cu — Thrust 复数类型辅助的项目自研 ZQ500 GPU FFT
 *      - 2 的幂长度：Cooley-Tukey Radix-2 FFT
 *      - 非 2 的幂：Bluestein's Chirp-Z Transform（自动切换）
 *   2. fft_dlfft.cu — ZQ500 官方 dlfft 后端入口
 *      - 文本复用 fft_cufft.cu 的 cuFFT-compatible 共享封装
 *      - 使用 <cufftXt.h>，链接 ZQ500 dlfft/curt
 *
 * 当前正式 cusignal_cpp CMake 要求显式且互斥地设置 USE_DLFFT/USE_THRUST：
 * 前者选择官方 dlfft，后者选择项目自研 GPU FFT。两者必须恰好一个为 ON。
 * 没有有效的 NVIDIA USE_CUFFT 主工程选项；test_all 中的 cufft-compatible 目标
 * 只是固定链接 dlfft 的兼容封装专项验证，不是第三个后端。
 *
 * 设计原则：
 *   - 接口层不直接依赖 cufft/dlfft 头文件
 *   - 底层 C API 与上层 C++ 封装共享同一语义
 *   - GPU FFT 路径统一经过本抽象层，不在业务函数中写死 backend
 *   - 纯 CPU FFT backend 见 `fft_interface_cpu.h`
 *
 * 支持的功能矩阵：
 *   ┌─────────────┬─────────────┬────────────────────┐
 *   │ 功能        │ 项目 GPU FFT │ ZQ500 dlfft        │
 *   ├─────────────┼─────────────┼────────────────────┤
 *   │ 1D C2C FFT  │ ✅          │ ✅                 │
 *   │ 1D R2C/C2R  │ ✅          │ ✅                 │
 *   │ 2D C2C FFT  │ ✅          │ ✅                 │
 *   │ Batch FFT   │ ✅          │ ✅                 │
 *   │ 非幂次长度  │ Bluestein    │ 需补零/按ZQ500限制 │
 *   │ In-place    │ ✅ (C2C)    │ ✅ (C2C)           │
 *   │ fftshift    │ ✅          │ ✅                 │
 *   │ 2D fftshift │ ✅          │ ✅                 │
 *   │ 频率轴生成  │ ✅          │ ✅                 │
 *   └─────────────┴─────────────┴────────────────────┘
 *
 * 归一化约定（两种实现一致）：
 *   - 正向变换：不归一化
 *   - 逆向变换：结果自动除以 N（1/N 缩放）
 *   - 与 NumPy numpy.fft.ifft 行为一致
 *
 * @author cusignal_cpp
 * @date 2024-2026
 */

#ifndef FFT_INTERFACE_H
#define FFT_INTERFACE_H

/**
 * @brief CUDA 运行时头文件
 *
 * 提供 cudaMalloc, cudaMemcpy 等基础 CUDA 内存管理函数。
 * 所有实现都需要此头文件。
 */
#include <cuda_runtime.h>
#include <cusignal/runtime/device_array.h>

/**
 * @brief 标准复杂类型定义
 *
 * 注意：本接口不使用 std::complex，而是定义自己的复数结构体，
 * 以避免与具体实现（Thrust/cuFFT）的复杂类型冲突。
 */
#include <algorithm>
#include <complex>
#include <cstdint>
#include <type_traits>

#ifdef __cplusplus
/**
 * @brief C 接口标记
 *
 * 使用 extern "C" 确保 C++ 编译器按 C 风格处理函数符号，
 * 保证与 C 代码的二进制兼容性。
 */
extern "C" {
#endif

/**
 * @brief FFT 计划句柄
 *
 * FFT 计算前的准备工作（规划）被称为 "plan"。
 * plan 包含了 FFT 的大小、类型、方向等信息，以及可能的内部缓存。
 * 使用 opaque handle 模式，隐藏实现细节。
 *
 * 使用流程：
 *   1. fft_plan_create_1d() 创建计划
 *   2. fft_execute() 执行 FFT
 *   3. fft_plan_destroy() 销毁计划
 */
typedef void* FFTPlanHandle;

/**
 * @brief 单精度复数类型
 *
 * 替代 std::complex<float>，用于与具体实现解耦。
 * 内存布局与 cuFFT 的 cufftComplex 兼容。
 *
 * @note 当前实现中，我们使用 thrust::complex<float> 进行 GPU 计算，
 *       此结构体主要用于接口定义和 Host/Device 数据传输。
 */
typedef struct {
    float re;  /**< 实部 */
    float im;  /**< 虚部 */
} ComplexFloat;

/**
 * @brief 双精度复数类型
 *
 * 替代 std::complex<double>，用于需要更高精度的场景。
 */
typedef struct {
    double re; /**< 实部 */
    double im; /**< 虚部 */
} ComplexDouble;

/**
 * @brief FFT 变换方向
 *
 * 定义 FFT 的变换方向：
 *   - FFT_FORWARD:  正变换，时域 -> 频域
 *   - FFT_INVERSE: 逆变换，频域 -> 时域
 *
 * @note 数值与 cuFFT 定义保持一致（CUFFT_FORWARD=1, CUFFT_INVERSE=-1）
 */
typedef enum {
    FFT_FORWARD = 1,  /**< 正向 FFT：时域 -> 频域 */
    FFT_INVERSE = -1  /**< 逆向 FFT：频域 -> 时域 */
} FFTDirection;

/**
 * @brief FFT 变换类型
 *
 * 定义 FFT 的数据类型转换模式：
 *   - C2C: 复数输入 -> 复数输出
 *   - R2C: 实数输入 -> 复数输出（只需 N/2+1 个点）
 *   - C2R: 复数输入（仅前半部分）-> 实数输出
 *
 * @note R2C 和 C2R 用于实信号处理，可节省一半内存和计算量
 */
typedef enum {
    FFT_TYPE_C2C = 0, /**< 复数到复数：输入输出均为复数 */
    FFT_TYPE_R2C = 1, /**< 实数到复数：实数输入，N点产生 N/2+1 个复数输出 */
    FFT_TYPE_C2R = 2  /**< 复数到实数：N/2+1 个复数输入，输出 N 点实数 */
} FFTType;

/**
 * @brief FFT 操作结果状态码
 *
 * 所有 FFT 函数返回此类型报告执行状态。
 * 0 表示成功，非零表示错误。
 */
typedef enum {
    FFT_SUCCESS = 0,         /**< 操作成功 */
    FFT_INVALID_VALUE = 1,    /**< 参数无效（如空指针、非法大小） */
    FFT_ALLOC_FAILED = 2,     /**< 内存分配失败 */
    FFT_EXEC_FAILED = 3,      /**< FFT 执行失败 */
    FFT_INTERNAL_ERROR = 99   /**< 内部错误（未预期的错误） */
} FFTResult;

/**
 * @brief 创建 1D FFT 计划
 *
 * 分配并初始化 FFT 计算所需的资源和缓存。
 * 创建的计划必须通过 fft_plan_destroy() 释放。
 *
 * @param[out] plan     输出参数，计划句柄指针
 * @param[in] nx       FFT 数据长度（支持任意正整数，非 2 幂自动使用 Bluestein 算法）
 * @param[in]  batch    批处理数量，一次计算多个 FFT
 * @param[in]  type     FFT 类型（C2C, R2C, C2R）
 * @param[in]  direction 变换方向（正变换或逆变换）
 *
 * @return FFT_SUCCESS 如果成功，其他值表示错误
 *
 * @note 计划创建是耗时操作，应尽量复用已创建的计划
 * @note 建议在首次使用时创建计划并缓存，不要每次调用都创建/销毁
 *
 * @par 示例：
 * @code
 * FFTPlanHandle plan;
 * FFTResult res = fft_plan_create_1d(&plan, 1024, 1, FFT_TYPE_C2C, FFT_FORWARD);
 * if (res != FFT_SUCCESS) {
 *     // 处理错误
 * }
 * @endcode
 */
FFTResult fft_plan_create_1d(
    FFTPlanHandle* plan,  /**< [out] 计划句柄输出地址 */
    int nx,               /**< [in]  FFT 长度（2的幂） */
    int batch,            /**< [in]  批处理数量 */
    FFTType type,         /**< [in]  FFT 数据类型 */
    FFTDirection direction /**< [in]  变换方向 */
);

/**
 * @brief 创建 2D FFT 计划
 *
 * @param[out] plan      输出参数，计划句柄指针
 * @param[in]  nx       第一维长度（行数）
 * @param[in]  ny       第二维长度（列数）
 * @param[in]  type     FFT 类型（当前仅支持 C2C）
 * @param[in]  direction FFT 方向 (FFT_FORWARD 或 FFT_INVERSE)
 *
 * @return FFT_SUCCESS 如果成功
 *
 * @note 2D FFT 先对每一行做 FFT，再对每一列做 FFT
 */
FFTResult fft_plan_create_2d(
    FFTPlanHandle* plan,
    int nx,
    int ny,
    FFTType type,
    FFTDirection direction = FFT_FORWARD
);

/**
 * @brief 执行 FFT 变换
 *
 * 使用已创建的计划执行 FFT 变换。
 * 输入数据必须在 GPU 设备内存中。
 *
 * @param[in]  plan   已创建的计划句柄
 * @param[out] output 输出缓冲区（必须与输入同类型且足够大）
 * @param[in]  input  输入数据（必须在 GPU 内存中）
 *
 * @return FFT_SUCCESS 如果成功
 *
 * @par 数据大小说明：
 *   - C2C: output 和 input 大小均为 nx * sizeof(ComplexFloat)
 *   - R2C: input 大小为 nx * sizeof(float)，output 大小为 (nx/2+1) * sizeof(ComplexFloat)
 *   - C2R: input 大小为 (nx/2+1) * sizeof(ComplexFloat)，output 大小为 nx * sizeof(float)
 *
 * @note output 和 input 必须在 GPU 设备内存中
 * @note 批处理时，数据按行优先排列
 */
FFTResult fft_execute(
    FFTPlanHandle plan,  /**< [in]  FFT 计划句柄 */
    void* output,       /**< [out] 输出缓冲区 */
    const void* input   /**< [in]  输入数据 */
);

/**
 * @brief 原位执行 FFT 变换
 *
 * 在原位（in-place）执行 FFT，节省内存。
 * 输出覆盖输入。
 *
 * @param[in] plan  已创建的计划句柄
 * @param[inout] data 输入输出缓冲区（必须是复数类型 C2C）
 *
 * @return FFT_SUCCESS 如果成功
 *
 * @note 仅支持 C2C 类型
 * @note 此函数修改 data 内容
 */
FFTResult fft_execute_inplace(
    FFTPlanHandle plan,
    void* data
);

/**
 * @brief 销毁 FFT 计划
 *
 * 释放计划相关的所有资源。
 *
 * @param[in] plan 要销毁的计划句柄
 *
 * @return FFT_SUCCESS 如果成功
 *
 * @note 调用后，plan 变为无效，不能再使用
 * @note 应在不再需要 FFT 计算时调用
 */
FFTResult fft_plan_destroy(FFTPlanHandle plan);

/**
 * @brief 频谱中心移动（fftshift）
 *
 * 将零频率分量移动到频谱中心。
 * 用于使频谱图更直观，便于分析。
 *
 * 变换规则：
 *   对于长度为 N 的频谱，将前 N/2 点与后 N/2 点交换位置。
 *
 * @param[out] output 输出缓冲区
 * @param[in]  input  输入频谱
 * @param[in]  nx     频谱长度
 * @param[in]  batch  批处理数量
 *
 * @return FFT_SUCCESS 如果成功
 *
 * @par 示意图：
 *   输入: [0, 1, 2, ..., N/2-1, N/2, ..., N-1]
 *   输出: [N/2, ..., N-1, 0, 1, ..., N/2-1]
 *
 * @note output 和 input 可以是同一块内存（原地操作）
 */
FFTResult fft_shift(
    void* output,
    const void* input,
    int nx,
    int batch
);

/**
 * @brief 频谱中心移动逆操作（ifftshift）
 *
 * fftshift 的逆操作，将零频率分量从中心移回边缘。
 *
 * @param[out] output 输出缓冲区
 * @param[in]  input  输入频谱
 * @param[in]  nx     频谱长度
 * @param[in]  batch  批处理数量
 *
 * @return FFT_SUCCESS 如果成功
 *
 * @note 当输入是 fftshift 的输出时，用此函数恢复
 * @note 偶数长度时，fftshift 和 ifftshift 等价
 */
FFTResult fft_inverse_shift(
    void* output,
    const void* input,
    int nx,
    int batch
);

/**
 * @brief 2D 频谱中心移动（fftshift）
 *
 * 将二维频谱的零频率分量移动到中心位置。
 * 等价于对每一行做 1D fftshift，再对每一列做 1D fftshift。
 * 与 NumPy 的 numpy.fft.fftshift 行为一致。
 *
 * @par 变换规则：
 *   对于 (nx × ny) 的二维数组，将四个象限交换：
 *   - 左上 ↔ 右下
 *   - 右上 ↔ 左下
 *
 * @param[out] output 输出缓冲区（大小 ≥ nx*ny*sizeof(ComplexFloat)）
 * @param[in]  input  输入二维频谱（行优先存储）
 * @param[in]  nx     第一维长度（行方向元素数 / 列数）
 * @param[in]  ny     第二维长度（列方向元素数 / 行数）
 *
 * @return FFT_SUCCESS 如果成功
 *
 * @note output 和 input 可以是同一块内存（原地操作）
 * @note 偶数维度时，fftshift_2d 与 ifftshift_2d 等价
 */
FFTResult fft_shift_2d(
    void* output,
    const void* input,
    int nx,
    int ny
);

/**
 * @brief 2D 频谱中心移动逆操作（ifftshift）
 *
 * fftshift_2d 的逆操作，将零频率分量从中心移回左上角。
 *
 * @param[out] output 输出缓冲区
 * @param[in]  input  输入频谱
 * @param[in]  nx     第一维长度
 * @param[in]  ny     第二维长度
 *
 * @return FFT_SUCCESS 如果成功
 *
 * @note 偶数维度时与 fftshift_2d 等价
 * @note 奇数维度时两者有细微差别（中心点处理不同）
 */
FFTResult fft_inverse_shift_2d(
    void* output,
    const void* input,
    int nx,
    int ny
);

/**
 * @brief 获取错误信息字符串
 *
 * 将 FFTResult 错误码转换为可读的错误描述。
 *
 * @param[in] result FFT 操作返回的错误码
 *
 * @return 错误信息字符串（以 null 结尾）
 *
 * @par 示例：
 * @code
 * FFTResult res = fft_execute(plan, output, input);
 * if (res != FFT_SUCCESS) {
 *     printf("FFT error: %s\n", fft_error_string(res));
 * }
 * @endcode
 */
const char* fft_error_string(FFTResult result);

/**
 * @brief 获取 FFT 接口版本
 *
 * @return 版本号（当前为 1）
 *
 * @note 可用于运行时检查接口版本兼容性
 */
int fft_version();

/**
 * @brief 生成 DFT 采样频率轴（复数 FFT 对应的完整频率）
 *
 * 返回长度为 n 的浮点数组，表示 DFT 各频率分量的采样频率。
 * 与 NumPy 的 numpy.fft.fftfreq(n, d) 行为一致。
 *
 * @par 频率分布：
 *   - n 为偶数: [0, 1/(n*d), ..., (n/2-1)/(n*d), -n/2/(n*d), ..., -1/(n*d)]
 *   - n 为奇数: [0, 1/(n*d), ..., (n-1)/2/(n*d), -(n-1)/2/(n*d), ..., -1/(n*d)]
 *
 * @param[out] freqs    输出频率数组（必须已分配 n 个 float 的空间，可在 Host 或 Device 内存中）
 * @param[in]  n        DFT 点数
 * @param[in]  d        采样间隔（单位：秒），默认 1.0
 *
 * @return FFT_SUCCESS 如果成功
 *
 * @note 输出数组 freqs 必须预先分配至少 n * sizeof(float) 字节
 * @note 此函数在 Host 端执行（纯计算，无需 GPU 加速）
 */
FFTResult fft_freq(float* freqs, int n, float d = 1.0f);

/**
 * @brief 生成 DFT 采样频率轴（实数 FFT 对应的非负频率）
 *
 * 返回长度为 n//2+1 的浮点数组，表示实信号 R2C 变换后的非负频率分量。
 * 与 NumPy 的 numpy.fft.rfftfreq(n, d) 行为一致。
 *
 * @par 频率分布：
 *   - 始终返回非负频率: [0, 1/(n*d), 2/(n*d), ..., (n/2)/(n*d)]
 *   - 数组长度 = n // 2 + 1
 *
 * @param[out] freqs    输出频率数组（必须已分配 (n//2+1) 个 float 的空间，可在 Host 或 Device 内存中）
 * @param[in]  n        DFT 点数
 * @param[in]  d        采样间隔（单位：秒），默认 1.0
 *
 * @return FFT_SUCCESS 如果成功
 *
 * @note 输出数组 freqs 必须预先分配至少 (n/2+1) * sizeof(float) 字节
 * @note 此函数在 Host 端执行（纯计算，无需 GPU 加速）
 */
FFTResult fft_rfftfreq(float* freqs, int n, float d = 1.0f);

#ifdef __cplusplus
}
#endif

#include <vector>
#include <string>

namespace cusignal {

using complexd = std::complex<double>;

inline std::size_t fft_next_power_of_two(std::size_t logical_size)
{
    std::size_t padded_size = 1;
    while (padded_size < logical_size) {
        padded_size <<= 1;
    }
    return padded_size;
}

inline std::vector<complexd> fft_prepare_complex_input(
    const std::vector<complexd>& input,
    std::size_t padded_size = 0)
{
    if (padded_size == 0) {
        padded_size = fft_next_power_of_two(input.size());
    }
    std::vector<complexd> output(padded_size, complexd(0.0, 0.0));
    const std::size_t copy_count = input.size() < padded_size ? input.size() : padded_size;
    std::copy(input.begin(), input.begin() + copy_count, output.begin());
    return output;
}

template <typename T>
std::vector<complexd> fft_promote_real_to_complex(
    const std::vector<T>& input,
    std::size_t padded_size = 0)
{
    static_assert(std::is_arithmetic<T>::value, "FFT real input promotion requires an arithmetic type");
    if (padded_size == 0) {
        padded_size = fft_next_power_of_two(input.size());
    }
    std::vector<complexd> output(padded_size, complexd(0.0, 0.0));
    const std::size_t copy_count = input.size() < padded_size ? input.size() : padded_size;
    for (std::size_t i = 0; i < copy_count; ++i) {
        output[i] = complexd(static_cast<double>(input[i]), 0.0);
    }
    return output;
}

class FFTInterface {
public:
    struct BatchOnly {};

    FFTInterface(int n);
    FFTInterface(int n, BatchOnly);
    ~FFTInterface() noexcept;
    FFTInterface(const FFTInterface&) = delete;
    FFTInterface& operator=(const FFTInterface&) = delete;
    FFTInterface(FFTInterface&&) = delete;
    FFTInterface& operator=(FFTInterface&&) = delete;
    std::vector<complexd> fft(const std::vector<complexd>& input);
    std::vector<complexd> ifft(const std::vector<complexd>& input);
    void fft_device(const DeviceArray<ComplexFloat>& input, DeviceArray<ComplexFloat>& output);
    void ifft_device(const DeviceArray<ComplexFloat>& input, DeviceArray<ComplexFloat>& output);
    void fft_device(const ComplexFloat* input, ComplexFloat* output, std::size_t count);
    void ifft_device(const ComplexFloat* input, ComplexFloat* output, std::size_t count);
    void fft_inplace_device(DeviceArray<ComplexFloat>& data);
    void ifft_inplace_device(DeviceArray<ComplexFloat>& data);
    void fft_batch_device(const DeviceArray<ComplexFloat>& input, DeviceArray<ComplexFloat>& output, int batch);
    void ifft_batch_device(const DeviceArray<ComplexFloat>& input, DeviceArray<ComplexFloat>& output, int batch);
    void fft_batch_device(const ComplexFloat* input, ComplexFloat* output, int batch, std::size_t count);
    void ifft_batch_device(const ComplexFloat* input, ComplexFloat* output, int batch, std::size_t count);
    std::vector<std::vector<complexd>> fft_batch(const std::vector<std::vector<complexd>>& input);
    std::vector<std::vector<complexd>> ifft_batch(const std::vector<std::vector<complexd>>& input);
    std::vector<ComplexFloat> ifft_batch_float(const std::vector<ComplexFloat>& input, int batch);
    std::vector<complexd> rfft(const std::vector<complexd>& input);
    std::vector<complexd> irfft(const std::vector<complexd>& input, int output_len = 0);
private:
    FFTInterface(int n, bool batch_only);
    int n_;
    FFTPlanHandle plan_forward_;
    FFTPlanHandle plan_inverse_;
    FFTPlanHandle plan_r2c_;
    FFTPlanHandle plan_c2r_;
    void* d_input_;
    void* d_output_;
};

class FFTInterface2D {
public:
    FFTInterface2D(int nx, int ny);
    FFTInterface2D(int nx, int ny, FFTInterface::BatchOnly);
    ~FFTInterface2D() noexcept;
    FFTInterface2D(const FFTInterface2D&) = delete;
    FFTInterface2D& operator=(const FFTInterface2D&) = delete;
    FFTInterface2D(FFTInterface2D&&) = delete;
    FFTInterface2D& operator=(FFTInterface2D&&) = delete;
    std::vector<std::vector<complexd>> fft2d(const std::vector<std::vector<complexd>>& input);
    std::vector<std::vector<complexd>> ifft2d(const std::vector<std::vector<complexd>>& input);
    void fft2d_device(const DeviceArray<ComplexFloat>& input, DeviceArray<ComplexFloat>& output);
    void ifft2d_device(const DeviceArray<ComplexFloat>& input, DeviceArray<ComplexFloat>& output);
private:
    FFTInterface2D(int nx, int ny, bool device_only);
    int nx_, ny_;
    FFTPlanHandle plan_forward_;
    FFTPlanHandle plan_inverse_;
    void* d_input_;
    void* d_output_;
};

} // namespace cusignal

#endif
