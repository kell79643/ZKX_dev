#pragma once

#include <cusignal/runtime/device_array.h>
#include "operator_compute_policy.h"
#include <cusignal/backends/fft/fft_interface.h>

#include <cuda_fp16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace cusignal {
namespace detail {

template <typename T>
struct ConvolutionTypePolicy {
    static constexpr bool supported =
        std::is_same_v<T, float> || std::is_same_v<T, std::int32_t> ||
        std::is_same_v<T, std::int16_t> || std::is_same_v<T, std::int8_t>;

    __host__ __device__ static float load(T value)
    {
        return static_cast<float>(value);
    }

    __host__ __device__ static T store(float value)
    {
        return static_cast<T>(value);
    }
};

template <>
struct ConvolutionTypePolicy<__half> {
    static constexpr bool supported = true;

    __host__ __device__ static float load(__half value)
    {
        return __half2float(value);
    }

    __host__ __device__ static __half store(float value)
    {
        return __float2half_rn(value);
    }
};

template <typename T>
inline constexpr bool is_convolution_input_v = ConvolutionTypePolicy<T>::supported;

}  // namespace detail

/**
 * @brief `correlate`公开计算路径；对应cuSignal的`method="direct"`与`method="fft"`。
 * @details `auto`不形成第三套数学语义：无workspace重载显式走direct，可复用workspace重载
 * 显式走构建时选择的 FFT 后端，调用者按规模选择。两条路径共享相同的mode、shape、相关方向和边界dtype。
 */
enum class CorrelateMethod {
    direct,
    fft,
    auto_select,
};

/**
 * @brief cuSignal `correlate` 的任意 rank 公开参数。
 * @details 两个输入必须具有相同 rank；空 shape 表示标量。`direct` 仅允许 rank 0/1，
 * 与 cuSignal 23.08 一致；`fft` 支持任意 rank，`auto_select` 对多维稳定选择 FFT。
 */
struct CorrelateNDOptions {
    std::vector<int> shape1;
    std::vector<int> shape2;
    std::string mode = "full";
    CorrelateMethod method = CorrelateMethod::auto_select;
};

struct CorrelateNDOutputShape {
    std::vector<int> dimensions;
    std::size_t elements = 0;

    [[nodiscard]] std::size_t size() const noexcept { return elements; }
};

/**
 * @brief 校验任意 rank `correlate` 参数并计算输出 shape。
 * @return 标量返回空 dimensions 且 elements=1；任一零长维按 cuSignal FFT 空输入行为
 * 返回 dimensions={0}、elements=0；其余按 full/same/valid 逐维计算。
 */
CorrelateNDOutputShape correlate_nd_output_shape(const CorrelateNDOptions& options);

/**
 * @brief 计算一维正式`correlate`输出元素数。
 * @param len1 原始第一输入长度。
 * @param len2 原始第二输入长度。
 * @param mode `full`、`same`或`valid`。
 * @return 任一输入为空时为0；否则依次为`L1+L2-1`、`L1`、`abs(L1-L2)+1`。
 * @throws std::invalid_argument mode非法；空输入也不会跳过mode校验。
 */
std::size_t correlate_output_size(
    std::size_t len1,
    std::size_t len2,
    const std::string& mode);

struct CorrelateDeviceParams {
    int len1 = 0;
    int len2 = 0;
    std::string mode = "same";
};

struct CorrelateDeviceWorkspace {
    DeviceArray<ComplexFloat> in1_time;
    DeviceArray<ComplexFloat> in2_time;
    DeviceArray<ComplexFloat> in1_freq;
    DeviceArray<ComplexFloat> in2_freq;
    DeviceArray<ComplexFloat> product_freq;
    DeviceArray<ComplexFloat> ifft_time;
    std::unique_ptr<FFTInterface> fft;
    int len1 = 0;
    int len2 = 0;
    int full_len = 0;
    int fft_len = 0;
    int out_len = 0;
    std::string mode = "same";

    CorrelateDeviceWorkspace() = default;
    explicit CorrelateDeviceWorkspace(const CorrelateDeviceParams& params) { reset(params); }

    void reset(const CorrelateDeviceParams& params);
    [[nodiscard]] std::size_t fft_size() const noexcept
    {
        return static_cast<std::size_t>(fft_len);
    }
    [[nodiscard]] std::size_t output_size() const noexcept
    {
        return static_cast<std::size_t>(out_len);
    }
};

/**
 * @brief 任意 rank `correlate` 的可复用 GPU workspace。
 * @details FFT 路径把每一维 full 长度补到二次幂，逐轴 pack 后调用项目
 * `FFTInterface` 所选后端的 batched 1D C2C 计划，最后逐维 crop。所有数组均为连续
 * row-major，调用期间 workspace 不得并发共享。
 */
struct CorrelateNDWorkspace {
    std::vector<int> shape1;
    std::vector<int> shape2;
    std::vector<int> output_shape;
    std::vector<int> full_shape;
    std::vector<int> fft_shape;
    std::vector<int> crop_start;
    std::vector<int> strides1;
    std::vector<int> strides2;
    std::vector<int> output_strides;
    std::vector<int> fft_strides;
    CorrelateMethod requested_method = CorrelateMethod::auto_select;
    CorrelateMethod effective_method = CorrelateMethod::fft;
    std::string mode = "full";
    int rank = 0;
    int input1_elements = 0;
    int input2_elements = 0;
    int output_elements = 0;
    int fft_elements = 0;

    DeviceArray<int> d_shape1;
    DeviceArray<int> d_shape2;
    DeviceArray<int> d_strides1;
    DeviceArray<int> d_strides2;
    DeviceArray<int> d_output_strides;
    DeviceArray<int> d_fft_strides;
    DeviceArray<int> d_crop_start;
    DeviceArray<ComplexFloat> in1_time;
    DeviceArray<ComplexFloat> in2_time;
    DeviceArray<ComplexFloat> in1_freq;
    DeviceArray<ComplexFloat> in2_freq;
    DeviceArray<ComplexFloat> product_freq;
    DeviceArray<ComplexFloat> transform_scratch;
    std::vector<std::unique_ptr<FFTInterface>> axis_fft;

    CorrelateNDWorkspace() = default;
    explicit CorrelateNDWorkspace(const CorrelateNDOptions& options) { reset(options); }
    void reset(const CorrelateNDOptions& options);
    [[nodiscard]] std::size_t output_size() const noexcept
    {
        return static_cast<std::size_t>(output_elements);
    }
};

struct Correlate2DOptions {
    int rows1 = 0;
    int cols1 = 0;
    int rows2 = 0;
    int cols2 = 0;
    std::string mode = "same";
    std::string boundary = "fill";
    double fillvalue = 0.0;
};

struct Correlate2DOutputShape {
    int rows = 0;
    int cols = 0;

    [[nodiscard]] std::size_t size() const noexcept
    {
        return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    }
};

/**
 * @brief 校验`correlate2d`参数并计算正式输出shape。
 * @return 任一输入维度为0时稳定返回`{0,0}`；full、same和valid分别按契约计算。
 * @throws std::invalid_argument mode/boundary/维度/valid支配关系或32位索引容量非法。
 */
Correlate2DOutputShape correlate2d_output_shape(const Correlate2DOptions& options);

struct Correlate2DWorkspace {
    int rows1 = 0;
    int cols1 = 0;
    int rows2 = 0;
    int cols2 = 0;
    int out_rows = 0;
    int out_cols = 0;
    int row_start = 0;
    int col_start = 0;
    int boundary_code = 0;
    std::string mode = "same";
    std::string boundary = "fill";
    double fillvalue = 0.0;
    __half fill_fp16{};
    float fill_fp32 = 0.0F;
    std::int32_t fill_int32 = 0;
    std::int16_t fill_int16 = 0;
    std::int8_t fill_int8 = 0;
    bool integral_fill_valid = true;

    Correlate2DWorkspace() = default;
    explicit Correlate2DWorkspace(const Correlate2DOptions& options) { reset(options); }

    void reset(const Correlate2DOptions& options);
    [[nodiscard]] std::size_t output_size() const noexcept
    {
        return static_cast<std::size_t>(out_rows) * static_cast<std::size_t>(out_cols);
    }
};

/**
 * @brief 在 CPU 上计算一维实数 cross-correlation reference。
 * @tparam T FP32、FP16、INT32、INT16或INT8真实业务输入，两个输入必须同型。
 * @param in1 host 一维输入 `[L1]`。
 * @param in2 host 一维输入 `[L2]`。
 * @param mode `full`/`same`/`valid`，默认full；输出长度分别为`L1+L2-1`、
 * 原始`L1`、`abs(L1-L2)+1`。任一输入为空时返回空，但mode仍须合法。
 * @param method 与GPU正式重载逐项对应；direct模拟同型direct kernel，fft模拟ComplexFP32 FFT
 * 最终边界转换。默认direct。
 * @return host 一维 dtype T 的相关结果，第二输入按相关定义共轭翻转（实数即翻转）。
 * @throws std::invalid_argument mode、method或长度计算非法。
 * @details CPU独立O(L1*L2) reference，不调用GPU、FFTInterface或读取GPU结果。FP32/FP16和
 * 窄整数项目扩展用FP32累加；INT32 direct按确定性二补码模2^32乘加；fft整数按FP32结果
 * ties-to-even舍入后同宽环绕，不进行饱和。实数输入的共轭为恒等；不创建GPU stream或
 * device workspace。
 * @note 对齐本地cuSignal-23.08.00的rank-one real子集；FP16/INT16/INT8 direct为批准扩展。
 */
template <typename T>
std::vector<T> correlate_typed_cpu(
    const std::vector<T>& in1,
    const std::vector<T>& in2,
    const std::string& mode = "full",
    CorrelateMethod method = CorrelateMethod::direct);

/**
 * @brief 任意 rank、连续 row-major 实数 `correlate` 独立 CPU reference。
 * @details 公开 shape/mode/method 与 cuSignal 23.08 对齐；CPU 使用独立逐元素求和作为
 * reference，FFT 方法仍采用 cuSignal 的 FP32 计算边界和整数 round/cast 语义。
 */
template <typename T>
std::vector<T> correlate_typed_cpu(
    const std::vector<T>& in1,
    const std::vector<T>& in2,
    const CorrelateNDOptions& options);

/**
 * @brief 在 GPU 上用 direct kernel 计算一维实数 cross-correlation。
 * @tparam T 五种真实业务输入输出类型；该无workspace重载对应`method="direct"`。
 * @param in1 调用者持有的 device 一维输入 `[L1]`。
 * @param in2 调用者持有的 device 一维输入 `[L2]`。
 * @param out 调用者预分配的device输出，长度由`correlate_output_size`确定；空输入要求空输出。
 * @param mode `full`/`same`/`valid`，默认 full。
 * @throws std::invalid_argument mode、输出shape、32位索引容量或输入输出别名非法。
 * @details 默认stream custom direct kernel，无workspace、FFTInterface/dlrand、隐式传输或同步。
 * INT32用uint32模乘加后恢复二补码；INT16/INT8批准扩展用FP32累加后同宽环绕；FP16在
 * FP32累加后转回FP16。输出不可与输入重叠，读取前由调用者同步。
 * @note 对齐cuSignal rank-one direct的full/same/valid方向与shape；`same`始终为原始L1。
 */
template <typename T>
void correlate_device(
    const DeviceArray<T>& in1,
    const DeviceArray<T>& in2,
    DeviceArray<T>& out,
    const std::string& mode = "full");

/**
 * @brief 在 GPU 上使用调用者复用的 FFT workspace 计算一维 cross-correlation。
 * @tparam T 五种真实业务输入输出类型；该workspace重载对应`method="fft"`。
 * @param in1 device 输入 `[workspace.len1]`。
 * @param in2 device 输入 `[workspace.len2]`。
 * @param out device 输出 `[workspace.out_len]`。
 * @param workspace 调用者按 len1/len2/mode 初始化的对象，拥有六块 `fft_len` ComplexFloat
 * scratch 和 FFTInterface plan；同一对象不得并发共享。
 * @throws std::invalid_argument shape/workspace/mode不一致、输入输出别名、容量非法或plan缺失。
 * @details 在默认stream执行输入转ComplexFP32、反转第二输入、两次前向FFT、频域乘法、IFFT和
 * 居中裁剪；整数执行ties-to-even舍入及同宽环绕，绝不饱和。workspace拥有六块ComplexFP32
 * scratch和plan，不可并发复用；无隐式Host传输或同步，读取前由调用者同步。
 * @note cuSignal整数FFT内部可能提升到FP64；ZQ500禁止device FP64，因此本项目按批准规则用
 * ComplexFP32 FFT保持同T公开边界。空输入不创建plan并稳定返回空。
 */
template <typename T>
void correlate_device(
    const DeviceArray<T>& in1,
    const DeviceArray<T>& in2,
    DeviceArray<T>& out,
    CorrelateDeviceWorkspace& workspace);

/**
 * @brief 任意 rank、连续 row-major 实数 `correlate` 正式 GPU 入口。
 * @details rank>1 的 `direct` 按上游拒绝；FFT 路径实际调用构建时选择的 `FFTInterface` 后端，
 * 无隐式 H2D/D2H 或同步，读取结果前由调用者同步。
 */
template <typename T>
void correlate_device(
    const DeviceArray<T>& in1,
    const DeviceArray<T>& in2,
    DeviceArray<T>& out,
    CorrelateNDWorkspace& workspace);

/**
 * @brief 在CPU上独立实现cuSignal row-major二维实数cross-correlation reference。
 * @tparam T FP32、FP16、INT32、INT16或INT8真实同型输入，输出保持T。
 * @param in1 Host row-major第一矩阵，size必须为`rows1*cols1`。
 * @param in2 Host row-major第二矩阵，size必须为`rows2*cols2`。
 * @param options 两个shape、full/same/valid、fill/wrap/symm和Host scalar fillvalue。
 * @return Host row-major单数组；shape由`correlate2d_output_shape`确定。
 * @throws std::invalid_argument shape、mode、boundary、valid支配关系、fill转换或容量非法。
 * @details CPU不调用GPU或读取GPU结果；直接按constant、circular或edge-inclusive symmetric
 * 映射取样；valid不使用padding，因此忽略boundary/fillvalue。same对每个偶数kernel维使用
 * `floor(K/2)`起点；INT32按模2^32乘加，窄整数
 * 批准扩展使用FP32累加及同宽环绕，均不饱和。空维稳定返回空数组和`{0,0}`；不创建
 * GPU stream或device workspace。
 * @note 对齐本地cuSignal-23.08.00公开rank-two real接口；FP16/INT16/INT8为批准扩展。
 */
template <typename T>
std::vector<T> correlate2d_typed_cpu(
    const std::vector<T>& in1,
    const std::vector<T>& in2,
    const Correlate2DOptions& options);

/**
 * @brief 在GPU上执行完整公开二维实数cross-correlation。
 * @tparam T 五种真实同型输入输出；FP16/INT16/INT8为FP32合法路径扩展。
 * @param in1 调用者持有的device row-major第一矩阵。
 * @param in2 调用者持有的device row-major第二矩阵。
 * @param out 调用者按workspace输出shape预分配的device单数组，不允许与输入重叠。
 * @param workspace Host参数、输出shape、边界代码及预转换的五类型fill scalar；无device scratch。
 * @throws std::invalid_argument 参数、shape、valid关系、fill转换、别名或32位容量非法。
 * @details 默认stream启动一个boundary-aware direct kernel；fill执行T边界常量，wrap执行循环
 * 映射，symm执行包含端点的对称映射，非fill及valid忽略fillvalue。INT32在device用uint32模乘加；
 * 其余类型使用FP32累加并按T写回。无FFT、RAND、隐式传输、同步或device FP64。
 * @note 对齐本地cuSignal-23.08.00的same输出规则：保持原始in1 shape且偶数kernel按
 * 固定版偏移；读取输出前由调用者同步。
 */
template <typename T>
void correlate2d_device(
    const DeviceArray<T>& in1,
    const DeviceArray<T>& in2,
    DeviceArray<T>& out,
    Correlate2DWorkspace& workspace);

}  // namespace cusignal
