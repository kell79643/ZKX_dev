#pragma once

#include <cusignal/runtime/device_array.h>
#include "simple_signal_typed.h"
#include <cusignal/backends/fft/fft_interface.h>

#include <optional>
#include <type_traits>
#include <vector>

namespace cusignal {

template <typename T> struct TypedComplex { T real; T imag; };

namespace detail {
template <typename T> struct IsTypedComplexInput : std::false_type {};
template <typename T>
struct IsTypedComplexInput<TypedComplex<T>>
    : std::bool_constant<is_simple_signal_input_v<T>> {};
template <typename T>
inline constexpr bool is_freq_shift_input_v =
    is_simple_signal_input_v<T> || IsTypedComplexInput<T>::value;
}

enum class DetrendMode { linear, constant };
enum class DetrendOutputDtype { fp16, fp32, fp64 };

struct DetrendOptions {
    std::vector<int> shape;
    int axis = -1;
    DetrendMode mode = DetrendMode::linear;
    std::vector<int> breakpoints;
    bool overwrite_data = false;
};

struct DetrendCpuResult {
    DetrendOutputDtype dtype{DetrendOutputDtype::fp32};
    std::vector<__half> fp16;
    std::vector<float> fp32;
    std::vector<double> fp64;
};

struct DetrendDeviceResult {
    DetrendOutputDtype dtype{DetrendOutputDtype::fp32};
    DeviceArray<__half> fp16;
    DeviceArray<float> fp32;
    DeviceArray<double> fp64;
};

struct ChannelizePolyCpuResult {
    std::vector<ComplexFloat> values;
    std::vector<int> shape;
};

struct ChannelizePolyDeviceResult {
    DeviceArray<ComplexFloat> values;
    std::vector<int> shape;
};

enum class HilbertOutputDtype { complex_fp32, complex_fp64 };

struct HilbertOptions {
    std::vector<int> shape;
    std::optional<int> fft_length;
    int axis = -1;
};

struct HilbertCpuResult {
    HilbertOutputDtype dtype{HilbertOutputDtype::complex_fp32};
    std::vector<ComplexFloat> complex_fp32;
    std::vector<ComplexDouble> complex_fp64;
    std::vector<int> shape;
};

struct HilbertDeviceResult {
    HilbertOutputDtype dtype{HilbertOutputDtype::complex_fp32};
    DeviceArray<ComplexFloat> complex_fp32;
    DeviceArray<ComplexDouble> complex_fp64;
    std::vector<int> shape;
};

struct Hilbert2Options {
    std::vector<int> shape;
    std::vector<int> fft_shape;
};

struct Hilbert2CpuResult {
    HilbertOutputDtype dtype{HilbertOutputDtype::complex_fp32};
    std::vector<ComplexFloat> complex_fp32;
    std::vector<ComplexDouble> complex_fp64;
    std::vector<int> shape;
};

struct Hilbert2DeviceResult {
    HilbertOutputDtype dtype{HilbertOutputDtype::complex_fp32};
    DeviceArray<ComplexFloat> complex_fp32;
    DeviceArray<ComplexDouble> complex_fp64;
    std::vector<int> shape;
};

struct LfilterZiDeviceResult {
    DeviceArray<double> fp64;
};

struct SosfiltOptions {
    std::vector<int> shape;
    int axis = -1;
    std::vector<int> zi_shape;
};

struct SosfiltCpuResult {
    std::vector<float> y;
    std::vector<float> zf;
    std::vector<int> y_shape;
    std::vector<int> zf_shape;
    bool has_zf = false;
};

struct SosfiltDeviceResult {
    DeviceArray<float> y;
    DeviceArray<float> zf;
    std::vector<int> y_shape;
    std::vector<int> zf_shape;
    bool has_zf = false;
};

struct WienerOptions {
    std::vector<int> shape;
    std::optional<int> scalar_window;
    std::vector<int> window_shape;
    std::optional<double> noise;
};

struct WienerCpuResult {
    std::vector<double> fp64;
    std::vector<int> shape;
};

struct WienerDeviceResult {
    DeviceArray<double> fp64;
    std::vector<int> shape;
};

enum class DecimateOutputDtype { fp32, fp64 };

struct DecimateOptions {
    std::vector<int> shape;
    int axis = -1;
    std::optional<int> filter_order;
    bool zero_phase = true;
};

struct DecimateCpuResult {
    DecimateOutputDtype dtype{DecimateOutputDtype::fp64};
    std::vector<float> fp32;
    std::vector<double> fp64;
    std::vector<int> shape;
};

struct DecimateDeviceResult {
    DecimateOutputDtype dtype{DecimateOutputDtype::fp64};
    DeviceArray<float> fp32;
    DeviceArray<double> fp64;
    std::vector<int> shape;
};

enum class ResampleDomain { time, frequency };
enum class ResampleOutputDtype { fp32, fp64 };

struct ResampleOptions {
    std::vector<int> shape;
    int axis = 0;
    ResampleDomain domain = ResampleDomain::time;
    std::vector<double> frequency_window;
};

struct ResampleCpuResult {
    ResampleOutputDtype dtype{ResampleOutputDtype::fp32};
    std::vector<float> fp32;
    std::vector<double> fp64;
    std::vector<double> time;
    std::vector<int> shape;
    bool has_time = false;
};

struct ResampleDeviceResult {
    ResampleOutputDtype dtype{ResampleOutputDtype::fp32};
    DeviceArray<float> fp32;
    DeviceArray<double> fp64;
    DeviceArray<double> time;
    std::vector<int> shape;
    bool has_time = false;
};

enum class ResamplePolyWindowMode { kaiser, hamming, explicit_values };
enum class ResamplePolyOutputDtype { fp16, fp32, fp64, int32, int16, int8 };
enum class UpfirdnOutputDtype { fp32, fp64 };

struct UpfirdnOptions {
    std::vector<int> shape;
    int axis = -1;
};

struct UpfirdnCpuResult {
    UpfirdnOutputDtype dtype{UpfirdnOutputDtype::fp32};
    std::vector<float> fp32;
    std::vector<double> fp64;
    std::vector<int> shape;
};

struct UpfirdnDeviceResult {
    UpfirdnOutputDtype dtype{UpfirdnOutputDtype::fp32};
    DeviceArray<float> fp32;
    DeviceArray<double> fp64;
    std::vector<int> shape;
};

struct ResamplePolyOptions {
    std::vector<int> shape;
    int axis = 0;
    ResamplePolyWindowMode window_mode = ResamplePolyWindowMode::kaiser;
    double kaiser_beta = 5.0;
    std::vector<double> design_window;
    bool gpupath = true;
};

struct ResamplePolyCpuResult {
    ResamplePolyOutputDtype dtype{ResamplePolyOutputDtype::fp64};
    std::vector<__half> fp16;
    std::vector<float> fp32;
    std::vector<double> fp64;
    std::vector<std::int32_t> int32;
    std::vector<std::int16_t> int16;
    std::vector<std::int8_t> int8;
    std::vector<int> shape;
};

struct ResamplePolyDeviceResult {
    ResamplePolyOutputDtype dtype{ResamplePolyOutputDtype::fp64};
    DeviceArray<__half> fp16;
    DeviceArray<float> fp32;
    DeviceArray<double> fp64;
    DeviceArray<std::int32_t> int32;
    DeviceArray<std::int16_t> int16;
    DeviceArray<std::int8_t> int8;
    std::vector<int> shape;
};

struct FirfilterOptions {
    std::vector<int> shape;
    int axis = -1;
    std::vector<int> zi_shape;
};

struct FirfilterCpuResult {
    std::vector<float> y;
    std::vector<float> zf;
    std::vector<int> y_shape;
    std::vector<int> zf_shape;
    bool has_zf = false;
};

struct FirfilterDeviceResult {
    DeviceArray<float> y;
    DeviceArray<float> zf;
    std::vector<int> y_shape;
    std::vector<int> zf_shape;
    bool has_zf = false;
};

enum class Firfilter2PadType { odd, even, constant, none };
enum class Firfilter2Method { pad, gust };
enum class Firfilter2OutputDtype { fp32, fp64 };

struct Firfilter2Options {
    std::vector<int> shape;
    int axis = -1;
    Firfilter2PadType padtype = Firfilter2PadType::odd;
    std::optional<int> padlen;
    Firfilter2Method method = Firfilter2Method::pad;
    std::optional<int> irlen;
};

struct Firfilter2CpuResult {
    Firfilter2OutputDtype dtype{Firfilter2OutputDtype::fp32};
    std::vector<float> fp32;
    std::vector<double> fp64;
    std::vector<int> shape;
};

struct Firfilter2DeviceResult {
    Firfilter2OutputDtype dtype{Firfilter2OutputDtype::fp32};
    DeviceArray<float> fp32;
    DeviceArray<double> fp64;
    std::vector<int> shape;
};

namespace detail {
struct ChannelizePolyPlan {
    int channel_count = 0;
    int tap_count = 0;
    int point_count = 0;
    int input_count = 0;
    std::size_t output_size = 0;
};

ChannelizePolyPlan prepare_channelize_poly(
    std::size_t input_size, std::size_t filter_size, int n_chans);

ChannelizePolyPlan prepare_channelize_poly(
    std::size_t input_size, std::size_t filter_size, int n_chans,
    const std::vector<int>& input_shape,
    const std::vector<int>& filter_shape);

struct HilbertLayoutPlan {
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    int axis = 0;
    int input_axis_length = 0;
    int fft_length = 0;
    int inner_count = 0;
    int outer_count = 0;
    int line_count = 0;
    std::size_t output_size = 0;
};

HilbertLayoutPlan prepare_hilbert_layout(
    std::size_t input_size, const HilbertOptions& options);

struct Hilbert2LayoutPlan {
    int input_rows = 0;
    int input_cols = 0;
    int fft_rows = 0;
    int fft_cols = 0;
    int output_rows = 0;
    int output_cols = 0;
    std::size_t fft_size = 0;
    std::size_t output_size = 0;
};

Hilbert2LayoutPlan prepare_hilbert2_layout(
    std::size_t input_size, const Hilbert2Options& options);

struct LfilterZiLayoutPlan {
    int denominator_offset = 0;
    int denominator_count = 0;
    int common_count = 0;
    int output_count = 0;
    std::size_t augmented_count = 0;
};

LfilterZiLayoutPlan prepare_lfilter_zi_layout(
    std::size_t numerator_count, std::size_t denominator_count,
    std::size_t denominator_offset);

struct SosfiltLayoutPlan {
    std::vector<int> input_shape;
    std::vector<int> state_shape;
    int axis = 0;
    int sections = 0;
    int sample_count = 0;
    int inner_count = 0;
    int outer_count = 0;
    int line_count = 0;
    std::size_t state_size = 0;
};

SosfiltLayoutPlan prepare_sosfilt_layout(
    std::size_t input_size, std::size_t sos_size, int sections,
    const SosfiltOptions& options, bool has_zi, std::size_t zi_size);

void validate_sos_coefficients(
    const std::vector<float>& coefficients, int sections);

struct WienerLayoutPlan {
    std::vector<int> input_shape;
    std::vector<int> input_strides;
    std::vector<int> window_shape;
    int rank = 0;
    int element_count = 0;
    int window_volume = 0;
};

WienerLayoutPlan prepare_wiener_layout(
    std::size_t input_size, const WienerOptions& options);

struct DecimateLayoutPlan {
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    int axis = 0;
    int sample_count = 0;
    int output_sample_count = 0;
    int inner_count = 0;
    int outer_count = 0;
    int line_count = 0;
    std::size_t output_size = 0;
};

DecimateLayoutPlan prepare_decimate_layout(
    std::size_t input_size, int q, const DecimateOptions& options);

struct ResampleLayoutPlan {
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    int axis = 0;
    int input_sample_count = 0;
    int output_sample_count = 0;
    int inner_count = 0;
    int outer_count = 0;
    int line_count = 0;
    std::size_t input_workspace_size = 0;
    std::size_t output_size = 0;
};

ResampleLayoutPlan prepare_resample_layout(
    std::size_t input_size, int output_sample_count,
    const ResampleOptions& options);

struct UpfirdnLayoutPlan {
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    int axis = 0;
    int up = 1;
    int down = 1;
    int input_sample_count = 0;
    int output_sample_count = 0;
    int inner_count = 0;
    int outer_count = 0;
    int line_count = 0;
    int coefficient_count = 0;
    std::size_t output_size = 0;
};

UpfirdnLayoutPlan prepare_upfirdn_layout(
    std::size_t input_size, std::size_t coefficient_count,
    int up, int down, const UpfirdnOptions& options);

/** @brief 在 Host 侧完成 upfirdn 的 FP32 到 FP64 存储扩宽。 */
DeviceArray<double> finalize_upfirdn_fp64_output(
    const DeviceArray<float>& computed);

struct ResamplePolyLayoutPlan {
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    int axis = 0;
    int up = 1;
    int down = 1;
    int input_sample_count = 0;
    int output_sample_count = 0;
    int inner_count = 0;
    int outer_count = 0;
    int line_count = 0;
    int coefficient_count = 0;
    int half_length = 0;
    int prefix_zeros = 0;
    int suffix_zeros = 0;
    int output_start = 0;
    bool identity = false;
    bool custom_coefficients = false;
    std::size_t output_size = 0;
};

ResamplePolyLayoutPlan prepare_resample_poly_layout(
    std::size_t input_size, int up, int down,
    const ResamplePolyOptions& options,
    bool custom_coefficients, std::size_t coefficient_count);

struct FirfilterLayoutPlan {
    std::vector<int> input_shape;
    std::vector<int> state_shape;
    std::vector<int> zi_line_offsets;
    int sample_count = 0;
    int state_count = 0;
    int inner_count = 0;
    int outer_count = 0;
    int zi_state_stride = 0;
};

FirfilterLayoutPlan prepare_firfilter_layout(
    std::size_t input_size, std::size_t coefficient_count,
    const FirfilterOptions& options, bool has_zi, std::size_t zi_size);

struct Firfilter2LayoutPlan {
    std::vector<int> input_shape;
    std::vector<int> extended_shape;
    int axis = 0;
    int sample_count = 0;
    int extended_sample_count = 0;
    int edge = 0;
    int inner_count = 0;
    int outer_count = 0;
    int state_count = 0;
};

Firfilter2LayoutPlan prepare_firfilter2_layout(
    std::size_t input_size, std::size_t coefficient_count,
    const Firfilter2Options& options);
}

/**
 * @brief 在GPU上执行cuSignal一维polyphase channelizer完整正式封装。
 * @tparam T x与h共同采用的FP32、FP16、INT32、INT16或INT8真实业务dtype。
 * @param x 调用者持有的连续device一维输入；尾部不足n_chans的一组按cuSignal忽略。
 * @param h 调用者持有的连续device一维FIR taps；尾部不足n_chans的一组同样忽略且允许为空。
 * @param n_chans 正channel数；输出shape第一维。
 * @param out 封装重建的ComplexFP32 device矩阵及`{n_chans,floor(N/n_chans)}` shape。
 * @throws std::invalid_argument n_chans非正、索引容量溢出或每相位taps超过32。
 * @details 默认stream上先执行FP32多相累加，再以所选FFTInterface后端进行batched FFT、共轭和
 * channel-major转置；polyphase、FFT输出和plan构成一次性workspace。无device FP64、RAND、
 * H2D/D2H或调用者外部workspace；FP16和整数为已批准的cuSignal五类型扩展。
 */
template <typename T>
void channelize_poly_device(
    const DeviceArray<T>& x, const DeviceArray<T>& h, int n_chans,
    ChannelizePolyDeviceResult& out);

/** @brief 固定版23.08以`len(x)`和`len(h)`取首维的高维兼容入口。 */
template <typename T>
void channelize_poly_device(
    const DeviceArray<T>& x, const DeviceArray<T>& h, int n_chans,
    const std::vector<int>& input_shape,
    const std::vector<int>& filter_shape,
    ChannelizePolyDeviceResult& out);
/**
 * @brief 在GPU上按cuSignal语义沿指定axis去除常量或分段线性趋势。
 * @tparam T FP32、FP16、INT32、INT16或INT8真实业务输入。
 * @param x 调用者持有的连续row-major device输入；元素数必须等于options.shape乘积。
 * @param out 输出dtype variant；函数会释放其旧内容并仅填充当前dtype对应的device数组。
 * @param options shape、axis、linear/constant、bp及overwrite_data完整选项；shape为空表示一维。
 * @throws std::invalid_argument shape乘积、axis、空linear轴、bp范围或int索引容量非法。
 * @details custom kernel只进行FP32计算并使用默认stream；FP64结果先D2H，在Host拓宽后H2D，
 * 不进行device FP64运算。breakpoint及FP32输入/输出构成内部workspace。linear且T=FP32时，
 * overwrite_data=true会把结果D2D写回x；其余dtype转换路径不修改输入。
 */
template <typename T>
void detrend_device(
    DeviceArray<T>& x, DetrendDeviceResult& out,
    const DetrendOptions& options = {});
/**
 * @brief 在GPU上按cuSignal语义沿指定axis执行causal FIR filter。
 * @tparam T 五种真实业务输入/系数类型；不原生支持的FP16和整数走已批准FP32合法路径。
 * @param b device非空一维FIR系数`[Nb]`。
 * @param x device连续row-major输入；元素数必须等于options.shape乘积。
 * @param out 结果结构；始终填充FP32 y，传入zi时还填充FP32 zf及两者shape。
 * @param options 输入shape、axis及可选zi_shape；shape为空表示一维，zi_shape允许非axis维广播1。
 * @param zi 可选device初态；非空指针表示按cuSignal返回`(y,zf)`，其shape须符合options。
 * @throws std::invalid_argument 空系数、空/非法shape、axis、zi秩或广播shape不合法。
 * @details 默认stream的custom kernel以FP32执行完整卷积、初态相加和末态提取；系数、输入、
 * 可选初态及line-offset表构成一次性workspace，不使用FFT/RAND或device FP64。正式封装覆盖
 * cuSignal的多维axis、zi广播和动态单返回/双返回结构。
 */
template <typename T>
void firfilter_device(
    const DeviceArray<T>& b, const DeviceArray<T>& x,
    FirfilterDeviceResult& out, const FirfilterOptions& options = {},
    const DeviceArray<T>* zi = nullptr);

/**
 * @brief 一维无初态 FP32 FIR 的预分配 resident 接口。
 * @details 系数、输入和输出均已驻留 device；稳定调用只启动一个 FIR kernel，不分配、
 * 不转换、不执行 H2D/D2H。完整多维、zi 和五类型公开语义仍由 `firfilter_device` 提供。
 */
void firfilter_fp32_resident_device(
    const DeviceArray<float>& b, const DeviceArray<float>& x,
    DeviceArray<float>& y);
/**
 * @brief 在GPU上按cuSignal完整pad方法执行前向-反向FIR零相位过滤。
 * @tparam T FP32、FP16、INT32、INT16或INT8真实业务输入/系数类型。
 * @param b device非空一维FIR系数`[Nb]`。
 * @param x device连续row-major多维输入；元素数必须等于options.shape乘积。
 * @param out dtype variant及原shape；INT32返回FP64存储，其余四类型返回FP32。
 * @param options axis、odd/even/constant/none、可选padlen、method和irlen完整参数。
 * @throws std::invalid_argument shape、axis、padtype、padlen或method非法；gust按固定版拒绝。
 * @details 默认stream上以FP32完成类型保持padding、稳态zi、双向direct FIR、reverse和crop；
 * 小系数融合路径直接写入最终输出且不创建中间workspace；INT32以整数位模式写出FP64存储，
 * 不执行FP64算术，也不发生Host拓宽往返。大系数通用路径仍使用一次性workspace及兼容层
 * Host收尾。无device FP64算术、FFT或RAND。irlen在pad方法下与cuSignal一致被忽略。
 */
template <typename T>
void firfilter2_device(
    const DeviceArray<T>& b, const DeviceArray<T>& x,
    Firfilter2DeviceResult& out, const Firfilter2Options& options = {});
/**
 * @brief 在GPU上对实信号或复信号执行cuSignal负向frequency shift。
 * @tparam Input FP32、FP16、INT32、INT16或INT8，或以该类型为分量的TypedComplex。
 * @param x 调用者持有的任意逻辑 rank、连续 row-major device 输入，实输入虚部按0处理。
 * @param out 调用者预分配的同 shape 展平 device ComplexFP64 存储。
 * @param freq 频移，单位与 fs 相同，可正可负。
 * @param fs 采样率；与固定源码一致，不强制为正，零和非有限值按浮点规则传播。
 * @throws std::invalid_argument 仅在输出size不匹配时抛出。
 * @details 默认stream的custom kernel以FP32完成sin/cos和复乘，ComplexFloat临时输出构成
 * workspace；随后D2H并在Host拓宽为ComplexFP64，再H2D写入out。无device FP64计算、
 * FFT或RAND；完整封装对齐cuSignal显式complex128输出。
 */
template <typename Input>
void freq_shift_device(
    const DeviceArray<Input>& x, DeviceArray<ComplexDouble>& out,
    double freq, double fs);
/**
 * @brief 在GPU上按cuSignal完整N/axis语义计算一维Hilbert analytic signal。
 * @tparam T FP32、FP16、INT32、INT16或INT8真实实数业务输入。
 * @param x 调用者持有的连续row-major device输入，元素数须等于options.shape乘积。
 * @param out 封装重建的dtype variant及输出shape；整数为ComplexFP64，其余为ComplexFP32。
 * @param options 输入shape、可选正N和axis；shape为空表示一维，N为空取所选轴长度。
 * @throws std::invalid_argument shape、axis、N、索引容量或输出容量非法。
 * @details 默认stream上pack所选axis、以ComplexFP32执行batched FFTInterface、parity mask、
 * IFFT和unpack；scratch与plan构成一次性workspace。整数结果D2H后在Host拓宽并可选H2D，
 * 无device FP64或RAND；输入不修改且不允许输出与输入原地重叠。
 */
template <typename T>
void hilbert_device(
    const DeviceArray<T>& x, HilbertDeviceResult& out,
    const HilbertOptions& options = {});
/**
 * @brief 在GPU上复现固定cuSignal版本的二维Hilbert及其方形mask行为。
 * @tparam T FP32、FP16、INT32、INT16或INT8真实实数业务输入。
 * @param x 调用者持有的连续row-major二维device输入。
 * @param out 封装重建的dtype variant及实际方形输出shape。
 * @param options 必填二维输入shape；fft_shape为空、单值或双值对应cuSignal三种N形式。
 * @throws std::invalid_argument shape不足两行、N非正、方形mask无法广播或容量溢出。
 * @details 默认stream上以ComplexFP32执行所选FFTInterface2D后端、固定版本方形mask和IFFT；
 * N0=1时先按cuSignal广播首行频谱。scratch及plan构成一次性workspace。整数结果D2H后
 * Host拓宽并可选H2D，无device FP64或RAND；输入不修改且输出不与输入重叠。
 */
template <typename T>
void hilbert2_device(
    const DeviceArray<T>& x, Hilbert2DeviceResult& out,
    const Hilbert2Options& options);
/**
 * @brief 在GPU上按固定cuSignal语义求`lfilter`稳态初始条件。
 * @tparam T FP32、FP16、INT32、INT16或INT8一维实系数；FP16为批准扩展。
 * @param b device numerator系数，允许为空并在公共长度内补零。
 * @param a 非空device denominator系数；Host只回读它以删除前导零并确定shape。
 * @param out 函数重建的单个FP64 device存储数组，长度为`max(len(trim(a)),len(b))-1`。
 * @throws std::invalid_argument a为空、公共长度小于2或索引/workspace容量溢出；
 * std::runtime_error FP32线性系统被判定为奇异。
 * @details 默认stream上将有效系数转换成FP32，在全局scratch workspace中执行
 * pivot/elimination/back-substitution；结果D2H后仅在Host拓宽并可选H2D为FP64。
 * 无device FP64、科学库或RAND；输入不修改，cuSignal的归一化、补零和单数组返回不变。
 */
template <typename T>
void lfilter_zi_device(
    const DeviceArray<T>& b, const DeviceArray<T>& a,
    LfilterZiDeviceResult& out);
/**
 * @brief 在GPU上按固定cuSignal语义执行多维级联二阶节IIR滤波。
 * @tparam T `sos`、`x`和可选`zi`共同采用的FP32、FP16、INT32、INT16或INT8真实业务dtype。
 * FP32是cuSignal原生路径；FP16和整数为已批准的FP32合法路径扩展。
 * @param sos 调用者持有的连续device系数矩阵`[sections,6]`，列顺序为
 * `b0,b1,b2,a0,a1,a2`；每个`a0`必须精确等于1。
 * @param sections 正section数，公开限制不超过512，且不能超过所选axis样本数。
 * @param x 调用者持有的连续row-major device输入；元素数必须等于`options.shape`乘积。
 * @param out 函数重建的FP32结果；`y_shape`等于输入shape，传入`zi`时额外填充`zf`。
 * @param options 输入shape、axis及`zi_shape`；shape为空表示一维，`zi_shape`必须精确等于
 * `[sections] + x.shape`并把所选axis长度替换为2，未传`zi`时必须为空。
 * @param zi 可选device初态；非空指针保持cuSignal的双返回结构，且输入不会被修改。
 * @throws std::invalid_argument shape乘积、axis、section数、SOS宽度、`a0`、样本数或zi非法。
 * @details 默认stream上每条信号由一个GPU线程按Direct Form II Transposed顺序递推；状态为
 * 动态FP32全局workspace，不再使用固定32节局部数组。为验证`a0`会显式D2H读取系数；其余主计算
 * 均在GPU完成。无device FP64、FFT或RAND；函数返回前同步默认stream，`y/zf`随后可读取；
 * 输出分配、D2H校验和该同步均属于正式封装性能范围。
 */
template <typename T>
void sosfilt_device(
    const DeviceArray<T>& sos, int sections, const DeviceArray<T>& x,
    SosfiltDeviceResult& out, const SosfiltOptions& options = {},
    const DeviceArray<T>* zi = nullptr);
/**
 * @brief 在GPU上按固定cuSignal语义执行N维局部Wiener滤波。
 * @tparam T FP32、FP16、INT32、INT16或INT8真实业务输入；五种均为cuSignal原生输入。
 * @param x 调用者持有的连续row-major device输入，元素数必须等于`options.shape`乘积。
 * @param out 函数重建的单个FP64 device存储数组及原shape。
 * @param options 输入shape；默认每维窗口3、可选标量窗口或逐维window_shape；noise为空时
 * 取全部局部方差的均值，否则使用给定Host FP64标量，负值及非有限值按浮点规则传播。
 * @throws std::invalid_argument shape、窗口秩、窗口元素、互斥窗口参数或索引容量非法。
 * @details 默认stream上以FP32执行完整N维零填充same局部和、输入dtype平方和、方差、自动noise
 * reduction及严格`variance < noise`后处理；mean/variance/noise及shape元数据构成一次性workspace。
 * 结果D2H后仅在Host拓宽为FP64并H2D形成正式输出；无device FP64、FFT或RAND。Host收尾同步，
 * 函数返回后输出可读取，全部分配、传输、Host转换和同步计入正式封装性能范围。
 */
template <typename T>
void wiener_device(
    const DeviceArray<T>& x, WienerDeviceResult& out,
    const WienerOptions& options = {});
/**
 * @brief 在GPU上按固定cuSignal FIR语义执行N维抽取。
 * @tparam T FP32、FP16、INT32、INT16或INT8真实业务输入；自定义系数必须与输入同dtype。
 * @param x 调用者持有的连续row-major device输入，元素数等于`options.shape`乘积。
 * @param q 正整数抽取因子。
 * @param out 函数重建的dtype variant结果及输出shape；选定轴长度为`ceil(N/q)`。
 * @param options 任意axis、默认`20*q`阶或显式非负filter_order及zero_phase开关。
 * @param coefficients 可选非空一维同dtype FIR；提供时不得同时给filter_order。
 * @throws std::invalid_argument q、shape、axis、阶数、系数或索引容量非法；默认设计在q=1
 * 时按固定firwin截止频率规则拒绝Nyquist端点。
 * @details 无自定义系数时Host按Hamming firwin设计FP64 taps并转FP32 H2D；自定义系数由
 * device转换为FP32。默认stream上的axis-aware upfirdn kernel执行因果路径，zero_phase路径
 * 额外按cuSignal half-length规则前补零并从中心偏移取样。所有device计算均为FP32；默认
 * 系数及INT32自定义系数结果D2H后在Host拓宽为FP64并可选H2D存储，无device FP64。
 * workspace、分配、传输、Host设计/拓宽和同步均属于正式封装性能范围。
 */
template <typename T>
void decimate_device(
    const DeviceArray<T>& x, int q, DecimateDeviceResult& out,
    const DecimateOptions& options = {},
    const DeviceArray<T>* coefficients = nullptr);
/**
 * @brief 在GPU上按固定cuSignal Fourier语义重采样N维实信号。
 * @tparam T FP32、FP16、INT32、INT16或INT8真实业务输入；可选t使用同一dtype。
 * @param x 调用者持有的连续row-major device输入。
 * @param output_sample_count 正目标长度num。
 * @param out 函数重建的y dtype variant、可选FP64时间数组、动态返回标志及y shape。
 * @param options 输入shape、任意axis、time/frequency domain及可选最终FFT顺序频谱窗；
 * `frequency_window`等价于固定源码完成callable/array/get_window和ifftshift后的长度Nx数组。
 * @param sample_positions 可选device同dtype时间坐标，至少两个元素；固定语义只读取前两项。
 * @throws std::invalid_argument 空选定轴、非正num、domain、shape/axis/window/t或索引容量非法；
 * frequency-domain整数偶数频谱上采样保持固定CuPy原位浮点结果不能same-kind写回整数Y的异常。
 * @details 默认stream上将任意axis打包为连续批次；time domain先做Nx点ComplexFP32 FFT，
 * frequency domain按输入dtype完成窗乘后提升；随后执行正负频谱复制、偶数Nyquist合并/拆分、
 * num点逆FFT和`num/Nx`缩放。FP16/FP32返回FP32，整数结果D2H后仅在Host拓宽FP64；
 * 可选时间公式也只在Host计算并H2D存储，无device FP64或RAND。FFT plan、频谱、窗、分配、
 * 传输、Host转换和同步均属于正式封装workspace与性能范围。
 */
template <typename T>
void resample_device(
    const DeviceArray<T>& x, int output_sample_count,
    ResampleDeviceResult& out, const ResampleOptions& options = {},
    const DeviceArray<T>* sample_positions = nullptr);
/**
 * @brief 在GPU上按固定cuSignal语义执行causal polyphase upfirdn。
 * @tparam T FP32、FP16、INT32、INT16或INT8真实业务输入；h与x使用同一dtype。
 * @param h 调用者持有的device一维非空FIR coefficients。
 * @param x 调用者持有的连续row-major 1D或2D device输入。
 * @param up 正整数上采样因子，不进行gcd约分。
 * @param down 正整数下采样因子。
 * @param out 函数重建的动态dtype结果和输出shape；仅选定axis长度改变。
 * @param options 输入shape和正/负axis；空shape按一维`{x.size()}`解释。
 * @throws std::invalid_argument taps、factor、rank、axis、shape、负输出长度或索引容量非法。
 * @details 输出长度严格为`floor((((N-1)*up+Nh)-1)/down)+1`，保留空轴固定行为；
 * 默认stream上先将h/x转换为FP32，再用stride-aware direct polyphase kernel执行causal
 * constant-zero extension。FP32/FP16/INT16/INT8返回FP32；INT32结果D2H后仅在Host拓宽为
 * FP64并可选H2D。无device FP64、FFT或RAND；转换、临时数组、输出、同步和Host收尾均属于
 * 正式封装workspace与性能范围。
 */
template <typename T>
void upfirdn_device(
    const DeviceArray<T>& h, const DeviceArray<T>& x,
    int up, int down, UpfirdnDeviceResult& out,
    const UpfirdnOptions& options = {});
/**
 * @brief 在GPU上按固定cuSignal语义执行有理数polyphase重采样。
 * @tparam T FP32、FP16、INT32、INT16或INT8真实业务输入；自定义系数必须使用同一dtype。
 * @param x 调用者持有的连续row-major device输入。
 * @param up 正整数上采样因子，与down先按gcd约分。
 * @param down 正整数下采样因子。
 * @param out 函数重建的动态dtype结果及输出shape；非恒等时选定轴长度为`ceil(N*up/down)`。
 * @param options 输入shape、axis、默认Kaiser(5)、Hamming或等价显式设计窗，以及兼容gpupath标志。
 * @param coefficients 可选一维同dtype原始FIR；允许为空，提供时完全覆盖options中的设计窗。
 * @throws std::invalid_argument factor、shape、非恒等rank/axis、设计窗或索引/workspace容量非法。
 * @details 约分后恒等比例在axis/window校验前D2D复制并保持T。其他比例在Host设计FP64 FIR，
 * 或在device按CuPy最小标量提升及整数环绕规则计算`up*coefficients`，再由默认stream上的
 * axis-aware FP32 upfirdn kernel执行零边界居中过滤。默认设计恒返回FP64；自定义结果按T和
 * reduced up动态选择FP32/FP64，FP64均D2H后仅在Host拓宽并可选H2D。无device FP64、FFT或RAND；
 * filter设计、taps、输出、传输、同步和Host收尾均属于正式封装workspace与性能范围。
 */
template <typename T>
void resample_poly_device(
    const DeviceArray<T>& x, int up, int down,
    ResamplePolyDeviceResult& out,
    const ResamplePolyOptions& options = {},
    const DeviceArray<T>* coefficients = nullptr);

/**
 * @brief 在CPU上独立实现cuSignal polyphase channelizer reference。
 * @tparam T x与h共同采用的五种真实业务dtype；结果固定ComplexFP32。
 * @param x host一维输入；仅完整的n_chans分组参与计算。
 * @param h host一维taps；每相位使用`floor(len(h)/n_chans)`项，允许为空。
 * @param n_chans 正channel数。
 * @return channel-major值和`{n_chans,floor(N/n_chans)}` shape结构。
 * @throws std::invalid_argument n_chans非正、索引容量溢出或每相位taps超过32。
 * @details CPU以独立double多相累加和正号direct DFT生成ComplexFP32边界，不调用GPU、
 * FFTInterface或科学库；寄存器历史和输出vector是Host workspace，无stream。
 */
template <typename T>
ChannelizePolyCpuResult channelize_poly_typed_cpu(
    const std::vector<T>& x, const std::vector<T>& h, int n_chans);

template <typename T>
ChannelizePolyCpuResult channelize_poly_typed_cpu(
    const std::vector<T>& x, const std::vector<T>& h, int n_chans,
    const std::vector<int>& input_shape,
    const std::vector<int>& filter_shape);

/**
 * @brief 在CPU上独立实现与GPU正式入口相同的多维detrend reference。
 * @tparam T 五种真实业务输入类型；输出dtype随输入和mode遵循cuSignal/CuPy提升规则。
 * @param x host连续row-major输入；元素数必须等于options.shape乘积。
 * @param options 参数语义与detrend_device一致。
 * @return host dtype variant及与输入相同的shape；只有dtype对应的vector非空。
 * @throws std::invalid_argument shape、axis、空linear轴或breakpoint非法。
 * @details CPU使用独立double最小二乘公式；FP32/FP16 constant按对应边界舍入。
 * 不调用device、kernel或GPU输出，无stream；临时索引与分段统计量是Host workspace。
 */
template <typename T>
DetrendCpuResult detrend_typed_cpu(
    std::vector<T>& x, const DetrendOptions& options = {});

/**
 * @brief 在CPU上独立实现完整cuSignal firfilter reference。
 * @tparam T 五种真实业务输入/系数类型；FP16和整数扩展先进入FP32合法路径。
 * @param b host非空一维FIR系数。
 * @param x host连续row-major输入；元素数必须等于options.shape乘积。
 * @param options shape、axis和zi_shape，与GPU正式入口一致。
 * @param zi 可选Host初态；提供时返回y和zf，否则只返回y。
 * @return FP32 y、可选FP32 zf、对应shape及动态返回标记。
 * @throws std::invalid_argument 空系数、空/非法shape、axis或zi广播shape不合法。
 * @details 使用独立double卷积和初态公式后收敛到FP32边界；不调用device、GPU输出或科学库，
 * 无stream，输出和line映射vector是Host workspace，并覆盖cuSignal多维axis/zi语义。
 */
template <typename T>
FirfilterCpuResult firfilter_typed_cpu(
    const std::vector<T>& b, const std::vector<T>& x,
    const FirfilterOptions& options = {}, const std::vector<T>* zi = nullptr);

/**
 * @brief 在CPU上独立实现cuSignal firfilter2完整pad方法reference。
 * @tparam T 五种真实业务输入/系数类型；INT32输出FP64，其余输出FP32。
 * @param b host非空一维FIR系数。
 * @param x host连续row-major多维输入。
 * @param options shape、axis、padtype、padlen、method和irlen，与GPU入口一致。
 * @return 原shape的FP32/FP64 dtype variant。
 * @throws std::invalid_argument shape、axis、padding或method非法；gust固定版不支持。
 * @details 独立Host公式保持输入dtype的padding舍入/整数环绕，再以double计算稳态zi和双向FIR；
 * 不调用device、GPU输出或科学库，无stream，line临时vector是Host workspace并覆盖cuSignal语义。
 */
template <typename T>
Firfilter2CpuResult firfilter2_typed_cpu(
    const std::vector<T>& b, const std::vector<T>& x,
    const Firfilter2Options& options = {});

/**
 * @brief 在CPU上对实信号或复信号执行frequency shift reference。
 * @tparam Input FP32、FP16、INT32、INT16或INT8，或以该类型为分量的TypedComplex。
 * @param x host 任意逻辑 rank 的连续 row-major 输入；shape 由调用者持有，实输入虚部按0处理。
 * @param freq 频移，可正可负，单位与 fs 相同。
 * @param fs 采样率，不额外限制符号或有限性。
 * @return 与 `x` 同 shape、展平存储的 host ComplexFP64；数学式为`x[i]*exp(-j*2*pi*freq*i/fs)`。
 * @throws std::bad_alloc 仅Host输出分配失败；参数不额外抛出异常。
 * @details 独立double数学reference，不调用device或GPU结果；无stream/FFT/RAND，输出vector
 * 是唯一Host workspace，并对齐cuSignal explicit complex128边界。
 */
template <typename Input>
std::vector<ComplexDouble> freq_shift_typed_cpu(
    const std::vector<Input>& x, double freq, double fs);

/**
 * @brief 在CPU上独立实现cuSignal Hilbert N/axis reference。
 * @tparam T 五种真实实数业务输入；整数输出ComplexFP64，其余ComplexFP32。
 * @param x host连续row-major输入。
 * @param options shape、正N和axis，与GPU正式入口完全一致。
 * @return dtype variant及所选轴替换为N后的shape。
 * @details CPU以独立double direct DFT、parity mask和IDFT计算，不调用device或科学库；
 * line频谱与输出vector是Host workspace，无stream，最终按cuSignal/CuPy dtype边界存储。
 * @throws std::invalid_argument shape、axis、N或索引容量非法。
 */
template <typename T>
HilbertCpuResult hilbert_typed_cpu(
    const std::vector<T>& x, const HilbertOptions& options = {});

/**
 * @brief 在CPU上独立复现固定cuSignal版本hilbert2 reference。
 * @tparam T 五种真实实数业务输入；整数输出ComplexFP64，其余ComplexFP32。
 * @param x host连续row-major输入。
 * @param options 二维shape及N形式，与GPU正式入口完全一致。
 * @return dtype variant及固定源码实际产生的方形shape。
 * @throws std::invalid_argument 输入行数、shape、N、广播条件或容量非法。
 * @details CPU以独立double 2D DFT、方形parity mask和IDFT计算，不调用device或科学库；
 * 频谱和输出vector是Host workspace，无stream，并保留N0=1广播行为与cuSignal dtype边界。
 */
template <typename T>
Hilbert2CpuResult hilbert2_typed_cpu(
    const std::vector<T>& x, const Hilbert2Options& options);
/**
 * @brief 在CPU上独立复现固定cuSignal版本`lfilter_zi` reference。
 * @tparam T 五种一维实系数输入；FP16是批准扩展，正式输出统一为FP64。
 * @param b Host numerator系数，允许为空。
 * @param a 非空Host denominator系数；按cuSignal循环删除前导零并以首项归一化。
 * @return 单个Host FP64状态数组，长度为补零后的公共系数长度减一。
 * @throws std::invalid_argument a为空、公共长度小于2或容量溢出；
 * std::runtime_error 线性系统奇异。
 * @details CPU以独立double matrix workspace和pivot求解，不调用device或科学库；无stream，
 * 仅作为FP32 GPU结果reference，并保留cuSignal的补零、shape和返回结构。
 */
template <typename T>
std::vector<double> lfilter_zi_typed_cpu(
    const std::vector<T>& b, const std::vector<T>& a);

/**
 * @brief `sosfilt_device`逐参数对应的独立C++ CPU reference。
 * @tparam T 与GPU正式入口相同的五种真实业务dtype；所有值先进入FP32合法输入边界。
 * @param sos Host端连续`[sections,6]`系数，`a0`必须精确等于1。
 * @param sections、x、options、zi 与GPU入口具有相同shape、axis、状态和异常语义。
 * @return FP32 `y`，以及仅在传入`zi`时存在的FP32 `zf`和对应shape。
 * @details CPU使用独立的double Host workspace和状态递推作为数值对照，最终边界收敛为FP32；
 * 不调用GPU kernel、不读取GPU结果，也不复用GPU数学核心。无stream，输入、系数和状态均不被
 * 修改，并保持cuSignal的axis、zi/zf、shape和动态返回结构。
 */
template <typename T>
SosfiltCpuResult sosfilt_typed_cpu(
    const std::vector<T>& sos, int sections, const std::vector<T>& x,
    const SosfiltOptions& options = {}, const std::vector<T>* zi = nullptr);

/**
 * @brief `wiener_device`逐参数对应的独立cuSignal C++ CPU reference。
 * @tparam T 与GPU入口相同的五种真实业务dtype，输出边界固定为FP64。
 * @param x Host端连续row-major输入。
 * @param options 与GPU入口相同的shape、默认/标量/逐维窗口和可选noise。
 * @return 单个Host FP64同shape结果结构。
 * @throws std::invalid_argument 与GPU入口相同的shape、窗口和容量异常。
 * @details CPU以独立double Host workspace执行五类型（含FP32）N维零填充统计、自动noise和后处理，不调用GPU、
 * 不读取GPU结果或复用device数学核心；无stream，并保留整数/FP16平方、NaN和严格比较语义。
 */
template <typename T>
WienerCpuResult wiener_typed_cpu(
    const std::vector<T>& x, const WienerOptions& options = {});

/**
 * @brief `decimate_device`逐参数对应的独立cuSignal C++ CPU reference。
 * @tparam T 与GPU入口相同的五种真实业务dtype；自定义系数使用同一T。
 * @param x Host端连续row-major输入。
 * @param q 正整数抽取因子。
 * @param options 与GPU入口相同的shape、axis、filter_order和zero_phase。
 * @param coefficients 可选非空一维同dtype自定义FIR。
 * @return 单个同shape-rank dtype variant；默认设计恒为FP64，INT32自定义也为FP64，
 * 其余四种同dtype自定义系数为FP32。
 * @throws std::invalid_argument 与GPU入口相同的参数异常。
 * @details CPU独立执行Hamming firwin、axis-aware因果/居中upfirdn和裁剪，不调用GPU、
 * 不读取GPU结果或复用device kernel；FP32输出用FP32累加，FP64输出用Host double累加；
 * 无stream，设计系数与临时累加数据构成Host workspace。
 */
template <typename T>
DecimateCpuResult decimate_typed_cpu(
    const std::vector<T>& x, int q,
    const DecimateOptions& options = {},
    const std::vector<T>* coefficients = nullptr);

/**
 * @brief `resample_device`逐参数对应的独立cuSignal C++ CPU reference。
 * @tparam T 与GPU入口相同的五种实业务dtype；可选t使用同一T。
 * @param x Host端连续row-major输入。
 * @param output_sample_count 正目标长度num。
 * @param options 与GPU入口相同的shape、axis、domain和预计算频谱窗。
 * @param sample_positions 可选Host同dtype时间坐标，至少两个元素。
 * @return y及可选t动态结果；FP16/FP32 y为FP32，三种整数y与全部t为FP64。
 * @throws std::invalid_argument 与GPU入口相同的参数异常。
 * @details CPU以独立Host direct DFT/IDFT实现完整频谱复制和Nyquist语义，不调用GPU、
 * 不读取GPU结果或复用device FFT核心；无stream，频谱和逐轴缓冲构成Host workspace，
 * FP32/FP64边界及动态返回与cuSignal契约一致。
 */
template <typename T>
ResampleCpuResult resample_typed_cpu(
    const std::vector<T>& x, int output_sample_count,
    const ResampleOptions& options = {},
    const std::vector<T>* sample_positions = nullptr);

/**
 * @brief `upfirdn_device`逐参数对应的独立cuSignal C++ CPU reference。
 * @tparam T 与GPU入口相同的五种真实业务dtype，h与x同型。
 * @param h Host一维非空FIR coefficients。
 * @param x Host连续row-major 1D或2D输入。
 * @param up 正整数上采样因子。
 * @param down 正整数下采样因子。
 * @param options 与GPU入口相同的shape和axis。
 * @return 单数组动态结果；INT32为FP64，其余四种输入为FP32。
 * @throws std::invalid_argument 与GPU入口相同的参数异常。
 * @details 独立执行精确输出长度、axis-aware causal zero-extension convolution；非INT32按
 * FP32转换/累加，INT32按固定`result_type(INT32,INT32,FP32)=FP64`在Host double累加。
 * 不调用GPU或读取GPU结果，无stream/FFT/RAND，Host临时数组是唯一workspace。
 */
template <typename T>
UpfirdnCpuResult upfirdn_typed_cpu(
    const std::vector<T>& h, const std::vector<T>& x,
    int up, int down, const UpfirdnOptions& options = {});

/**
 * @brief `resample_poly_device`逐参数对应的独立cuSignal C++ CPU reference。
 * @tparam T 与GPU入口相同的五种实业务dtype，自定义系数与x同型。
 * @param x Host连续row-major输入。
 * @param up 正整数上采样因子。
 * @param down 正整数下采样因子。
 * @param options 与GPU入口相同的shape、axis、设计窗和gpupath兼容参数。
 * @param coefficients 可选一维同dtype原始FIR，允许为空并覆盖设计窗。
 * @return 单数组动态结果；恒等时保持T，默认设计为FP64，自定义非恒等为FP32或FP64。
 * @throws std::invalid_argument 与GPU入口相同的参数异常。
 * @details CPU独立执行Kaiser/Hamming/显式窗FIR设计、CuPy标量提升、整数环绕、axis-aware
 * 零边界upfirdn及中心裁剪，不调用GPU或读取GPU结果。FP32结果按FP32累加，FP64结果按Host
 * double累加；无stream，设计窗、taps和逐轴输出构成Host workspace。
 */
template <typename T>
ResamplePolyCpuResult resample_poly_typed_cpu(
    const std::vector<T>& x, int up, int down,
    const ResamplePolyOptions& options = {},
    const std::vector<T>* coefficients = nullptr);

}  // namespace cusignal
