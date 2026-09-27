#pragma once

#include <cusignal/operators/filtering/filtering_typed.h>

#include <cstdint>
#include <vector>

namespace cusignal {

enum class AmbgfunCut { two_dimensional, delay, doppler };
enum class AmbgfunOutputDtype { fp32, fp64 };

struct AmbgfunOptions {
    double fs = 1.0;
    double prf = 1.0;
    AmbgfunCut cut = AmbgfunCut::two_dimensional;
    double cut_value = 0.0;
    // Empty preserves the ordinary one-dimensional public call.  For the
    // fixed 23.08 default 2-D cut, higher-rank inputs are accepted but
    // len(x)/len(y), i.e. only shape[0], controls the raw-kernel prefix.
    std::vector<int> input_shape;
    std::vector<int> reference_shape;
};

struct AmbgfunCpuResult {
    AmbgfunOutputDtype dtype{AmbgfunOutputDtype::fp32};
    std::vector<float> fp32;
    std::vector<double> fp64;
    std::vector<int> shape;
};

struct AmbgfunDeviceResult {
    AmbgfunOutputDtype dtype{AmbgfunOutputDtype::fp32};
    DeviceArray<float> fp32;
    DeviceArray<double> fp64;
    std::vector<int> shape;
};

enum class CfarDetection : std::uint8_t { no = 0, yes = 1 };

struct CaCfarOptions {
    std::vector<int> guard_cells;
    std::vector<int> reference_cells;
    double pfa = 1.0e-3;
};

struct CaCfarCpuResult {
    std::vector<float> threshold;
    std::vector<CfarDetection> detections;
    std::vector<int> shape;
};

struct CaCfarDeviceResult {
    DeviceArray<float> threshold;
    DeviceArray<CfarDetection> detections;
    std::vector<int> shape;
};

enum class RadarWindowKind {
    none,
    hann,
    hamming,
    explicit_fp32,
    explicit_fp64
};

struct PulseCompressionOptions {
    int num_pulses{0};
    int samples_per_pulse{0};
    bool normalize{false};
    int nfft{-1};
    RadarWindowKind window{RadarWindowKind::none};
};

struct PulseDopplerOptions {
    int num_pulses{0};
    int samples_per_pulse{0};
    int nfft{-1};
    RadarWindowKind window{RadarWindowKind::none};
};

struct RadarComplexCpuResult {
    std::vector<ComplexFloat> fp32;
    std::vector<ComplexDouble> fp64;
    std::vector<int> shape;
    bool is_fp64{false};
};

struct RadarComplexDeviceResult {
    DeviceArray<ComplexFloat> fp32;
    DeviceArray<ComplexDouble> fp64;
    std::vector<int> shape;
    bool is_fp64{false};
};

/**
 * @brief 独立CPU reference，完整对应固定版`cusignal.ambgfun`三种cut。
 * @tparam T `TypedComplex<T>`分量为真实FP32、FP16、INT32、INT16或INT8；FP16及整数复
 * 分量是已批准扩展，公开输入不预先改型。
 * @param x 非空Host连续row-major复信号。`input_shape`为空时按一维解释；默认2d cut允许
 * 任意非空rank，并按固定版`len(x)==shape[0]`只计算连续前缀，但归一化范数取完整数组。
 * @param options `fs/prf/cut/cut_value`；固定版`prf`不参与返回值；delay/doppler要求
 * `fs!=0`。
 * @param y 可选非空Host参考；空指针表示`y=x`，允许与x不等长。提供y时其逻辑shape由
 * `reference_shape`描述；未提供y时`reference_shape`必须为空。
 * @return 一个动态dtype数组：2d时FP32/FP16输入返回FP32、整数输入返回FP64，shape为
 * `[Nx+Ny-1,Nfreq]`；delay/doppler五类型均返回FP64，shape分别为`[Nfreq]`和
 * `[Nx+Ny-1]`。`Nfreq`是不小于`Nx+Ny-1`的最小2次幂。
 * @throws std::invalid_argument 空输入、shape/rank非法、非法cut枚举、长度溢出或cut需要
 * 但fs为0。固定版delay/doppler没有一致的高rank数组契约，因此shape rank>1只允许2d cut。
 * @details 2d按全部Ny样点构造`y*conj(x_shifted)`并零填充，修复固定cuSignal在不等长
 * y上的越界/截断缺陷；零范数按CuPy除法传播NaN，不返回项目自定义全零结果；CPU
 * reference不创建GPU stream或device workspace。
 */
template <class T>
AmbgfunCpuResult ambgfun_typed_cpu(
    const std::vector<TypedComplex<T>>& x,
    const AmbgfunOptions& options = {},
    const std::vector<TypedComplex<T>>* y = nullptr);

/**
 * @brief 完整五类型GPU正式入口，参数、动态dtype、shape和异常与CPU reference一致。
 * @tparam T FP32、FP16、INT32、INT16或INT8复数分量。
 * @param x 调用者持有的非空device输入；shape和首维退化规则与CPU reference一致。
 * @param output 返回对象；FP32或FP64中仅与`dtype`对应的数组非空。
 * @param options 完整`fs/prf/cut/cut_value`。
 * @param y 可选device参考；空指针表示`y=x`，允许不等长。
 * @details 默认stream上所有归一化、相位、FFT和幅值均在GPU以FP32/ComplexFP32完成；
 * 内部workspace随本次调用创建；FP64只作为
 * cuSignal对应的公开存储dtype，由D2H、Host拓宽和可见H2D收尾产生，device不执行FP64
 * 运算。函数内部创建本次FFT临时量和plan；FP64收尾及同步均计入完整封装性能。
 */
template <class T>
void ambgfun_device(
    const DeviceArray<TypedComplex<T>>& x,
    AmbgfunDeviceResult& output,
    const AmbgfunOptions& options = {},
    const DeviceArray<TypedComplex<T>>* y = nullptr);

/**
 * @brief 固定版`cusignal.ca_cfar`的一维/二维独立CPU reference。
 * @tparam T 真实FP32、FP16、INT32、INT16或INT8业务输入。
 * @param x row-major Host输入；shape为空时必须含一个标量，否则size必须等于shape乘积。
 * @param shape rank必须为0、1或2；输出保持相同shape。
 * @param options rank一时guard/reference各含一个值，rank二时各含两个值；guard非负、
 * reference至少为1，`pfa`位于`(0,1]`。rank零时这些计算参数不参与返回值。
 * @return `(threshold,detections)`的C++结构；threshold固定FP32，detections是逐元素
 * BOOL逻辑值，边界无完整参考窗时threshold为0。
 * @throws std::invalid_argument rank、shape、参数长度/范围、输入size或可表示容量非法。
 * @details reference按固定版顺序先做axis0、再做axis1的FP32累计和，并以相同矩形差分
 * 公式生成threshold；INT32检测直接比较原始整数与FP32阈值，不经FP32改型丢失精度；
 * CPU reference不创建GPU stream或device workspace，输出语义保持cuSignal一致。
 */
template <class T>
CaCfarCpuResult ca_cfar_typed_cpu(
    const std::vector<T>& x,
    const std::vector<int>& shape,
    const CaCfarOptions& options);

/**
 * @brief 五类型一维/二维CA-CFAR正式GPU完整封装。
 * @tparam T FP32、FP16、INT32、INT16或INT8真实输入。
 * @param x 调用者持有的row-major device输入，shape/参数规则与CPU reference一致。
 * @return 自有精确shape的FP32 threshold和一字节`CfarDetection`逻辑BOOL数组。
 * @throws std::invalid_argument rank、shape、参数、workspace可表示容量或输入size非法。
 * @details Host只计算FP32 alpha并组织结果；GPU template kernel以FP32累加参考单元，
 * 不使用device FP64、科学库、隐式D2H或外部stream。内部按host thread复用只增不减的
 * FP32积分图workspace；首次调用或shape增长时分配，稳定shape的后续调用不重复分配，
 * 调用者无需提供额外workspace。返回后由调用者读取时同步。
 * @note `CfarDetection`是BOOL语义的连续一字节枚举；没有使用不可提供普通连续
 * `data()`的`std::vector<bool>`，也不再把cuSignal BOOL结果错误公开成INT32 mask。
 */
template <class T>
CaCfarDeviceResult ca_cfar_device(
    const DeviceArray<T>& x,
    const std::vector<int>& shape,
    const CaCfarOptions& options);

/**
 * @brief 使用调用者预分配输出的CA-CFAR重载。
 * @tparam T FP32、FP16、INT32、INT16或INT8真实输入。
 * @param x、shape、options 语义与完整`ca_cfar_device`入口相同。
 * @param threshold 必须与输入元素数完全相同的预分配FP32 device输出。
 * @param detections 必须与输入元素数完全相同的预分配BOOL语义device输出。
 * @details 不分配或转移公开输出所有权；稳定shape调用可复用两个输出buffer。内部积分图
 * workspace、默认stream、kernel顺序、FP32累计、边界和检测语义与完整入口完全相同。
 * 调用者不得在完成当前默认stream工作前并发改写输出。
 */
template <class T>
void ca_cfar_device(
    const DeviceArray<T>& x,
    const std::vector<int>& shape,
    const CaCfarOptions& options,
    DeviceArray<float>& threshold,
    DeviceArray<CfarDetection>& detections);

/**
 * @brief 在 CPU 上计算 CFAR threshold multiplier reference。
 * @tparam T FP32、FP16、INT32、INT16 或 INT8 的编码概率输入。
 * @param pfa 真实业务dtype概率值；正式有效域为`(0,1]`。
 * @param reference 总 reference 单元数，必须至少为1。
 * @return 单个 FP64 alpha，公式为 `reference*(p^(-1/reference)-1)`。
 * @throws std::invalid_argument reference非法、pfa非有限或不在`(0,1]`。
 * @details 无数组 shape、workspace、stream 或科学计算库。
 * @note 对齐cuSignal的Python标量公式与FP64标量返回；整数有效概率只有1，结果为0。
 */
template <class T>
double cfar_alpha_typed_cpu(T pfa, int reference);

/**
 * @brief 在 GPU 上计算单个 CFAR threshold multiplier。
 * @tparam T 五种真实业务概率输入类型；输出固定FP64。
 * @param pfa 真实业务dtype概率值，有效域为`(0,1]`。
 * @param reference 总 reference 单元数，必须至少为1。
 * @return 自有一个元素的device FP64标量存储。
 * @throws std::invalid_argument reference非法、pfa非有限或不在`(0,1]`。
 * @details 单线程template kernel只做FP32计算，随后D2H、Host拓宽FP64、H2D形成正式
 * 输出；默认stream执行且不需要workspace，device端不执行FP64算术，不接受接口外
 * `input_scale`量化扩展；返回dtype与cuSignal一致。
 */
template <class T>
DeviceArray<double> cfar_alpha_device(T pfa, int reference);

/**
 * @brief 固定版`pulse_compression`二维完整CPU reference。
 * @tparam T FP32、FP16、INT32、INT16或INT8复数分量。
 * @param x Host row-major二维脉冲输入。
 * @param pulse_template Host一维匹配模板。
 * @param options cuSignal shape、nfft与窗选项。
 * @param explicit_window 可选Host FP32显式窗系数。
 * @return cuSignal规则的ComplexFP32或ComplexFP64结果。
 * @throws std::invalid_argument shape、nfft、窗口或容量非法时抛出异常。
 * @details 保持`[num_pulses,samples_per_pulse]`输入、一维模板、窗后归一化、任意正nfft
 * 截断/补零和循环匹配滤波。无窗浮点返回ComplexFP32；整数或任意窗返回ComplexFP64；
 * CPU reference不创建GPU stream或device workspace。
 */
template <class T>
RadarComplexCpuResult pulse_compression_typed_cpu(
    const std::vector<TypedComplex<T>>& x,
    const std::vector<TypedComplex<T>>& pulse_template,
    const PulseCompressionOptions& options,
    const std::vector<float>* explicit_window = nullptr);

/**
 * @brief 五类型二维脉冲压缩正式GPU完整封装。
 * @tparam T FP32、FP16、INT32、INT16或INT8复数分量。
 * @param x device row-major二维脉冲输入。
 * @param pulse_template device一维匹配模板。
 * @param options cuSignal shape、nfft与窗选项。
 * @param explicit_window 可选device FP32显式窗系数。
 * @return cuSignal规则的ComplexFP32或Host拓宽ComplexFP64结果。
 * @throws std::invalid_argument shape、nfft、窗口或容量非法时抛出异常。
 * @details template kernel完成加窗、归一化和ComplexFP32 staging，dlfft执行批量FFT/IFFT；
 * 显式系数承载任意array/callable/named窗，ComplexFP64只经Host拓宽形成；临时workspace
 * 和plan属于本次调用，计算在默认stream执行。
 */
template <class T>
RadarComplexDeviceResult pulse_compression_device(
    const DeviceArray<TypedComplex<T>>& x,
    const DeviceArray<TypedComplex<T>>& pulse_template,
    const PulseCompressionOptions& options,
    const DeviceArray<float>* explicit_window = nullptr);

/**
 * @brief 固定版`pulse_doppler`完整CPU reference。
 * @tparam T FP32、FP16、INT32、INT16或INT8复数分量。
 * @param x Host row-major二维脉冲输入。
 * @param options cuSignal shape、nfft与窗选项。
 * @param explicit_window 可选Host FP32显式窗系数。
 * @return cuSignal规则的ComplexFP32或ComplexFP64结果。
 * @throws std::invalid_argument shape、nfft、窗口或容量非法时抛出异常。
 * @details 保持二维row-major、axis0 FFT、任意正nfft截断/补零和窗前乘法；显式系数承载
 * 任意array/callable/named窗。无窗浮点返回ComplexFP32；整数或任意窗返回ComplexFP64；
 * CPU reference不创建GPU stream或device workspace。
 */
template <class T>
RadarComplexCpuResult pulse_doppler_typed_cpu(
    const std::vector<TypedComplex<T>>& x,
    const PulseDopplerOptions& options,
    const std::vector<float>* explicit_window = nullptr);

/**
 * @brief 五类型Pulse-Doppler正式GPU完整封装。
 * @tparam T FP32、FP16、INT32、INT16或INT8复数分量。
 * @param x device row-major二维脉冲输入。
 * @param options cuSignal shape、nfft与窗选项。
 * @param explicit_window 可选device FP32显式窗系数。
 * @return cuSignal规则的ComplexFP32或Host拓宽ComplexFP64结果。
 * @throws std::invalid_argument shape、nfft、窗口或容量非法时抛出异常。
 * @details template kernel按axis0重排加窗，以samples为batch进入dlfft再转回`[nfft,samples]`；
 * ComplexFP64仅作Host拓宽后的正式存储，device端始终ComplexFP32计算；临时workspace与
 * plan属于本次调用并在默认stream执行。
 */
template <class T>
RadarComplexDeviceResult pulse_doppler_device(
    const DeviceArray<TypedComplex<T>>& x,
    const PulseDopplerOptions& options,
    const DeviceArray<float>* explicit_window = nullptr);

/**
 * @brief 将Pulse-Doppler的原生ComplexFP32计算结果写入预分配device输出的同名重载。
 * @param output 元素数必须为`nfft*samples_per_pulse`；稳定shape时由调用者复用。
 * @details 计算kernel、窗口、FFT和默认stream与完整封装相同；不执行仅用于公开动态dtype的
 * Host拓宽及可见回传。Task内部FP32流水线可直接把该输出交给后续GPU步骤。
 */
template <class T>
void pulse_doppler_device(
    const DeviceArray<TypedComplex<T>>& x,
    const PulseDopplerOptions& options,
    DeviceArray<ComplexFloat>& output,
    const DeviceArray<float>* explicit_window = nullptr);

}  // namespace cusignal
