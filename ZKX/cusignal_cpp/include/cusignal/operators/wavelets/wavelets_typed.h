#pragma once

#include <cusignal/operators/filtering/filtering_typed.h>

#include <functional>
#include <vector>

namespace cusignal {

/**
 * @brief 在 CPU 上生成 Morlet 小波，作为五类型 GPU reference。
 * @tparam T FP32、FP16、INT32、INT16或INT8参数类型；输出固定ComplexFP64。
 * @param n 输出点数，必须至少为 1。
 * @param w 无量纲中心频率，dtype T。
 * @param scale 时间尺度，dtype T，转换后必须大于 0。
 * @param complete true 时减去 admissibility correction，false 时使用 standard 形式。
 * @return host上一维`[n]`、固定`ComplexDouble`dtype的单个复数数组。
 * @throws std::invalid_argument `n<1` 或 `scale<=0`。
 * @details 无 workspace、stream、数据传输或科学计算库；与 cuSignal `morlet` 的复数结果
 * 对应，但本 C++ typed API 使用整数 n 和标量参数，不支持其他 axis/batch 形式。
 */
template<class T>
std::vector<ComplexDouble> morlet_typed_cpu(
    int n, T w, T scale, bool complete);

/**
 * @brief 在 GPU 上生成 Morlet 小波。
 * @tparam T 五种参数类型；device执行ComplexFP32计算，正式输出ComplexFP64。
 * @param n 输出点数，必须至少为 1。
 * @param w 无量纲中心频率。
 * @param scale 时间尺度，转换后必须大于 0。
 * @param complete 是否应用 admissibility correction。
 * @param out device连续输出`[n]`，由Host拓宽后作为`ComplexDouble`存储写回。
 * @throws std::invalid_argument n/scale 非法或 `out.size()!=n`。
 * @details 默认stream上的custom kernel只执行FP32复数计算，随后D2H、Host拓宽实部/虚部、
 * 按需H2D；不需要额外workspace。
 * @note 对齐 cuSignal morlet 的 complex wavelet 和 complete 语义；本接口固定一维 n 点
 * 输出，不接受 Python array-like 或 axis/batch 参数。
 */
template<class T>
void morlet_device(
    int n, T w, T scale, bool complete, DeviceArray<ComplexDouble>& out);

/**
 * @brief 在 CPU 上生成 Morlet2 小波，作为 GPU reference。
 * @tparam T FP32、FP16、INT32、INT16 或 INT8 参数类型。
 * @param n 输出点数，必须至少为 1。
 * @param scale 小波尺度，转换后必须大于 0。
 * @param w 无量纲中心频率。
 * @return host上`[n]`、固定`ComplexDouble`dtype的复数小波。
 * @throws std::invalid_argument `n<1` 或 `scale<=0`。
 * @details CPU reference使用FP64 exp/sin/cos计算；不使用workspace、stream或科学计算库。
 * @note 对齐cuSignal morlet2的scale/w中心频率定义，输出固定ComplexFP64。
 */
template<class T>
std::vector<ComplexDouble> morlet2_typed_cpu(int n, T scale, T w);

/**
 * @brief 在 GPU 上生成 Morlet2 小波。
 * @tparam T 五种参数类型；device执行ComplexFP32计算，正式输出ComplexFP64。
 * @param n 输出点数，必须至少为 1。
 * @param scale 正尺度参数。
 * @param w 无量纲中心频率。
 * @param out 调用者预分配的 device 输出 `[n]`。
 * @throws std::invalid_argument n/scale 非法或输出 size 不匹配。
 * @details 默认stream上的custom kernel只执行FP32复数计算，随后D2H、Host拓宽实部/虚部、
 * 按需H2D；不需要额外workspace并保持cuSignal Morlet2语义。
 */
template<class T>
void morlet2_device(
    int n, T scale, T w, DeviceArray<ComplexDouble>& out);

/**
 * @brief 在 CPU 上生成 Ricker（Mexican hat）小波 reference。
 * @tparam T FP32、FP16、INT32、INT16 或 INT8 宽度参数；计算和固定输出为 FP32。
 * @param n 输出点数，必须至少为 1。
 * @param a 正宽度/尺度参数，dtype T。
 * @return host上`[n]`、dtype FP64的单个实数组。
 * @throws std::invalid_argument `n<1` 或 `a<=0`。
 * @details 与 cuSignal Ricker 点采样语义对应；无 workspace、stream 或科学计算库。
 */
template<class T>
std::vector<double> ricker_typed_cpu(int n, T a);

/**
 * @brief 在 GPU 上生成 Ricker 小波。
 * @tparam T FP32、FP16、INT32、INT16或INT8宽度参数；固定输出FP64。
 * @param n 输出点数，必须至少为 1。
 * @param a 正宽度/尺度参数。
 * @param out device输出`[n]`，由Host拓宽后作为FP64存储写回。
 * @throws std::invalid_argument n/a 非法或输出 size 不匹配。
 * @details 默认stream上device只进行FP32计算，随后D2H、Host转FP64并按需H2D；无设备
 * FP64算术或额外workspace，保持cuSignal Ricker语义。
 */
template<class T>
void ricker_device(int n, T a, DeviceArray<double>& out);

using CwtRealWaveletCallable =
    std::function<std::vector<double>(int points, int width)>;
using CwtComplexWaveletCallable =
    std::function<std::vector<ComplexDouble>(int points, int width)>;

/** @brief 由任意C++ callable显式构造的CWT device wavelet bank。 */
struct CwtDeviceWorkspace {
    int data_count{0};
    int width_count{0};
    int max_wavelet_length{0};
    bool complex_output{false};
    DeviceArray<int> lengths;
    DeviceArray<ComplexFloat> wavelets;

    std::size_t output_size() const noexcept
    {
        return static_cast<std::size_t>(data_count) *
            static_cast<std::size_t>(width_count);
    }
};

/**
 * @brief 为实输出callable准备显式CWT workspace。
 * @details 对每个width调用`wavelet(min(10*int(width),N),int(width))`；准备阶段显式
 * 完成Host到device传输，后续GPU计算不再调用或缩窄callable。
 */
template<class T>
CwtDeviceWorkspace prepare_cwt_workspace(
    int data_count, const std::vector<T>& widths,
    const CwtRealWaveletCallable& wavelet);

/** @brief 为复输出callable准备显式CWT workspace。 */
template<class T>
CwtDeviceWorkspace prepare_cwt_workspace(
    int data_count, const std::vector<T>& widths,
    const CwtComplexWaveletCallable& wavelet);

/**
 * @brief 任意实wavelet callable的FP64 CPU reference。
 * @tparam T FP32、FP16、INT32、INT16或INT8信号和width类型。
 * @param x Host一维信号`[N]`。
 * @param widths Host尺度序列`[W]`。
 * @param wavelet 与cuSignal二参数wavelet等价的实数组callable。
 * @return Host row-major FP64数组，逻辑shape为`[W,N]`。
 * @throws std::invalid_argument callable为空、width产生非正长度或返回长度不符。
 * @details 纯CPU reference，不使用device、stream或workspace；内部精度不限，接口输出
 * 与cuSignal实wavelet CWT的FP64 dtype及same卷积语义一致。
 */
template<class T>
std::vector<double> cwt_typed_cpu(
    const std::vector<T>& x, const std::vector<T>& widths,
    const CwtRealWaveletCallable& wavelet);

/** @brief 任意复wavelet callable的ComplexFP64 CPU reference，输出shape为`[W,N]`。 */
template<class T>
std::vector<ComplexDouble> cwt_typed_cpu(
    const std::vector<T>& x, const std::vector<T>& widths,
    const CwtComplexWaveletCallable& wavelet);

/**
 * @brief 使用显式实wavelet bank执行CWT，正式输出FP64。
 * @tparam T FP32、FP16、INT32、INT16或INT8真实device信号输入类型。
 * @param x 调用者持有的device一维信号`[N]`。
 * @param workspace 由任意cuSignal等价callable显式准备的device wavelet bank。
 * @param out 调用者预分配的FP64存储，逻辑shape为`[W,N]`。
 * @throws std::invalid_argument workspace输出种类、shape或输出size不匹配。
 * @details 默认stream上执行FP32 same卷积、设备实部提取及FP64存储位编码；workspace
 * 准备是唯一Host到Device边界，正式调用内部无D2H/H2D，不使用FP64算术，也不调用
 * dlfft/dlrand。
 */
template<class T>
void cwt_device(
    const DeviceArray<T>& x, const CwtDeviceWorkspace& workspace,
    DeviceArray<double>& out);

/**
 * @brief 使用显式复wavelet bank执行CWT，device仅ComplexFP32计算，正式输出ComplexFP64。
 */
template<class T>
void cwt_device(
    const DeviceArray<T>& x, const CwtDeviceWorkspace& workspace,
    DeviceArray<ComplexDouble>& out);

/**
 * @brief 在 CPU 上由低通系数生成 QMF 高通系数 reference。
 * @tparam T FP32、FP16、INT32、INT16 或 INT8。
 * @param hk host 一维低通系数 `[N]`。
 * @return host上`[N]`、dtype INT64的索引序列；空输入返回空数组。
 * @details 固定版cuSignal 23.08.00不读取hk数值，只使用长度：
 * `out[i]=(N-(i+1))*(i为奇数?-1:1)`；CPU reference不创建GPU stream或device
 * workspace，五类型含FP32输入均只贡献shape。
 * @throws std::bad_alloc host 输出分配失败。
 */
template<class T>
std::vector<std::int64_t> qmf_typed_cpu(const std::vector<T>& hk);

/**
 * @brief 固定版23.08高维兼容入口。
 * @details `len(hk)`只取首维，因此输入可为任意rank连续数组，输出shape为`[shape[0]]`。
 */
template<class T>
std::vector<std::int64_t> qmf_typed_cpu(
    const std::vector<T>& hk, const std::vector<int>& shape);

/**
 * @brief 在 GPU 上由低通系数生成 QMF 高通系数。
 * @tparam T FP32、FP16、INT32、INT16或INT8输入；输出固定为INT64。
 * @param hk 调用者持有的 device 一维低通系数 `[N]`。
 * @param out 调用者预分配的 device 输出 `[N]`。
 * @throws std::invalid_argument 输入输出 size 不同。
 * @details 空输入直接返回；默认 stream 异步 custom template kernel，无 workspace、
 * dlfft/dlrand 或隐式传输，调用者读取输出前负责同步。
 * @note 对齐固定版cuSignal的实际索引生成行为；输入值本身不参与计算。
 */
template<class T>
void qmf_device(const DeviceArray<T>& hk, DeviceArray<std::int64_t>& out);

template<class T>
void qmf_device(
    const DeviceArray<T>& hk, DeviceArray<std::int64_t>& out,
    const std::vector<int>& shape);

}  // namespace cusignal
