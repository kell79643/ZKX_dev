#pragma once

#include <cusignal/runtime/device_array.h>
#include <cusignal/runtime/device_complex.h>
#include "simple_signal_typed.h"

#include <string>
#include <vector>

namespace cusignal {

struct WaveformBroadcastOptions {
    std::vector<int> t_shape;
    std::vector<int> control_shape;
};

struct WaveformBroadcastCpuResult {
    std::vector<double> values;
    std::vector<int> shape;
};

struct WaveformBroadcastDeviceResult {
    DeviceArray<double> values;
    std::vector<int> shape;
};

/**
 * @brief 在 CPU 上生成线性调频 Chirp，作为五类型 GPU 入口的 reference。
 * @tparam T FP32、FP16、INT32、INT16 或 INT8；输入先提升为 FP32，输出按项目统一规则转换回 T。
 * @param t host 上任意逻辑 rank 的连续 row-major 时间采样；shape 由调用者持有。
 * @param f0 `t=0` 时频率，单位为时间单位的倒数。
 * @param t1 达到 `f1` 的参考时刻。
 * @param f1 `t=t1` 时频率，单位与 `f0` 相同。
 * @param phase 初相位，单位为度，默认 0。
 * @return 与 `t` 同 shape、展平存储、dtype T 的实波形；空输入返回空 vector。
 * @throws std::bad_alloc Host输出分配失败；linear参数沿用cuSignal浮点传播语义。
 * @details 无 workspace、stream、H2D/D2H 或科学计算库调用；该快捷入口使用cuSignal
 * linear方法。
 */
template<class T>
std::vector<T> chirp_typed_cpu(
    const std::vector<T>& t, float f0, float t1, float f1, float phase = 0);

/**
 * @brief 覆盖cuSignal四种实数method及别名的完整CPU reference。
 * @throws std::invalid_argument method非法，或logarithmic/hyperbolic频率约束不满足。
 */
template<class T>
std::vector<T> chirp_typed_cpu(
    const std::vector<T>& t, float f0, float t1, float f1,
    const std::string& method, float phase = 0, bool vertex_zero = true);

/** @brief cuSignal linear `type="complex"`的ComplexFP32 CPU reference。 */
template<class T>
std::vector<ComplexFloat> chirp_complex_typed_cpu(
    const std::vector<T>& t, float f0, float t1, float f1, float phase = 0);

/**
 * @brief 在 GPU 上生成线性调频 Chirp。
 * @tparam T FP32、FP16、INT32、INT16 或 INT8；FP32 计算后按统一舍入/饱和策略写回 T。
 * @param t 调用者持有的任意逻辑 rank、连续 row-major device 时间数组。
 * @param out 调用者预分配的同 shape 展平 device 输出，dtype T；不得与 `t` 尺寸不同。
 * @param f0 `t=0` 时频率。
 * @param t1 达到 `f1` 的参考时刻。
 * @param f1 `t=t1` 时频率。
 * @param phase 初相位（度），默认 0。
 * @throws std::invalid_argument `out.size()!=t.size()`。
 * @details 使用默认 stream 启动共享 custom template kernel，不分配 workspace，不执行隐式
 * H2D/D2H，不调用 dlfft/dlrand。函数只检查 launch 错误而不做完成同步；读取 `out` 前由调用者同步。
 * @note 对齐cuSignal chirp的linear实输出；复杂输出使用`chirp_complex_device`。
 */
template<class T>
void chirp_device(
    const DeviceArray<T>& t, DeviceArray<T>& out,
    float f0, float t1, float f1, float phase = 0);

/**
 * @brief 在 GPU 上按指定方法生成 Chirp。
 * @tparam T FP32、FP16、INT32、INT16 或 INT8；输入输出 dtype 均为 T，中间计算为 FP32。
 * @param t 调用者持有的 device 一维时间数组 `[N]`。
 * @param out 调用者预分配的 device 输出 `[N]`。
 * @param f0 起始频率。
 * @param t1 达到 `f1` 的参考时刻。
 * @param f1 终止参考频率。
 * @param method 支持四个完整名称及cuSignal别名`lin/li/quad/q/log/lo/hyp`。
 * @param phase 初相位（度），默认 0。
 * @param vertex_zero quadratic 模式的顶点位置选择，默认 true；其他模式忽略该参数。
 * @throws std::invalid_argument method非法、频率约束不满足或输入输出size不同。
 * @details 与 cuSignal 的四种实数 method 对应；本 C++ API 接受标量控制参数，时间数组可为任意逻辑 rank。
 * 无 workspace/科学库/隐式传输，默认 stream 异步执行，调用者负责同步与数组生命周期。
 */
template<class T>
void chirp_device(
    const DeviceArray<T>& t, DeviceArray<T>& out,
    float f0, float t1, float f1, const std::string& method,
    float phase = 0, bool vertex_zero = true);

/**
 * @brief cuSignal linear `type="complex"`的正式GPU入口。
 * @details 五种真实时间输入均返回ComplexFP32；设备端FP32计算，无隐式传输或FP64。
 */
template<class T>
void chirp_complex_device(
    const DeviceArray<T>& t, DeviceArray<ComplexFloat>& out,
    float f0, float t1, float f1, float phase = 0);

template <class T>
struct GausspulseTypedResult {
    std::vector<T> in_phase;
    std::vector<T> quadrature;
    std::vector<T> envelope;
};

/**
 * @brief cuSignal gausspulse CPU实部快捷入口；固定`bwr=-6`，输出dtype与输入T一致。
 * @tparam T FP32、FP16、INT32、INT16或INT8时间输入及数组输出类型。
 * @param t Host 任意逻辑 rank 的连续 row-major 时间数组，shape 由调用者持有。
 * @param fc 非负中心频率。
 * @param bw 正分数带宽。
 * @return Host同T实部波形。
 * @throws std::invalid_argument fc或bw非法时抛出异常。
 * @details 使用FP32兼容中间计算并委托正式CPU reference，不创建GPU stream或device
 * workspace。
 */
template<class T>
std::vector<T> gausspulse_typed_cpu(
    const std::vector<T>& t, float fc, float bw);

/**
 * @brief 完整CPU reference，覆盖cuSignal的`retquad/retenv`四种返回组合。
 * @tparam T FP32、FP16、INT32、INT16或INT8时间输入及数组输出类型。
 * @param t host 任意逻辑 rank 的连续 row-major 时间数组。
 * @param fc 中心频率，允许0，负值非法。
 * @param bw 正分数带宽。
 * @param bwr 负带宽参考电平。
 * @param retquad 是否填充quadrature；false时对应vector为空。
 * @param retenv 是否填充envelope；false时对应vector为空。
 */
template<class T>
GausspulseTypedResult<T> gausspulse_typed_cpu(
    const std::vector<T>& t, float fc, float bw, float bwr,
    bool retquad, bool retenv);

/**
 * @brief 对齐`gausspulse("cutoff",...)`的Host FP64标量路径。
 * @details 不启动GPU kernel，因而不产生设备端FP64运算。
 */
double gausspulse_cutoff(
    double fc = 1000.0, double bw = 0.5,
    double bwr = -6.0, double tpr = -60.0);

/**
 * @brief 在 GPU 上生成 Gausspulse 实部。
 * @tparam T FP32、FP16、INT32、INT16 或 INT8；中间计算为 FP32，输出为 T。
 * @param t 调用者持有的 device 时间数组 `[N]`。
 * @param out 调用者预分配的 device 实部输出 `[N]`。
 * @param fc 中心频率，允许0，负值非法。
 * @param bw 分数带宽，必须大于 0。
 * @throws std::invalid_argument 参数非法或输入输出size不同。
 * @details 复用正式 Gausspulse template kernel，不生成 I/Q envelope，不调用 dlfft/dlrand，
 * 不执行隐式传输。默认 stream 异步 launch；调用者在读取输出前负责同步。
 * @note 对齐cuSignal且固定`bwr=-6`；完整返回组合使用下方可选输出入口。
 * @note 该入口不使用内部 workspace；所有 FP32 中间量在线程寄存器中计算。
 */
template<class T>
void gausspulse_device(
    const DeviceArray<T>& t, DeviceArray<T>& out, float fc, float bw);

/**
 * @brief 完整GPU数组入口；空指针分别表示`retquad=false`或`retenv=false`。
 * @tparam T FP32、FP16、INT32、INT16或INT8时间及输出类型。
 * @param t device 任意逻辑 rank 的连续 row-major 时间数组。
 * @param in_phase device同T实部输出。
 * @param quadrature 可选device同T正交输出。
 * @param envelope 可选device同T包络输出。
 * @param fc 非负中心频率。
 * @param bw 正分数带宽。
 * @param bwr 负带宽参考电平。
 * @throws std::invalid_argument cuSignal参数或任一输出size非法时抛出异常。
 * @details 输入及所有被请求数组输出保持T；默认stream执行FP32中间计算，无设备端FP64
 * 或内部workspace。
 */
template<class T>
void gausspulse_device(
    const DeviceArray<T>& t,
    DeviceArray<T>& in_phase,
    DeviceArray<T>* quadrature,
    DeviceArray<T>* envelope,
    float fc, float bw, float bwr = -6.0F);

/**
 * @brief 在 GPU 上生成 Gausspulse 的同类型 I/Q/envelope 三路输出。
 * @tparam T FP32、FP16、INT32、INT16 或 INT8；输入和三路输出 dtype 均为 T。
 * @param t 调用者持有的 device 时间数组 `[N]`。
 * @param in_phase 调用者预分配的同类型实部 `[N]`。
 * @param quadrature 调用者预分配的同类型正交分量 `[N]`。
 * @param envelope 调用者预分配的同类型包络 `[N]`。
 * @param fc 中心频率，允许0，负值非法。
 * @param bw 分数带宽，必须大于0。
 * @param bwr 带宽参考电平（dB），必须小于0，默认 -6 dB。
 * @throws std::invalid_argument 参数非法或任一输出 size 与输入不同。
 * @details 与实部 overload 复用正式 Gausspulse template kernel；FP32 中间计算，默认
 * stream 异步执行，无 workspace、科学库或隐式传输。cuSignal 的 cutoff 查询参数 `tpr`
 * 不影响波形样值，因此本签名不接收后再忽略。
 * @note 对齐 cuSignal gausspulse 的 `retquad/retenv` 波形生成语义。
 */
template<class T>
void gausspulse_device(
    const DeviceArray<T>& t,
    DeviceArray<T>& in_phase,
    DeviceArray<T>& quadrature,
    DeviceArray<T>& envelope,
    float fc, float bw, float bwr = -6.0F);

/**
 * @brief 在 CPU 上生成周期 Sawtooth，作为 GPU reference。
 * @tparam T FP32、FP16、INT32、INT16 或 INT8。
 * @param t host 上任意逻辑 rank 的连续 row-major 相位数组，单位为弧度，按 `2*pi` 周期折返。
 * @param width 上升段占周期比例，默认1；范围外值逐元素产生NaN。
 * @return 与 `t` 同 shape、展平存储、dtype FP64 的实波形。
 * @details 标量重载按全数组广播；数组重载要求width长度为1或与t相同；CPU reference
 * 不创建GPU stream或device workspace并保持cuSignal周期语义。
 * @throws std::bad_alloc host 输出分配失败。
 */
template<class T>
std::vector<double> sawtooth_typed_cpu(
    const std::vector<T>& t, double width = 1.0);

template<class T>
std::vector<double> sawtooth_typed_cpu(
    const std::vector<T>& t, const std::vector<T>& width);

/** @brief 对齐固定版cuSignal/CuPy逐尾维广播规则的任意rank入口。 */
template<class T>
WaveformBroadcastCpuResult sawtooth_typed_cpu(
    const std::vector<T>& t, const std::vector<T>& width,
    const WaveformBroadcastOptions& options);

/**
 * @brief 在 GPU 上生成周期 Sawtooth。
 * @tparam T FP32、FP16、INT32、INT16或INT8相位输入；固定输出FP64。
 * @param t 调用者持有的 device 相位数组 `[N]`，单位为弧度。
 * @param out device输出`[N]`，kernel直接写入FP64存储位模式。
 * @param width 标量上升段比例，默认1，范围外值产生NaN。
 * @throws std::invalid_argument 输入输出 size 不同。
 * @details 设备只执行FP32计算，随后D2H、Host转FP64并按需H2D；数组重载支持长度1
 * 或与t等长的width广播子集；默认stream执行且不复用workspace，保持cuSignal语义。
 */
template<class T>
void sawtooth_device(
    const DeviceArray<T>& t, DeviceArray<double>& out, double width = 1.0);

template<class T>
void sawtooth_device(
    const DeviceArray<T>& t, const DeviceArray<T>& width,
    DeviceArray<double>& out);

/** @brief 与CPU广播入口一一对应，返回广播后的shape与FP64 device结果。 */
template<class T>
WaveformBroadcastDeviceResult sawtooth_device(
    const DeviceArray<T>& t, const DeviceArray<T>& width,
    const WaveformBroadcastOptions& options);

/**
 * @brief 在 CPU 上生成周期 Square，作为 GPU reference。
 * @tparam T FP32、FP16、INT32、INT16或INT8相位输入；固定输出FP64。
 * @param t host 任意逻辑 rank 的连续 row-major 相位数组，单位为弧度并按 `2*pi` 折返。
 * @param duty 高电平占空比，默认0.5，范围外值逐元素产生NaN。
 * @return 与 `t` 同 shape、展平存储、dtype FP64 的`+1/-1/NaN`波形。
 * @details 标量重载按全数组广播；数组重载要求duty长度为1或与t相同；不创建device
 * workspace并保持cuSignal周期语义。
 * @throws std::bad_alloc host 输出分配失败。
 * @note CPU reference 不使用 stream；范围外 duty 按项目 clamp 规则处理。
 */
template<class T>
std::vector<double> square_typed_cpu(
    const std::vector<T>& t, double duty = 0.5);

template<class T>
std::vector<double> square_typed_cpu(
    const std::vector<T>& t, const std::vector<T>& duty);

/** @brief 对齐固定版cuSignal/CuPy逐尾维广播规则的任意rank入口。 */
template<class T>
WaveformBroadcastCpuResult square_typed_cpu(
    const std::vector<T>& t, const std::vector<T>& duty,
    const WaveformBroadcastOptions& options);

/**
 * @brief 在 GPU 上生成周期 Square。
 * @tparam T FP32、FP16、INT32、INT16或INT8相位输入；输出固定FP64。
 * @param t 调用者持有的 device 相位数组 `[N]`，单位为弧度。
 * @param out device输出`[N]`，由Host拓宽后作为FP64存储写回。
 * @param duty 标量占空比，默认0.5，范围外值产生NaN。
 * @throws std::invalid_argument 输入输出 size 不同。
 * @details 设备以FP32执行相位与duty判断，并直接写FP64的+1/-1/NaN存储位模式，
 * 不执行FP64算术或转换；数组重载支持长度1或与t等长的duty广播子集，默认stream执行，
 * 且不创建内部workspace；输出语义对齐固定版cuSignal。
 */
template<class T>
void square_device(
    const DeviceArray<T>& t, DeviceArray<double>& out, double duty = 0.5);

template<class T>
void square_device(
    const DeviceArray<T>& t, const DeviceArray<T>& duty,
    DeviceArray<double>& out);

/** @brief 与CPU广播入口一一对应，返回广播后的shape与FP64 device结果。 */
template<class T>
WaveformBroadcastDeviceResult square_device(
    const DeviceArray<T>& t, const DeviceArray<T>& duty,
    const WaveformBroadcastOptions& options);

/**
 * @brief 在 CPU 上生成一维 Unit impulse，作为 GPU reference。
 * @tparam T TASK_F 五类型契约用例标签；本生成器没有业务数组输入，T 不参与计算或决定输出类型。
 * @param shape 输出长度；`shape<=0` 返回空 vector。
 * @param idx 脉冲位置，默认 0；越界时返回全零，不抛异常。
 * @return host上长度`max(shape,0)`、dtype FP64的一维数组，合法idx处为1。
 * @details 固定版cuSignal 23.08源码声明dtype参数，但实际实现未使用该参数，固定kernel输出
 * FP64；不经过FP32计算。CPU reference保持这一实际行为，不创建GPU stream或device workspace。
 */
template<class T>
std::vector<double> unit_impulse_typed_cpu(int shape, int idx = 0);

/**
 * @brief 对齐固定版23.08对tuple/list shape与idx的实际处理。
 * @details 固定版kernel只使用shape[0]和idx[0]，因此返回仍为一维首维长度。
 */
template<class T>
std::vector<double> unit_impulse_typed_cpu(
    const std::vector<int>& shape, const std::vector<int>& idx = {});

/** @brief `idx="mid"`入口；其他字符串抛出std::invalid_argument。 */
template<class T>
std::vector<double> unit_impulse_typed_cpu(
    const std::vector<int>& shape, const std::string& idx);

/**
 * @brief 在 GPU 上向预分配数组写入一维 Unit impulse。
 * @tparam T TASK_F 五类型契约用例标签；不是业务输入dtype，固定输出FP64。
 * @param out 预分配的device一维FP64输出`[N]`；函数原位写入，不替换其存储。
 * @param idx 脉冲位置，默认 0；越界时输出全零，不抛异常。
 * @details 默认stream上的device kernel使用uint64_t写入IEEE-754 FP64的0/1确定位模式，
 * 不执行FP32/FP64转换或算术，不产生临时分配、D2H、Host拓宽或H2D，也不需要额外workspace。
 * @note 对齐 cuSignal unit_impulse 的一维整数索引子集；tuple shape、字符串 `mid` 和多维
 * 坐标未进入该 typed 签名，越界索引明确产生全零。
 */
template<class T>
void unit_impulse_device(DeviceArray<double>& out, int idx = 0);

template<class T>
void unit_impulse_device(
    DeviceArray<double>& out, const std::vector<int>& shape,
    const std::vector<int>& idx = {});

template<class T>
void unit_impulse_device(
    DeviceArray<double>& out, const std::vector<int>& shape,
    const std::string& idx);

}  // namespace cusignal
