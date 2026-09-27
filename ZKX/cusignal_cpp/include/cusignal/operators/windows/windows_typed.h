#pragma once

#include <cusignal/runtime/device_array.h>
#include "simple_signal_typed.h"

#include <vector>

namespace cusignal {

/**
 * @brief 在 CPU 上生成 Dolph-Chebyshev 窗，作为五类型 GPU reference。
 * @tparam T 五种attenuation输入类型；输出固定FP64。
 * @param n 输出长度；0返回空数组，负数非法。
 * @param attenuation 旁瓣衰减绝对值，单位 dB；实现对负值取绝对值。
 * @param sym true 生成 n 点对称窗；false 按 n+1 点窗计算后截取前 n 点，默认 true。
 * @return host上一维`[n]`、dtype FP64的归一化实窗。
 * @throws std::invalid_argument `n<0`。
 * @details CPU reference将五类型（含FP32）参数转为FP64 custom direct DFT；M为0或1时
 * attenuation不参与计算；不创建GPU stream或device workspace，保持cuSignal语义。
 */
template <class T>
std::vector<double> chebwin_typed_cpu(int n, T attenuation, bool sym = true);

/**
 * @brief 在 GPU 上生成 Dolph-Chebyshev 窗。
 * @tparam T 五种attenuation输入类型；device计算与正式输出dtype解耦。
 * @param n 输出长度；0要求空输出，负数非法。
 * @param attenuation 旁瓣衰减绝对值（dB）。
 * @param out device输出，由Host拓宽后作为FP64存储写回。
 * @param sym true 使用 n 点，false 使用 n+1 点内部长度再截断，默认 true。
 * @throws std::invalid_argument `n<0`或输出size不匹配；CUDA/Thrust失败走项目异常机制。
 * @details 默认stream上M大于1时分配FP32临时数组并用Thrust直接取得窗口最大值；随后
 * D2H、Host拓宽并按需H2D。
 * @note 对齐 cuSignal chebwin 的 sym 和 dB attenuation 语义；为 ZQ500 高衰减稳定性使用
 * custom direct DFT，而不是宣称通过 dlfft，输出可观察语义仍由 CPU reference 检查。
 * @note FP32临时数组是函数内部workspace，不对调用者暴露或跨调用复用。
 */
template <class T>
void chebwin_device(
    int n, T attenuation, DeviceArray<double>& out, bool sym = true);

/**
 * @brief 在 CPU 上生成广义余弦窗 reference。
 * @tparam T 五种系数输入类型；固定版cuSignal先转FP64，输出固定FP64。
 * @param n 输出长度；0返回空数组，负数非法。
 * @param coefficients host一维系数`[K]`；空数组在M大于1时产生全零窗。
 * @param sym true 使用 n 点对称相位；false 使用 n+1 点相位并截前 n 点，默认 true。
 * @return host上一维`[n]`、dtype FP64的实窗。
 * @throws std::invalid_argument `n<0`。
 * @details CPU reference将五类型（含FP32）系数转FP64累加；M为0或1时系数不参与计算；
 * 不创建GPU stream或device workspace。
 */
template <class T>
std::vector<double> general_cosine_typed_cpu(
    int n, const std::vector<T>& coefficients, bool sym = true);

/**
 * @brief 固定版23.08系数数组高维兼容入口。
 * @details 上游以`len(a)`作为系数数目，故只使用连续存储的前`shape[0]`个系数。
 */
template <class T>
std::vector<double> general_cosine_typed_cpu(
    int n, const std::vector<T>& coefficients,
    const std::vector<int>& coefficient_shape, bool sym = true);

/**
 * @brief 在 GPU 上生成广义余弦窗。
 * @tparam T 五种系数输入类型；device计算与正式输出dtype解耦。
 * @param n 输出长度；0要求空输出，负数非法。
 * @param coefficients 调用者持有的device一维系数，可为空。
 * @param out device输出，由Host拓宽后作为FP64存储写回。
 * @param sym 对称/周期规则同 CPU reference，默认 true。
 * @throws std::invalid_argument `n<0`或输出size不匹配。
 * @details 默认stream上M大于1时custom kernel执行FP32累加，随后D2H、Host拓宽并按需H2D；
 * FP32临时数组是内部workspace。
 * @note 对齐固定版cuSignal的强制FP64输出、空系数、长度守卫和sym语义。
 */
template <class T>
void general_cosine_device(
    int n, const DeviceArray<T>& coefficients, DeviceArray<double>& out,
    bool sym = true);


template <class T>
void general_cosine_device(
    int n, const DeviceArray<T>& coefficients,
    const std::vector<int>& coefficient_shape, DeviceArray<double>& out,
    bool sym = true);

/**
 * @brief 在 CPU 上生成广义高斯窗 reference。
 * @tparam T 五种power/sig输入类型；输出固定FP64。
 * @param n 输出长度；0返回空数组，负数非法。
 * @param power 正负幂参数均按当前公式接受，无额外 guard。
 * @param width 尺度参数；允许负数和0并遵循cuSignal公式结果。
 * @param sym true 使用 n 点；false 使用 n+1 点内部长度并截前 n 点，默认 true。
 * @return host上一维`[n]`、dtype FP64的实窗。
 * @throws std::invalid_argument `n<0`。
 * @details CPU reference将五类型（含FP32）参数用于FP64 exp/pow；M为0或1时参数不参与
 * 计算；不创建GPU stream或device workspace。
 */
template <class T>
std::vector<double> general_gaussian_typed_cpu(
    int n, T power, T width, bool sym = true);

/**
 * @brief 在 GPU 上生成广义高斯窗。
 * @tparam T 五种power/sig输入类型；device计算与正式输出dtype解耦。
 * @param n 输出长度；0要求空输出，负数非法。
 * @param power 幂参数。
 * @param width 尺度参数；允许负数和0。
 * @param out device输出，由Host拓宽后作为FP64存储写回。
 * @param sym 对称/周期规则同 CPU reference，默认 true。
 * @throws std::invalid_argument `n<0`或输出size不匹配。
 * @details 默认stream上M大于1时custom kernel执行FP32计算，随后D2H、Host拓宽并按需H2D；
 * FP32临时数组是内部workspace。
 * @note 对齐固定版cuSignal的float64输出、长度守卫、参数公式和sym语义。
 */
template <class T>
void general_gaussian_device(
    int n, T power, T width, DeviceArray<double>& out, bool sym = true);

/**
 * @brief 在 CPU 上生成 Hamming 窗 reference。
 * @tparam T FP32、FP16、INT32、INT16或INT8业务请求类型；输出固定FP64。
 * @param n 输出长度；小于1时返回空数组。
 * @param sym true 使用 n 点对称窗；false 且 n 为偶数时按 n+1 点计算后截断，n 为奇数时仍使用 n 点。
 * @return host上长度`max(n,0)`、dtype FP64的一个实窗。
 * @throws std::bad_alloc Host输出分配失败；非正长度按cuSignal返回空数组而非异常。
 * @details CPU reference使用FP64余弦计算；无workspace/stream/科学计算库。
 * @note 对齐固定版cuSignal的float64输出、非正长度空数组和sym行为。
 */
template <class T>
std::vector<double> hamming_typed_cpu(int n, bool sym = true);

/**
 * @brief 在 GPU 上生成 Hamming 窗。
 * @tparam T 五种业务请求类型；device计算与正式输出dtype解耦。
 * @param n 输出长度；小于1时要求输出为空。
 * @param out device输出，由Host拓宽后作为FP64存储写回。
 * @param sym 规则同 CPU reference，默认 true。
 * @throws std::invalid_argument 输出size与`max(n,0)`不匹配。
 * @details 默认stream上的custom kernel只执行FP32计算，随后D2H、Host拓宽并按需H2D；
 * 无workspace并保持cuSignal语义。
 */
template <class T>
void hamming_device(int n, DeviceArray<double>& out, bool sym = true);

/**
 * @brief 生成可直接交给下游 FP32 device 算子的 Hamming 窗。
 * @details 输出由调用者预分配；稳定调用只启动一个 FP32 kernel，不分配、不传输、不执行
 * FP64 算术。该 resident 接口不改变公开 `hamming_device` 的固定 FP64 输出契约。
 */
template <class T>
void hamming_resident_device(int n, DeviceArray<float>& out, bool sym = true);

/**
 * @brief 在 CPU 上生成 Kaiser 窗 reference。
 * @tparam T FP32、FP16、INT32、INT16或INT8 beta输入类型；输出固定FP64。
 * @param n 输出长度；小于1时返回空数组。
 * @param beta 无量纲形状参数，当前实现接受任意有限 T 值。
 * @param sym true 使用 n 点；false 且 n 为偶数时用 n+1 点后截断，默认 true。
 * @return host上长度`max(n,0)`、dtype FP64的归一化实窗。
 * @throws std::bad_alloc Host输出分配失败；非正长度按cuSignal返回空数组而非异常。
 * @details CPU reference使用FP64 custom Bessel I0；无workspace/stream/科学计算库。
 * @note 对齐固定版cuSignal的float64输出、长度守卫和偶数周期窗扩展语义。
 */
template <class T>
std::vector<double> kaiser_typed_cpu(int n, T beta, bool sym = true);

/**
 * @brief 在 GPU 上生成 Kaiser 窗。
 * @tparam T 五种beta输入类型；device计算与正式输出dtype解耦。
 * @param n 输出长度；小于1时要求输出为空。
 * @param beta 无量纲形状参数。
 * @param out device输出，由Host拓宽后作为FP64存储写回。
 * @param sym 规则同 CPU reference，默认 true。
 * @throws std::invalid_argument 输出size与`max(n,0)`不匹配。
 * @details 默认stream上的custom kernel使用FP32 Bessel I0，随后D2H、Host拓宽并按需H2D；
 * 无workspace并保持cuSignal语义。
 */
template <class T>
void kaiser_device(int n, T beta, DeviceArray<double>& out, bool sym = true);

/**
 * @brief 在 CPU 上生成 Parzen 窗 reference。
 * @tparam T FP32、FP16、INT32、INT16或INT8业务请求类型；输出固定FP64。
 * @param n 输出长度；0返回空数组，负数非法。
 * @param sym true 使用 n 点；false 使用 n+1 点内部长度并截前 n 点，默认 true。
 * @return host上一维`[n]`、dtype FP64的实窗。
 * @throws std::invalid_argument `n<0`。
 * @details CPU reference使用FP64分段多项式；无workspace/stream/科学计算库。
 * @note 对齐固定版cuSignal的float64输出、长度守卫和sym截断语义。
 */
template <class T>
std::vector<double> parzen_typed_cpu(int n, bool sym = true);

/**
 * @brief 在 GPU 上生成 Parzen 窗。
 * @tparam T 五种业务请求类型；device计算与正式输出dtype解耦。
 * @param n 输出长度；0要求空输出，负数非法。
 * @param out device输出，由Host拓宽后作为FP64存储写回。
 * @param sym 对称/周期规则同 CPU reference，默认 true。
 * @throws std::invalid_argument `n<0`或输出size不匹配。
 * @details 默认stream上的custom kernel只执行FP32计算，随后D2H、Host拓宽并按需H2D；
 * 无workspace并保持cuSignal语义。
 */
template <class T>
void parzen_device(int n, DeviceArray<double>& out, bool sym = true);

/**
 * @brief 在 CPU 上生成 Taylor 窗 reference。
 * @tparam T 五种sll输入类型；输出固定FP64。
 * @param n 输出长度；0返回空数组，负数非法。
 * @param nbar 近似等幅旁瓣数；M大于1时必须至少为1。
 * @param sidelobe_level 旁瓣抑制度；0合法，负值按公式传播NaN。
 * @param normalize true 时使连续中心峰值为 1，默认 true。
 * @param sym true 使用 n 点；false 使用 n+1 点内部长度并截前 n 点，默认 true。
 * @return host上一维`[n]`、dtype FP64的实窗。
 * @throws std::invalid_argument `n<0`，或M大于1且`nbar<1`。
 * @details CPU reference将五类型（含FP32）参数转FP64系数并累加；M为0或1时其他参数不
 * 参与计算；不创建GPU stream或device workspace，保持cuSignal语义。
 */
template <class T>
std::vector<double> taylor_typed_cpu(
    int n, int nbar, T sidelobe_level, bool normalize = true,
    bool sym = true);

/**
 * @brief 在 GPU 上生成 Taylor 窗。
 * @tparam T 五种sll输入类型；device计算与正式输出dtype解耦。
 * @param n 输出长度；0要求空输出，负数非法。
 * @param nbar 近似等幅旁瓣数；M大于1时必须至少为1。
 * @param sidelobe_level 正旁瓣抑制度（dB）。
 * @param out device输出，由Host拓宽后作为FP64存储写回。
 * @param normalize 是否归一化，默认 true。
 * @param sym 对称/周期规则同 CPU reference，默认 true。
 * @throws std::invalid_argument 长度/nbar非法或输出size不匹配；CUDA失败走项目异常机制。
 * @details 默认stream上M大于1时在Host生成FP32系数并H2D，再由kernel执行FP32累加；
 * 随后Host拓宽。
 * @note 对齐固定版cuSignal的FP64输出、nbar/sll/norm/sym与边界顺序。
 * @note host 系数与临时 device 系数共同构成函数内部 workspace，返回前释放。
 */
template <class T>
void taylor_device(
    int n, int nbar, T sidelobe_level, DeviceArray<double>& out,
    bool normalize = true, bool sym = true);

/**
 * @brief 在 CPU 上生成三角窗 reference。
 * @tparam T FP32、FP16、INT32、INT16或INT8业务请求类型；输出固定FP64。
 * @param n 输出长度；0返回空数组，负数非法。
 * @param sym true 使用 n 点；false 使用 n+1 点内部长度并截前 n 点，默认 true。
 * @return host上一维`[n]`、dtype FP64的实窗。
 * @throws std::invalid_argument `n<0`。
 * @details 奇偶长度按整数rank公式计算；无workspace、stream或科学计算库。
 * @note 对齐固定版cuSignal的float64输出、长度守卫和sym截断语义。
 */
template <class T>
std::vector<double> triang_typed_cpu(int n, bool sym = true);

/**
 * @brief 在 GPU 上生成三角窗。
 * @tparam T 五种业务请求类型；device计算与正式输出dtype解耦。
 * @param n 输出长度；0要求空输出，负数非法。
 * @param out device输出，由Host拓宽后作为FP64存储写回。
 * @param sym 对称/周期规则同 CPU reference，默认 true。
 * @throws std::invalid_argument `n<0`或输出size不匹配。
 * @details 默认stream上的custom kernel只执行FP32计算，随后D2H、Host拓宽并按需H2D；
 * 无workspace并保持cuSignal语义。
 */
template <class T>
void triang_device(int n, DeviceArray<double>& out, bool sym = true);

}  // namespace cusignal
