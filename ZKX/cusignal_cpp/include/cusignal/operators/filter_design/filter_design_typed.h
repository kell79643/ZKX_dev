#pragma once

#include <cusignal/runtime/device_array.h>
#include "simple_signal_typed.h"

#include <vector>

namespace cusignal {

enum class FirwinPassZero {
    boolean_true,
    boolean_false,
    lowpass,
    highpass,
    bandpass,
    bandstop
};

enum class FirwinWindowMode { hamming, none, explicit_values };

struct FirwinOptions {
    bool use_width = false;
    double width = 0.0;
    FirwinWindowMode window_mode = FirwinWindowMode::hamming;
    std::vector<double> window;
    FirwinPassZero pass_zero = FirwinPassZero::boolean_true;
    bool scale = true;
    double fs = 2.0;
};

/**
 * @brief `firwin` 的预准备 device-resident workspace。
 *
 * @details 构造阶段按五种业务类型解释并校验Host cutoff，准备bands/window后只上传一次。稳定执行阶段
 * 复用这些device数组，不再读取Host参数、不分配且不发生H2D/D2H。该workspace服务内部
 * FP32流水线；公开cuSignal兼容入口仍保持FP64输出存储。
 */
class FirwinDeviceWorkspace {
public:
    FirwinDeviceWorkspace(
        int numtaps, const std::vector<float>& cutoff,
        const FirwinOptions& options = {});
    FirwinDeviceWorkspace(
        int numtaps, const std::vector<__half>& cutoff,
        const FirwinOptions& options = {});
    FirwinDeviceWorkspace(
        int numtaps, const std::vector<std::int32_t>& cutoff,
        const FirwinOptions& options = {});
    FirwinDeviceWorkspace(
        int numtaps, const std::vector<std::int16_t>& cutoff,
        const FirwinOptions& options = {});
    FirwinDeviceWorkspace(
        int numtaps, const std::vector<std::int8_t>& cutoff,
        const FirwinOptions& options = {});

    FirwinDeviceWorkspace(const FirwinDeviceWorkspace&) = delete;
    FirwinDeviceWorkspace& operator=(const FirwinDeviceWorkspace&) = delete;
    FirwinDeviceWorkspace(FirwinDeviceWorkspace&&) noexcept = default;
    FirwinDeviceWorkspace& operator=(FirwinDeviceWorkspace&&) noexcept = default;

    [[nodiscard]] int numtaps() const noexcept { return numtaps_; }

private:
    int numtaps_{0};
    bool scale_{true};
    DeviceArray<float> bands_;
    DeviceArray<float> window_;
    DeviceArray<float> normalization_;

    friend void firwin_resident_device(
        FirwinDeviceWorkspace&, DeviceArray<float>&);
    friend void firwin_resident_fp64_storage_device(
        FirwinDeviceWorkspace&, DeviceArray<double>&);
    friend void firwin_resident_stage_c_fp64_storage_device(
        FirwinDeviceWorkspace&, DeviceArray<double>&);
};

/**
 * @brief 使用已准备workspace生成FP32 device-resident FIR系数。
 * @param workspace 构造阶段已完成参数校验、分配和上传的workspace。
 * @param out 调用者预分配的FP32 device输出 `[numtaps]`。
 * @details 稳定调用仅启动device计算；不分配、不执行H2D/D2H或Host后处理。该接口用于
 * Task流水线和resident benchmark，不能把其耗时冒充公开FP64兼容入口耗时。
 */
void firwin_resident_device(
    FirwinDeviceWorkspace& workspace, DeviceArray<float>& out);

/**
 * @brief 使用已准备workspace生成公开契约要求的FP64 device存储。
 * @param workspace 构造阶段已完成参数校验、分配和上传的workspace。
 * @param out 调用者预分配的FP64 device输出 `[numtaps]`。
 * @details 稳定调用只执行FP32计算，并以整数位操作写入等值IEEE-754 FP64存储；不执行
 * FP64算术，也不分配、不发生H2D/D2H或Host后处理。该接口用于五种业务类型完成setup后
 * 的同语义resident benchmark，不能跨compatibility scope计算加速比。
 */
void firwin_resident_fp64_storage_device(
    FirwinDeviceWorkspace& workspace, DeviceArray<double>& out);

/**
 * @brief 保留阶段C两kernel结构，用于与融合实现做同scope受控消融。
 * @details 该入口同样复用workspace并保持FP64存储语义，但分别启动归一化归约和输出kernel；
 * 不作为小规模正式快路径，大于融合上限时作为正确的通用回退。
 */
void firwin_resident_stage_c_fp64_storage_device(
    FirwinDeviceWorkspace& workspace, DeviceArray<double>& out);

/**
 * @brief device cutoff 与 FP32 显式窗口直连的低通 FIR resident 接口。
 * @tparam T 五种真实 cutoff 输入类型。
 * @details 仅覆盖单 cutoff、低通、显式窗口场景；输出由调用者预分配为 FP32。稳定调用只启动
 * 一个融合 kernel，不读取 Host、不分配、不传输。公开通用 `firwin_device` 语义不变。
 */
template <typename T>
void firwin_lowpass_explicit_window_resident_device(
    int numtaps, const DeviceArray<T>& cutoff,
    const DeviceArray<float>& window, DeviceArray<float>& out,
    float fs, bool scale = true);

/**
 * @brief 在CPU上按完整firwin选项设计FIR，作为GPU reference。
 * @tparam T FP32、FP16、INT32、INT16 或 INT8 cutoff 输入；固定输出为FP64。
 * @param numtaps 正 tap 数。
 * @param cutoff host 一维严格递增数组 `[K]`，K>=1，每项位于 `(0,fs/2)`。
 * @param options width、window、完整pass_zero、scale和fs选项。
 * @return host `[numtaps]`、dtype FP64 的对称 FIR 系数；numtaps=1 使用窗值1。
 * @throws std::invalid_argument numtaps、cutoff、pass_zero、window、Nyquist或fs非法。
 * @details 支持low/high/band-pass/band-stop和多频带交替语义；width按cuSignal规则
 * 覆盖window并生成Kaiser。CPU sinc/window只作reference，无device或stream；临时bands和
 * window是workspace。
 */
template <typename T>
std::vector<double> firwin_typed_cpu(
    int numtaps, const std::vector<T>& cutoff,
    const FirwinOptions& options = {});

/**
 * @brief 在GPU上以FP32完成主计算，并按cuSignal契约返回FP64存储。
 * @tparam T 五种业务cutoff输入类型；设备计算为FP32，正式输出为FP64。
 * @param numtaps 正 tap 数。
 * @param cutoff 调用者持有的 device 严格递增 `[K]` cutoff，K>=1。
 * @param out 调用者预分配的device FP64输出`[numtaps]`。
 * @param options 完整width/window/pass_zero/scale/fs选项。
 * @throws std::invalid_argument shape、cutoff、pass_zero、window、Nyquist或fs非法。
 * @details cutoff先D2H校验并在Host准备bands/window workspace；custom kernel执行
 * FP32计算后以整数位操作把FP32值写成等值IEEE-754 FP64存储，不执行FP64算术，也不再
 * D2H→Host拓宽→H2D。GPU kernel 使用默认stream；调用者在读取输出或跨stream复用
 * workspace 前负责同步。参数准备仍属于公开compatibility成本；高性能流水线应使用
 * `FirwinDeviceWorkspace` 和 `firwin_resident_device`。
 */
template <typename T>
void firwin_device(
    int numtaps, const DeviceArray<T>& cutoff, DeviceArray<double>& out,
    const FirwinOptions& options = {});

enum class Firwin2WindowMode { hamming, none, explicit_values };

struct Firwin2Options {
    int nfreqs = 0;
    Firwin2WindowMode window_mode = Firwin2WindowMode::hamming;
    std::vector<float> window;
    bool antisymmetric = false;
    double fs = 2.0;
};

/**
 * @brief 在CPU上按任意频率响应采样设计Firwin2 reference。
 * @tparam T 五种业务freq/gain输入类型；固定输出FP64。
 * @param numtaps 正 tap 数。
 * @param freq host一维频率`[K]`，首项0、末项fs/2，允许单次重复。
 * @param gain host 一维响应 `[K]`，长度必须与 freq 相同。
 * @param options nfreqs、Hamming/None/任意显式window、antisymmetric及fs完整选项。
 * @return host `[numtaps]`、dtype FP64的FIR系数。
 * @throws std::invalid_argument shape、nfreqs、频点重复、filter type gain、window或fs非法。
 * @details 对齐cuSignal firwin2的插值、相位、IRFFT和type I-IV规则；CPU reference不使用
 * device或stream，临时频谱/time/window是一次性workspace，五类型可使用FP32或更高
 * Host中间精度计算。
 */
template <typename T>
std::vector<double> firwin2_typed_cpu(
    int numtaps, const std::vector<T>& freq, const std::vector<T>& gain,
    const Firwin2Options& options = {});

/**
 * @brief 在GPU上按频率响应采样设计Firwin2，正式输出FP64。
 * @tparam T 五种业务freq/gain输入类型；device仅FP32/ComplexFP32计算。
 * @param numtaps 正 tap 数。
 * @param freq device `[K]` 频率，端点/单调/范围规则同 CPU reference。
 * @param gain device `[K]` 响应。
 * @param out device FP64存储`[numtaps]`，由Host拓宽后写回。
 * @param options 完整nfreqs/window/antisymmetric/fs选项。
 * @throws std::invalid_argument 参数规则同CPU reference；FFT/CUDA失败走项目异常。
 * @details freq/gain先显式D2H验证并转为FP32插值点；默认stream上执行ComplexFP32 IFFT
 * 和window乘法，随后D2H、Host拓宽、可选H2D。workspace含频谱/time/window和FFT plan；
 * 对齐cuSignal且CUDA kernel不使用FP64，无dlrand。
 */
template <typename T>
void firwin2_device(
    int numtaps, const DeviceArray<T>& freq, const DeviceArray<T>& gain,
    DeviceArray<double>& out, const Firwin2Options& options = {});

}  // namespace cusignal
