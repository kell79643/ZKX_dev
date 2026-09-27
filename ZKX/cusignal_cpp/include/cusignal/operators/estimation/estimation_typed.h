#pragma once

#include <cusignal/runtime/device_array.h>
#include "simple_signal_typed.h"

#include <cstddef>
#include <vector>

namespace cusignal {

/** @brief 固定版cuSignal KalmanFilter的批量布局。 */
struct KalmanLayout {
    int dim_x = 0;
    int dim_z = 0;
    int points = 0;
    std::size_t x_count = 0;
    std::size_t p_count = 0;
    std::size_t h_count = 0;
    std::size_t r_count = 0;
    std::size_t z_count = 0;

    KalmanLayout() = default;
    KalmanLayout(int dim_x_in, int dim_z_in, int points_in)
    {
        reset(dim_x_in, dim_z_in, points_in);
    }

    void reset(int dim_x_in, int dim_z_in, int points_in);
};

/**
 * @brief predict可选参数。
 * @details `control_provided=true`严格对应cuSignal固定版的`u is not None`错误；
 * `process_noise_scalar=true`时忽略Q数组并以`q_scalar*I`作为每个point的Q。
 */
struct KalmanPredictOptions {
    bool control_provided = false;
    bool process_noise_scalar = false;
    float q_scalar = 1.0F;
};

/**
 * @brief update可选参数。
 * @details `measurement_present=false`对应`update(z=None)`并在解析H/R之前直接返回；
 * `measurement_noise_scalar=true`时忽略R数组并以`r_scalar*I`作为每个point的R。
 */
struct KalmanUpdateOptions {
    bool measurement_present = true;
    bool measurement_noise_scalar = false;
    float r_scalar = 1.0F;
};

/** @brief CPU reference持有的FP32状态；shape分别为`[points,dim_x,1]`和`[points,dim_x,dim_x]`。 */
struct KalmanHostState {
    KalmanLayout layout;
    std::vector<float> x;
    std::vector<float> p;
};

/**
 * @brief 正式GPU路径持有的FP32状态。
 * @details 五种业务输入只在初始化和矩阵参数边界出现；cuSignal不支持的FP16/整数
 * 进入批准的FP32合法路径后，x/P持续保持FP32，不量化回原始输入类型。
 */
struct KalmanDeviceState {
    KalmanLayout layout;
    DeviceArray<float> x;
    DeviceArray<float> p;

    KalmanDeviceState() = default;
    explicit KalmanDeviceState(const KalmanLayout& layout_in) { reset(layout_in); }
    void reset(const KalmanLayout& layout_in);
};

/**
 * @brief 调用者持有并复用的FP32 device workspace。
 * @details workspace不保存跨调用业务状态；所有buffer由本对象分配和释放，不能跨stream
 * 并发复用。predict/update不隐式同步，调用者读取状态、复用不同stream或计时时负责同步。
 * reset会按layout重新分配；容量不足由DeviceArray/CUDA错误显式报告。
 */
struct KalmanDeviceWorkspace {
    KalmanLayout layout;
    DeviceArray<float> f;
    DeviceArray<float> q;
    DeviceArray<float> alpha_sq;
    DeviceArray<float> h;
    DeviceArray<float> r;
    DeviceArray<float> z;
    DeviceArray<float> next_x;
    DeviceArray<float> next_p;
    DeviceArray<float> fp;
    DeviceArray<float> pht;
    DeviceArray<float> innovation;
    DeviceArray<float> s;
    DeviceArray<float> augmented;
    DeviceArray<float> gain;
    DeviceArray<float> i_kh;
    DeviceArray<float> left_covariance;

    KalmanDeviceWorkspace() = default;
    explicit KalmanDeviceWorkspace(const KalmanLayout& layout_in) { reset(layout_in); }
    void reset(const KalmanLayout& layout_in);
};

/** @brief 将五种Host业务输入转换为正式FP32 Kalman状态。 */
template <typename T>
KalmanHostState make_kalman_host_state(
    const std::vector<T>& x,
    const std::vector<T>& p,
    const KalmanLayout& layout);

/**
 * @brief 独立CPU predict reference。
 * @tparam T FP32、FP16、INT32、INT16或INT8业务输入。
 * @param state Host FP32状态，成功后原地更新。
 * @param f 状态转移矩阵输入。
 * @param q 过程噪声标量或矩阵输入。
 * @param alpha_sq 每个point的衰减系数平方。
 * @param options 已验证的cuSignal布局与control选项。
 * @throws std::invalid_argument shape、容量或workspace布局非法时抛出异常。
 * @details 计算`x=F*x`、`P=alpha_sq*F*P*F^T+Q`；F、Q和alpha_sq为真实T输入，
 * 输出状态固定FP32。Q scalar模式、control错误和逐point alpha与固定版cuSignal一致；
 * CPU reference不创建GPU stream或device workspace。
 */
template <typename T>
void kalman_predict_typed_cpu(
    KalmanHostState& state,
    const std::vector<T>& f,
    const std::vector<T>& q,
    const std::vector<T>& alpha_sq,
    const KalmanPredictOptions& options = {});

/**
 * @brief 独立CPU update reference。
 * @tparam T FP32、FP16、INT32、INT16或INT8业务输入。
 * @param state Host FP32状态，成功后原地更新。
 * @param h 观测矩阵输入。
 * @param r 观测噪声输入。
 * @param z 每个point的观测输入。
 * @param options 已验证的cuSignal布局选项。
 * @throws std::invalid_argument shape、容量或workspace布局非法时抛出异常。
 * @details 每个point使用独立z，执行固定版行排序Gauss-Jordan逆和完整Joseph协方差
 * `(I-KH)P(I-KH)^T+KRK^T`；不增加奇异阈值或跳过分支。z缺省时原样返回；
 * 五类型输入转FP32计算，CPU reference不创建GPU stream或device workspace。
 */
template <typename T>
void kalman_update_typed_cpu(
    KalmanHostState& state,
    const std::vector<T>& h,
    const std::vector<T>& r,
    const std::vector<T>& z,
    const KalmanUpdateOptions& options = {});

/** @brief 将五种device业务输入转换到调用者持有的FP32正式状态；无隐式同步。 */
template <typename T>
void initialize_kalman_device_state(
    const DeviceArray<T>& x,
    const DeviceArray<T>& p,
    KalmanDeviceState& state);

/**
 * @brief 正式GPU predict封装；五类型F/Q/alpha输入，FP32 state原地更新。
 * @tparam T FP32、FP16、INT32、INT16或INT8业务输入。
 * @param state 调用者持有的device FP32状态。
 * @param f device状态转移矩阵。
 * @param q device过程噪声标量或矩阵。
 * @param alpha_sq device逐point系数。
 * @param workspace 调用者持有并复用的FP32工作区。
 * @param options cuSignal布局与control选项。
 * @throws std::invalid_argument shape、容量或workspace不一致时抛出异常。
 * @details 仅使用default stream、custom FP32 kernels和调用者workspace；无内部申请、
 * H2D/D2H、FP64或同步。control输入按固定版cuSignal抛出`std::logic_error`。
 */
template <typename T>
void kalman_predict_device(
    KalmanDeviceState& state,
    const DeviceArray<T>& f,
    const DeviceArray<T>& q,
    const DeviceArray<T>& alpha_sq,
    KalmanDeviceWorkspace& workspace,
    const KalmanPredictOptions& options = {});

/**
 * @brief 正式GPU update封装；五类型H/R/z输入，FP32 state原地更新。
 * @tparam T FP32、FP16、INT32、INT16或INT8业务输入。
 * @param state 调用者持有的device FP32状态。
 * @param h device观测矩阵。
 * @param r device观测噪声。
 * @param z device逐point观测。
 * @param workspace 调用者持有并复用的FP32工作区。
 * @param options cuSignal布局选项。
 * @throws std::invalid_argument shape、容量或workspace不一致时抛出异常。
 * @details z必须为`[points,dim_z,1]`，不提供旧实现自创的跨point广播；完整Joseph
 * 更新和固定版cuSignal逆矩阵在FP32 device路径、默认stream执行。z缺省模式不读取H/R/z
 * 且不启动kernel。
 */
template <typename T>
void kalman_update_device(
    KalmanDeviceState& state,
    const DeviceArray<T>& h,
    const DeviceArray<T>& r,
    const DeviceArray<T>& z,
    KalmanDeviceWorkspace& workspace,
    const KalmanUpdateOptions& options = {});

/**
 * @brief 在设备端连续执行单状态、单观测维度的 Kalman predict/update 序列。
 * @details 仅接受 `KalmanLayout(1,1,1)`；observations 是同一状态的时间序列，
 * 不能按独立 points 批处理。函数无内部申请、H2D/D2H或同步，只启动一个顺序递推
 * kernel，数学步骤与逐次 `kalman_predict_device`/`kalman_update_device` 一致。
 */
void kalman_predict_update_scalar_sequence_device(
    KalmanDeviceState& state,
    const DeviceArray<float>& f,
    const DeviceArray<float>& q,
    const DeviceArray<float>& alpha_sq,
    const DeviceArray<float>& h,
    const DeviceArray<float>& r,
    const DeviceArray<float>& observations);

/**
 * @brief FP32旧调用形态的便利适配，仅用于现有调用方迁移。
 * @details 等价于alpha_sq全1、完整Q/R数组和逐point z；内部仍进入上述正式FP32状态路径。
 * 该适配会创建临时workspace并同步后回写，性能测试不得以它代替正式workspace入口。
 */
void kalman_predict_device(
    DeviceArray<float>& x,
    DeviceArray<float>& p,
    const DeviceArray<float>& q,
    const DeviceArray<float>& f,
    int dim_x,
    int points);

void kalman_update_device(
    DeviceArray<float>& x,
    DeviceArray<float>& p,
    const DeviceArray<float>& h,
    const DeviceArray<float>& r,
    const DeviceArray<float>& z,
    int dim_x,
    int dim_z,
    int points);

}  // namespace cusignal
