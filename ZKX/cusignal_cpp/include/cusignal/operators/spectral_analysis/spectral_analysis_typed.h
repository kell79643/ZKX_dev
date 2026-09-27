#pragma once

#include <cusignal/runtime/device_array.h>
#include <cusignal/backends/fft/fft_interface.h>
#include <cusignal/operators/filtering/filtering_typed.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cusignal {

/**
 * @brief typed 频谱算子使用的 FP32 窗函数参数。
 */
struct WindowParams {
    std::string type = "hann";
    float param = 0.0F;
    std::vector<float> custom_window;

    WindowParams() = default;
    explicit WindowParams(std::string type_in) : type(std::move(type_in)) {}
    WindowParams(std::string type_in, float param_in)
        : type(std::move(type_in)), param(param_in) {}
    explicit WindowParams(std::vector<float> custom_window_in)
        : custom_window(std::move(custom_window_in)) {}
};

struct StftDeviceParams {
    float fs = 1.0F;
    WindowParams window;
    int nperseg = 256;
    int noverlap = -1;
    int nfft = 0;
    std::string detrend;
    bool return_onesided = true;
    std::string boundary = "zeros";
    bool padded = true;
    int axis = -1;
    std::vector<int> shape;
};

namespace spectral_layout {

struct AxisPlan {
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    std::vector<int> line_bases;
    int axis = 0;
    int axis_length = 0;
    int axis_stride = 1;
    int inner_count = 1;
    int outer_count = 1;
    int line_count = 1;
};

AxisPlan prepare_axis_plan(
    std::size_t element_count, const std::vector<int>& shape, int axis,
    int frequency_count, int time_count, bool append_time);

struct CsdPlan {
    std::vector<int> x_shape;
    std::vector<int> y_shape;
    std::vector<int> output_shape;
    std::vector<int> x_line_bases;
    std::vector<int> y_line_bases;
    int x_axis_length = 0;
    int y_axis_length = 0;
    int x_axis_stride = 1;
    int y_axis_stride = 1;
    int output_inner_count = 1;
    int line_count = 1;
};

CsdPlan prepare_csd_plan(
    std::size_t x_element_count, std::size_t y_element_count,
    const std::vector<int>& x_shape, const std::vector<int>& y_shape,
    int axis, int frequency_count);

} // namespace spectral_layout

struct StftDeviceWorkspace {
    DeviceArray<float> window;
    DeviceArray<ComplexFloat> fft_input;
    DeviceArray<ComplexFloat> fft_output;
    std::unique_ptr<FFTInterface> fft;
    ComplexFloat* fft_input_ptr = nullptr;
    ComplexFloat* fft_output_ptr = nullptr;
    std::size_t fft_input_capacity = 0;
    std::size_t fft_output_capacity = 0;
    int input_size = 0;
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    DeviceArray<int> line_bases;
    int axis = 0;
    int axis_length = 0;
    int axis_stride = 1;
    int inner_count = 1;
    int outer_count = 1;
    int line_count = 1;
    int nperseg = 0;
    int noverlap = 0;
    int nfft = 0;
    int nframes = 0;
    int nf = 0;
    bool return_onesided = true;
    bool complex_input = false;
    std::string boundary = "zeros";
    bool padded = true;
    std::string detrend = "constant";
    float scale = 1.0F;

    StftDeviceWorkspace() = default;
    StftDeviceWorkspace(
        int input_size_in, const StftDeviceParams& params,
        bool complex_input_in = false)
    {
        reset(input_size_in, params, complex_input_in);
    }

    void reset(
        int input_size_in, const StftDeviceParams& params,
        bool complex_input_in = false);
    void bind_fft_scratch(
        DeviceArray<ComplexFloat>& input_scratch,
        DeviceArray<ComplexFloat>& output_scratch);
    void clear_fft_scratch_bindings();
    [[nodiscard]] ComplexFloat* fft_input_data() noexcept { return fft_input_ptr; }
    [[nodiscard]] ComplexFloat* fft_output_data() noexcept { return fft_output_ptr; }
    [[nodiscard]] const ComplexFloat* fft_input_data() const noexcept { return fft_input_ptr; }
    [[nodiscard]] const ComplexFloat* fft_output_data() const noexcept { return fft_output_ptr; }
    [[nodiscard]] std::size_t fft_scratch_size() const noexcept
    {
        return static_cast<std::size_t>(line_count)
            * static_cast<std::size_t>(nframes) * static_cast<std::size_t>(nfft);
    }
    [[nodiscard]] std::size_t output_size() const noexcept
    {
        return static_cast<std::size_t>(line_count)
            * static_cast<std::size_t>(nf) * static_cast<std::size_t>(nframes);
    }
};

struct SpectrogramDeviceParams {
    float fs = 1.0F;
    WindowParams window{"tukey", 0.25F};
    int nperseg = 0;
    int noverlap = -1;
    int nfft = 0;
    std::string detrend = "constant";
    bool return_onesided = true;
    std::string scaling = "density";
    int axis = -1;
    std::string mode = "psd";
    std::vector<int> shape;
};

struct SpectrogramDeviceWorkspace {
    StftDeviceWorkspace stft;
    float post_scale = 1.0F;

    SpectrogramDeviceWorkspace() = default;
    SpectrogramDeviceWorkspace(
        int input_size_in, const SpectrogramDeviceParams& params)
    {
        reset(input_size_in, params);
    }

    void reset(int input_size_in, const SpectrogramDeviceParams& params);
    [[nodiscard]] std::size_t output_size() const noexcept { return stft.output_size(); }
};

struct CsdDeviceParams {
    float fs = 1.0F;
    WindowParams window;
    int nperseg = 0;
    int noverlap = -1;
    int nfft = 0;
    std::string detrend = "constant";
    bool return_onesided = true;
    std::string scaling = "density";
    int axis = -1;
    std::string average = "mean";
    std::vector<int> x_shape;
    std::vector<int> y_shape;
};

struct CsdDeviceWorkspace {
    DeviceArray<float> window;
    DeviceArray<ComplexFloat> x_fft_input;
    DeviceArray<ComplexFloat> y_fft_input;
    DeviceArray<ComplexFloat> x_fft_output;
    DeviceArray<ComplexFloat> y_fft_output;
    std::unique_ptr<FFTInterface> fft;
    ComplexFloat* x_fft_input_ptr = nullptr;
    ComplexFloat* y_fft_input_ptr = nullptr;
    ComplexFloat* x_fft_output_ptr = nullptr;
    ComplexFloat* y_fft_output_ptr = nullptr;
    std::size_t x_fft_input_capacity = 0;
    std::size_t y_fft_input_capacity = 0;
    std::size_t x_fft_output_capacity = 0;
    std::size_t y_fft_output_capacity = 0;
    int input_size = 0;
    int x_input_size = 0;
    int y_input_size = 0;
    std::vector<int> x_shape;
    std::vector<int> y_shape;
    std::vector<int> output_shape;
    DeviceArray<int> x_line_bases;
    DeviceArray<int> y_line_bases;
    int x_axis_length = 0;
    int y_axis_length = 0;
    int x_axis_stride = 1;
    int y_axis_stride = 1;
    int output_inner_count = 1;
    int line_count = 1;
    int nperseg = 0;
    int noverlap = 0;
    int nfft = 0;
    int nframes = 0;
    int nf = 0;
    bool return_onesided = true;
    std::string detrend = "constant";
    float scale = 1.0F;

    CsdDeviceWorkspace() = default;
    CsdDeviceWorkspace(int input_size_in, const CsdDeviceParams& params)
    {
        reset(input_size_in, params);
    }

    CsdDeviceWorkspace(
        int x_input_size_in, int y_input_size_in,
        const CsdDeviceParams& params)
    {
        reset(x_input_size_in, y_input_size_in, params);
    }

    void reset(int input_size_in, const CsdDeviceParams& params);
    void reset(
        int x_input_size_in, int y_input_size_in,
        const CsdDeviceParams& params);
    void bind_fft_scratch(
        DeviceArray<ComplexFloat>& x_input_scratch,
        DeviceArray<ComplexFloat>& y_input_scratch,
        DeviceArray<ComplexFloat>& x_output_scratch,
        DeviceArray<ComplexFloat>& y_output_scratch);
    void clear_fft_scratch_bindings();
    [[nodiscard]] ComplexFloat* x_fft_input_data() noexcept { return x_fft_input_ptr; }
    [[nodiscard]] ComplexFloat* y_fft_input_data() noexcept { return y_fft_input_ptr; }
    [[nodiscard]] ComplexFloat* x_fft_output_data() noexcept { return x_fft_output_ptr; }
    [[nodiscard]] ComplexFloat* y_fft_output_data() noexcept { return y_fft_output_ptr; }
    [[nodiscard]] const ComplexFloat* x_fft_input_data() const noexcept { return x_fft_input_ptr; }
    [[nodiscard]] const ComplexFloat* y_fft_input_data() const noexcept { return y_fft_input_ptr; }
    [[nodiscard]] const ComplexFloat* x_fft_output_data() const noexcept { return x_fft_output_ptr; }
    [[nodiscard]] const ComplexFloat* y_fft_output_data() const noexcept { return y_fft_output_ptr; }
    [[nodiscard]] std::size_t fft_scratch_size() const noexcept
    {
        return static_cast<std::size_t>(line_count)
            * static_cast<std::size_t>(nframes) * static_cast<std::size_t>(nfft);
    }
    [[nodiscard]] std::size_t output_size() const noexcept
    {
        return static_cast<std::size_t>(line_count)
            * static_cast<std::size_t>(nf);
    }
};

struct IstftDeviceWorkspace {
    DeviceArray<ComplexFloat> spectrum;
    DeviceArray<ComplexFloat> ifft_output;
    DeviceArray<float> window;
    std::unique_ptr<FFTInterface> fft;
    int frames = 0;
    int bins = 0;
    int nperseg = 0;
    int nfft = 0;
    int hop = 0;
    int output_length = 0;
    int final_output_length = 0;
    bool input_onesided = false;
    bool boundary = false;
    float window_sum = 0.0F;

    IstftDeviceWorkspace() = default;
    IstftDeviceWorkspace(int frames_in, int bins_in, int hop_in)
    {
        reset(frames_in, bins_in, hop_in);
    }

    void reset(int frames_in, int bins_in, int hop_in);
    void reset(
        int frames_in, int frequency_count_in, int nperseg_in,
        int noverlap_in, int nfft_in, bool input_onesided_in,
        bool boundary_in, const WindowParams& window_params);
    [[nodiscard]] std::size_t fft_scratch_size() const noexcept
    {
        return static_cast<std::size_t>(frames) * static_cast<std::size_t>(nfft);
    }
};

enum class SpectralComplexDtype { complex_fp32, complex_fp64, fp64 };

struct SpectralComplexCpuResult {
    SpectralComplexDtype dtype{SpectralComplexDtype::complex_fp32};
    std::vector<double> frequencies;
    std::vector<double> times;
    std::vector<double> fp64;
    std::vector<ComplexFloat> complex_fp32;
    std::vector<ComplexDouble> complex_fp64;
    int frequency_count{0};
    int time_count{0};
    std::vector<int> shape;
};

struct SpectralComplexDeviceResult {
    SpectralComplexDtype dtype{SpectralComplexDtype::complex_fp32};
    DeviceArray<double> frequencies;
    DeviceArray<double> times;
    DeviceArray<double> fp64;
    DeviceArray<ComplexFloat> complex_fp32;
    DeviceArray<ComplexDouble> complex_fp64;
    int frequency_count{0};
    int time_count{0};
    std::vector<int> shape;
};

enum class SpectrogramOutputDtype {
    fp32,
    fp64,
    complex_fp32,
    complex_fp64
};

struct SpectrogramCpuResult {
    SpectrogramOutputDtype dtype{SpectrogramOutputDtype::fp32};
    std::vector<double> frequencies;
    std::vector<double> times;
    std::vector<float> fp32;
    std::vector<double> fp64;
    std::vector<ComplexFloat> complex_fp32;
    std::vector<ComplexDouble> complex_fp64;
    int frequency_count{0};
    int time_count{0};
    std::vector<int> shape;
};

struct SpectrogramDeviceResult {
    SpectrogramOutputDtype dtype{SpectrogramOutputDtype::fp32};
    DeviceArray<double> frequencies;
    DeviceArray<double> times;
    DeviceArray<float> fp32;
    DeviceArray<double> fp64;
    DeviceArray<ComplexFloat> complex_fp32;
    DeviceArray<ComplexDouble> complex_fp64;
    int frequency_count{0};
    int time_count{0};
    std::vector<int> shape;
};

struct IstftDeviceParams {
    float fs = 1.0F;
    WindowParams window;
    int nperseg = 0;
    int noverlap = -1;
    int nfft = 0;
    bool input_onesided = true;
    bool boundary = true;
};

struct IstftCpuResult {
    std::vector<double> times;
    std::vector<float> fp32;
    std::vector<double> fp64;
    bool is_fp64{false};
};

struct IstftDeviceResult {
    DeviceArray<double> times;
    DeviceArray<float> fp32;
    DeviceArray<double> fp64;
    bool is_fp64{false};
};

/**
 * @brief 在 CPU 上计算单段 Cross Spectral Density，作为五类型 GPU convenience 入口 reference。
 * @tparam T FP32、FP16、INT32、INT16 或 INT8 实输入；计算/输出为 FP32 complex。
 * @param x host 一维输入 `[N]`，N 必须至少为2。
 * @param y host 一维输入 `[N]`，长度必须与 x 相同。
 * @return host `[N]`、`ComplexFloat` 的双边频率序列，采用全长 periodic Hann、
 * spectrum scaling、无 detrend、单 frame；布局为连续 frequency-major 一维数组。
 * @throws std::invalid_argument N<2 或输入长度不同。
 * @details CPU direct DFT 仅作 reference，无 workspace/stream/科学计算库；完整分段参数面由
 * workspace device overload 提供。
 * @note 对齐 cuSignal CSD 的单 segment periodic-Hann/two-sided/spectrum 子集。
 */
template <class T> std::vector<ComplexFloat> csd_typed_cpu(
    const std::vector<T>& x, const std::vector<T>& y);

template <class T> SpectralComplexCpuResult csd_typed_cpu(
    const std::vector<T>& x, const std::vector<T>& y,
    const CsdDeviceParams& params);

/**
 * @brief 在 GPU 上按固定 convenience 参数计算单段 CSD。
 * @tparam T 五种业务实输入类型；固定输出 ComplexFloat。
 * @param x 调用者持有的 device 一维输入 `[N]`，N>=2。
 * @param y 调用者持有的 device 一维输入 `[N]`。
 * @param out 调用者预分配的 device 输出 `[N]`。
 * @throws std::invalid_argument 输入 shape 或输出 size 非法；FFT/CUDA 失败走项目异常机制。
 * @details 函数内部临时建立全长 Hann/two-sided/spectrum/mean 的 CSD workspace 和 FFTInterface
 * plan；无 dlrand/隐式 H2D/D2H/外部 stream，资源在返回前释放。重复调用应使用下方 workspace overload。
 * @note 对齐 cuSignal CSD 的单 segment、periodic Hann、two-sided spectrum 子集；不返回
 * frequency 数组，频率坐标由 N 和调用者采样率解释。
 * @note 五类型输入提升为 FP32 complex，ComplexFloat 累加和输出；临时 workspace 生命周期
 * 由 convenience overload 管理。
 */
template <class T> void csd_device(
    const DeviceArray<T>& x, const DeviceArray<T>& y,
    DeviceArray<ComplexFloat>& out);

/**
 * @brief 在 GPU 上使用调用者复用的 workspace 计算分段 CSD。
 * @tparam T 五种业务实输入类型；输入提升为 FP32 complex，输出 ComplexFloat。
 * @param x device row-major 输入，shape 由 `params.x_shape` 描述；空 shape 表示一维。
 * @param y device row-major 输入，shape 由 `params.y_shape` 描述；相关轴外各维按广播规则匹配，
 * 相关轴长度可不同并按固定版 cuSignal 在较短输入尾部补零。
 * @param out device 展平输出，逻辑 shape 为 `workspace.output_shape`；对 nframes 做 mean 平均。
 * @param workspace 调用者初始化并持有的 CSD workspace，包含 Hann/指定 window、四块
 * `[line_count*nframes*nfft]` FFT scratch、广播 line map 和 `FFTInterface` plan；调用期间不得并发共享。
 * @param params 必须与 workspace 一致；average 仅支持 `mean`，detrend 仅 `constant`/空，
 * scaling 仅 `density`/`spectrum`，nfft>=nperseg，noverlap<nperseg；axis 支持完整正负索引。
 * @throws std::invalid_argument 参数、shape、workspace 或输出 size 不一致。
 * @details `return_onesided` 决定 nf=`nfft/2+1` 或 nfft；默认 stream 经 batched
 * 构建时选择的 FFTInterface 后端，无隐式传输。调用者管理 workspace/输入/输出生命周期并在读取前同步。
 * @note 对齐固定版 cuSignal csd 的任意 rank/axis、外维广播及 mean average；正式返回入口
 * 另支持 median。任意 callable detrend 未实现，构造/调用参数不一致会抛异常。
 */
template <class T> void csd_device(
    const DeviceArray<T>& x, const DeviceArray<T>& y,
    DeviceArray<ComplexFloat>& out, CsdDeviceWorkspace& workspace,
    const CsdDeviceParams& params);

/**
 * @brief 正式 CSD 返回入口，同时返回 FP64 频率坐标和动态精度谱。
 * @details INT32 输入按固定版 `result_type(..., complex64)` 返回 ComplexFP64，其他四种
 * 业务输入返回 ComplexFP32；空输入遵循 cuSignal 的无 dtype `cupy.empty` 早退，返回
 * 实数 FP64 空谱。median 在 Device 对逐 frame 交叉谱的实部、虚部分别选择中位数；
 * ComplexFP64 storage finalization 也在 Device 完成，无整段谱的隐式 H2D/D2H。
 */
template <class T> SpectralComplexDeviceResult csd_device(
    const DeviceArray<T>& x, const DeviceArray<T>& y,
    const CsdDeviceParams& params);

/**
 * @brief 在 CPU 上计算固定参数 STFT，作为 convenience GPU 入口 reference。
 * @tparam T 五种业务实输入类型；计算/输出为 FP32 complex。
 * @param x host 一维输入 `[N]`；N<frame 时合法返回空输出。
 * @param frame segment/FFT 长度，必须>=2。
 * @param hop 相邻 frame 步长，必须在 `[1,frame]`。
 * @return host frequency-major `ComplexFloat` 数组 `[frame,nframes]`，其中
 * `nframes=N<frame?0:1+(N-frame)/hop`；periodic Hann、无 boundary/padding/detrend、双边频谱。
 * @throws std::invalid_argument frame/hop 非法。
 * @details CPU direct DFT 仅作 reference，无 workspace/stream/科学计算库。
 * @note 对齐 cuSignal STFT 的 periodic-Hann/two-sided/无 padding 子集。
 */
template <class T> std::vector<ComplexFloat> stft_typed_cpu(
    const std::vector<T>& x, int frame, int hop);

template <class T> SpectralComplexCpuResult stft_typed_cpu(
    const std::vector<T>& x, const StftDeviceParams& params);

/**
 * @brief 在 GPU 上按 frame/hop convenience 参数计算 STFT。
 * @tparam T 五种业务实输入类型；输入提升为 FP32，输出 ComplexFloat。
 * @param x 调用者持有的 device 一维输入 `[N]`。
 * @param frame nperseg=nfft，必须>=2。
 * @param hop 步长，必须在 `[1,frame]`，对应 noverlap=`frame-hop`。
 * @param out 调用者预分配的 device frequency-major 输出 `[frame,nframes]`。
 * @throws std::invalid_argument 参数或输出 shape 非法。
 * @details 固定 periodic Hann、双边、无 boundary/padding/detrend；函数临时创建 workspace
 * 和 FFTInterface plan，不执行隐式传输，不接受外部 stream。重复调用应使用下方 overload。
 * @note 对齐 cuSignal STFT 的 periodic Hann、two-sided、无 padding 子集；不返回频率/时间
 * 坐标数组，输出固定 frequency-major `[frame,nframes]`。
 */
template <class T> void stft_device(
    const DeviceArray<T>& x, int frame, int hop,
    DeviceArray<ComplexFloat>& out);

/**
 * @brief 在 GPU 上使用调用者复用的 workspace 计算 STFT。
 * @tparam T 五种业务实输入类型；固定 ComplexFloat 输出。
 * @param x device row-major 输入，shape 由 `params.shape` 描述；空 shape 表示一维。
 * @param out device 展平输出，逻辑 shape 为 `workspace.output_shape`：原 axis 替换为频率维，
 * 时间维追加到末尾。
 * @param workspace 调用者持有的 real-input STFT workspace、window、FFT scratch 与 FFTInterface plan；
 * 可绑定外部 scratch，但容量必须>=`line_count*nframes*nfft`，同一对象不得并发跨 stream 使用。
 * @param params 必须与 workspace 的 nperseg/noverlap/nfft/onesided/boundary/padded/detrend 一致；
 * axis 支持完整正负索引，boundary 仅 `zeros`/空/`None`，detrend 仅 `constant`/空，nfft>=nperseg。
 * @throws std::invalid_argument 参数、workspace、plan 或输出 shape 不一致。
 * @details `return_onesided` 决定 nf；boundary/padded 决定 nframes。默认 stream 经 batched
 * 构建时选择的 FFTInterface 后端，无隐式 H2D/D2H；调用者管理生命周期和读取同步。
 * @note 对齐固定版 cuSignal stft 的任意 rank/axis 输出布局及受支持
 * boundary/padded/detrend/onesided 参数；其他 boundary 和 callable detrend 明确拒绝。
 * @note 五类型输入在分帧时提升为 FP32 complex，输出固定 ComplexFloat。
 */
template <class T> void stft_device(
    const DeviceArray<T>& x, DeviceArray<ComplexFloat>& out,
    StftDeviceWorkspace& workspace, const StftDeviceParams& params);

/**
 * @brief 正式 STFT 返回入口，返回 `(frequencies, times, Zxx)` 完整三元组。
 * @details 坐标均为 FP64 storage；INT32 的非空 Zxx 为 ComplexFP64，其余四种为
 * ComplexFP32；空输入遵循 cuSignal 返回实数 FP64 空 Zxx。设备计算始终使用
 * ComplexFP32/FFTInterface。
 */
template <class T> SpectralComplexDeviceResult stft_device(
    const DeviceArray<T>& x, const StftDeviceParams& params);

/**
 * @brief 在 CPU 上执行固定双边频谱 ISTFT，作为 GPU reference。
 * @tparam T FP32、FP16、INT32、INT16 或 INT8 复数分量输入类型；实输出固定 FP32。
 * @param z host frequency-major `TypedComplex<T>` 输入 `[bins,frames]`，size 必须为乘积。
 * @param frames frame 数，必须>=1。
 * @param bins 每 frame 的双边 FFT bin 数，必须>=2。
 * @param hop overlap-add 步长，必须在 `[1,bins]`。
 * @return host 一维 FP32 输出，长度 `(frames-1)*hop+bins`；周期 Hann、NOLA 分母小于
 * `1e-7` 的位置确定性写0，不把连续重建结果量化为整数。
 * @throws std::invalid_argument 参数或 z size 非法。
 * @details CPU direct IDFT 仅作 reference，无 workspace/stream/科学计算库；typed 子集固定
 * 双边、frequency-major、无 boundary trimming。
 * @note 对齐 cuSignal ISTFT 的 periodic-Hann overlap-add 子集。
 */
template <class T> std::vector<float> istft_typed_cpu(
    const std::vector<TypedComplex<T>>& z, int frames, int bins, int hop);

/**
 * @brief 完整 ISTFT CPU reference；输入按 `[frequency_count,time_count]` frequency-major。
 * @return `(FP64 time, x)`；INT32 返回 FP64 x，其余四型返回 FP32 x。
 */
template <class T> IstftCpuResult istft_typed_cpu(
    const std::vector<TypedComplex<T>>& z, int frequency_count,
    int time_count, const IstftDeviceParams& params);

template <class T> IstftCpuResult istft_typed_cpu(
    const std::vector<T>& z, int frequency_count,
    int time_count, const IstftDeviceParams& params);

/**
 * @brief 在 GPU 上按 frames/bins/hop convenience 参数执行 ISTFT。
 * @tparam T 五种业务复数分量输入类型；频谱提升为 ComplexFloat，实输出固定 FP32。
 * @param z 调用者持有的 device frequency-major 输入 `[bins,frames]`。
 * @param frames frame 数，必须>=1。
 * @param bins 双边 FFT 长度，必须>=2。
 * @param hop 步长，必须在 `[1,bins]`。
 * @param out 调用者预分配的 device 一维输出，size 为 `(frames-1)*hop+bins`。
 * @throws std::invalid_argument 参数、输入或输出 shape 非法；FFT/CUDA 失败走项目异常机制。
 * @details 函数内部临时创建 Hann、两块 `[frames*bins]` scratch 和 FFTInterface plan；
 * 无 dlrand/隐式传输/外部 stream。重复调用应使用下方 workspace overload。
 * @note 对齐 cuSignal ISTFT 的 periodic Hann、双边 frequency-major overlap-add 子集；
 * 不返回 time 坐标，不接受 boundary/onesided/time-axis 参数。
 * @note 频谱提升为 FP32 complex，IFFT/overlap-add accumulator 为 FP32；内部 workspace
 * 由 convenience overload 创建并释放。
 */
template <class T> void istft_device(
    const DeviceArray<TypedComplex<T>>& z, int frames, int bins, int hop,
    DeviceArray<float>& out);

/**
 * @brief 在 GPU 上使用调用者复用的 workspace 执行 ISTFT。
 * @tparam T 五种业务复数分量输入类型；FP32 IFFT/overlap-add 后固定输出 FP32。
 * @param z device frequency-major 输入 `[workspace.bins,workspace.frames]`。
 * @param out device 一维输出 `[workspace.output_length]`。
 * @param workspace 调用者持有且已按 frames/bins/hop 初始化的对象，拥有 spectrum、IFFT
 * scratch、periodic Hann 和 FFTInterface plan；同一对象不得并发共享。
 * @throws std::invalid_argument 输入/输出 shape 不匹配或 FFT plan 未初始化。
 * @details 默认 stream 执行 spectrum build、batched IFFT 和 FP32 overlap-add；无隐式
 * H2D/D2H。函数不接受 boundary/onesided/time-axis 选项，这是相对完整 cuSignal ISTFT 的明确子集。
 */
template <class T> void istft_device(
    const DeviceArray<TypedComplex<T>>& z, DeviceArray<float>& out,
    IstftDeviceWorkspace& workspace);

template <class T> void istft_device(
    const DeviceArray<T>& z, DeviceArray<float>& out,
    IstftDeviceWorkspace& workspace);

/**
 * @brief 完整 ISTFT GPU 正式入口，支持单双边、nperseg/noverlap/nfft 和 boundary trim。
 * @details GPU 执行 ComplexFP32 IFFT/FP32 overlap-add；INT32 的 FP64 契约输出由 Device
 * storage finalizer 直接生成，不执行 device FP64 数学或 Host 往返。
 */
template <class T> IstftDeviceResult istft_device(
    const DeviceArray<TypedComplex<T>>& z, int frequency_count,
    int time_count, const IstftDeviceParams& params);

template <class T> IstftDeviceResult istft_device(
    const DeviceArray<T>& z, int frequency_count,
    int time_count, const IstftDeviceParams& params);

/**
 * @brief 在 CPU 上计算固定参数 PSD Spectrogram，作为 convenience GPU reference。
 * @tparam T 五种业务实输入类型；计算和输出为 FP32。
 * @param x host 一维输入 `[N]`；N<frame 时合法返回空数组。
 * @param frame segment/FFT 长度，必须>=2。
 * @param hop frame 步长，必须在 `[1,frame]`。
 * @return host frequency-major FP32 PSD `[frame,nframes]`，periodic Hann、双边、
 * spectrum scaling、无 detrend/boundary/padding。
 * @throws std::invalid_argument frame/hop 非法。
 * @details CPU direct DFT 仅作 reference，无 workspace/stream/科学计算库。
 * @note 对齐 cuSignal spectrogram 的 PSD/periodic-Hann/two-sided/spectrum 子集。
 */
template <class T> std::vector<float> spectrogram_typed_cpu(
    const std::vector<T>& x, int frame, int hop);

/**
 * @brief 完整 Spectrogram CPU reference，返回 FP64 frequency/time 与 mode 对应结果。
 * @details 支持任意 rank/axis 及 psd、complex、magnitude、angle、phase；原 axis 替换为
 * 频率维，时间维追加到末尾，phase 沿频率轴 unwrap。
 */
template <class T> SpectrogramCpuResult spectrogram_typed_cpu(
    const std::vector<T>& x, const SpectrogramDeviceParams& params);

/**
 * @brief 在 GPU 上按 frame/hop convenience 参数计算 PSD Spectrogram。
 * @tparam T 五种业务实输入类型；固定 FP32 输出。
 * @param x 调用者持有的 device 一维输入 `[N]`。
 * @param frame nperseg=nfft，必须>=2。
 * @param hop 步长，必须在 `[1,frame]`。
 * @param out 调用者预分配的 device frequency-major 输出 `[frame,nframes]`。
 * @throws std::invalid_argument 参数或输出 shape 非法。
 * @details 固定 Hann/two-sided/spectrum/psd，无 detrend；函数临时建立 workspace 和
 * FFTInterface plan，无隐式传输/外部 stream。重复调用应使用下方 overload。
 * @note 对齐 cuSignal spectrogram 的 PSD、periodic Hann、two-sided spectrum 子集；
 * 不返回 frequency/time 坐标，mode/window/scaling 扩展使用下方 workspace overload。
 */
template <class T> void spectrogram_device(
    const DeviceArray<T>& x, int frame, int hop, DeviceArray<float>& out);

/**
 * @brief 在 GPU 上使用调用者复用的 workspace 计算 Spectrogram。
 * @tparam T 五种业务实输入类型；输出固定 FP32。
 * @param x device row-major 输入，shape 由 `params.shape` 描述；空 shape 表示一维。
 * @param out device 展平输出，逻辑 shape 为 `workspace.stft.output_shape`。
 * @param workspace 调用者持有的对象，复用 STFT window/scratch/FFTInterface plan 并保存后处理 scale；
 * 同一对象不得并发共享。
 * @param params 构造 workspace 的参数：mode 支持 `psd`/`magnitude`/`angle`/`phase`，typed
 * real-output overload 明确拒绝 `complex`；scaling 仅 `density`/`spectrum`，axis 支持完整正负索引，
 * detrend 仅 `constant`/空，return_onesided 控制 nf。
 * @throws std::invalid_argument mode、参数、workspace 或输出 shape 非法。
 * @details 默认 stream 经所选 batched FFTInterface 后端和 FP32 后处理；phase unwrap 在
 * Device 上沿频率轴顺序执行，无隐式传输。
 * @note 对齐固定版 cuSignal spectrogram 的任意 rank/axis、输出布局和五种 mode；正式入口
 * 对 `phase` 沿频率轴执行 unwrap。
 */
template <class T> void spectrogram_device(
    const DeviceArray<T>& x, DeviceArray<float>& out,
    SpectrogramDeviceWorkspace& workspace,
    const SpectrogramDeviceParams& params);

/**
 * @brief 完整 Spectrogram GPU 正式入口，返回 `(frequencies,times,Sxx)`。
 * @details INT32 的非空实/复结果为 FP64/ComplexFP64，其余四型为 FP32/ComplexFP32；
 * 任意输入 dtype 的空结果均遵循 cuSignal 返回实数 FP64 空数组。设备端统一 FP32
 * 计算，phase unwrap 和 FP64 storage finalization 均在 Device 完成。
 */
template <class T> SpectrogramDeviceResult spectrogram_device(
    const DeviceArray<T>& x, const SpectrogramDeviceParams& params);

/**
 * @brief 在 CPU 上计算 Lomb-Scargle periodogram，作为 GPU reference。
 * @tparam T FP32、FP16、INT32、INT16 或 INT8；三组输入同型。
 * @param time host 一维采样时刻 `[N]`，必须非空；单位与 frequency 的倒数对应。
 * @param value host 一维观测值 `[N]`，长度必须与 time 相同。
 * @param frequency host 一维角频率 `[F]`，单位 rad/time；允许为空并返回空输出。
 * @param precenter true 时先减去 value 均值，默认 false。
 * @param normalize true 时按原始 value 的平方和乘 `2/dot(value,value)`；零平方和保持系数1。
 * @return host `[F]`、FP64 的非均匀采样 periodogram。
 * @throws std::invalid_argument time 为空或 time/value 长度不同。
 * @details CPU direct 三角归约仅作 reference；无 workspace/stream/FFTInterface/dlrand。typed 子集
 * 不接受 weights/floating_mean 等扩展参数。
 * @note 对齐 cuSignal lombscargle 的 precenter/normalize 基本语义。
 */
template <class T> std::vector<double> lombscargle_typed_cpu(
    const std::vector<T>& time, const std::vector<T>& value,
    const std::vector<T>& frequency, bool precenter = false,
    bool normalize = false);

/**
 * @brief 在 GPU 上计算 Lomb-Scargle periodogram。
 * @tparam T 五种业务输入类型；正式输出为 FP64。
 * @param time 调用者持有的 device 一维时刻 `[N]`，必须非空。
 * @param value 调用者持有的 device 一维观测 `[N]`。
 * @param frequency 调用者持有的 device 角频率 `[F]`；F=0 合法且不启动 kernel。
 * @param precenter 是否预中心化，默认 false。
 * @param normalize 是否按信号功率归一化，默认 false。
 * @return device FP64 输出 `[F]`。GPU 以 FP32 计算并由 Device finalizer 生成 FP64 storage。
 * @throws std::invalid_argument time/value shape 非法；launch 失败走 CUDA 异常。
 * @details 每个 frequency 使用一个线程的 custom FP32 kernel；为遵守设备端禁用 FP64，
 * 正式 FP64 输出只做设备端位编码，不包含 D2H/Host 扩宽/H2D；不提供复用 workspace。
 * @note 对齐 cuSignal lombscargle 的 precenter/normalize 基本语义；weights、floating_mean
 * 和自动频率网格未实现，frequency 必须由调用者以角频率显式提供。
 */
template <class T> DeviceArray<double> lombscargle_device(
    const DeviceArray<T>& time, const DeviceArray<T>& value,
    const DeviceArray<T>& frequency,
    bool precenter = false, bool normalize = false);

struct VectorStrengthCpuResult {
    std::vector<double> strength;
    std::vector<double> phase;
    bool scalar_period{false};
};

struct VectorStrengthDeviceResult {
    DeviceArray<double> strength;
    DeviceArray<double> phase;
    bool scalar_period{false};
};

/**
 * @brief 在 CPU 上计算事件相对于给定周期的 vector strength reference。
 * @tparam T FP32、FP16、INT32、INT16 或 INT8。
 * @param events host 一维事件时刻 `[N]`；空输入按 cuSignal 返回 NaN strength/phase。
 * @param period 正周期，dtype T，转换后必须大于0。
 * @return FP64 strength 与 mean phase；标量 period 保持标量形态标记。
 * @throws std::invalid_argument period<=0。
 * @details 无 workspace/stream/科学计算库；同时支持标量和一维 period 数组。
 */
template <class T> VectorStrengthCpuResult vectorstrength_typed_cpu(
    const std::vector<T>& events, T period);

template <class T> VectorStrengthCpuResult vectorstrength_typed_cpu(
    const std::vector<T>& events, const std::vector<T>& periods);

/**
 * @brief 在 GPU 上计算标量或一维周期数组的 vector strength 与 mean phase。
 * @tparam T 五种业务输入类型；正式输出为 FP64。
 * @param events 调用者持有的 device 一维事件时刻 `[N]`；允许为空。
 * @param period 正周期，单位与 events 相同。
 * @return device FP64 strength/phase；标量 period 保持标量形态标记。
 * @throws std::invalid_argument 任一 period<=0。
 * @details 每个 period 使用一个 block 的 FP32 reduction；两项 FP64 storage 输出均由
 * Device finalizer 生成。period 数组合法性检查仍会 D2H 读取输入，不提供外部 stream 或
 * 复用 workspace；输出收尾不再发生 Host 往返。
 * @note 对齐固定版 cuSignal 的 strength、phase、标量/数组 period 与空 events 语义。
 */
template <class T> VectorStrengthDeviceResult vectorstrength_device(
    const DeviceArray<T>& events, T period);

template <class T> VectorStrengthDeviceResult vectorstrength_device(
    const DeviceArray<T>& events, const DeviceArray<T>& periods);

}  // namespace cusignal
