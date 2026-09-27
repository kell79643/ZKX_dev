#include <cusignal/operators/spectral_analysis/spectral_analysis_typed.h>
#include "spectral_analysis_kernels.cuh"
#include <cusignal/runtime/cuda_utils.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cusignal {

namespace {

constexpr float kPi = 3.14159265358979323846F;
constexpr float kTwoPi = 2.0F * kPi;

float bessel_i0(float x)
{
    const float ax = std::abs(x);
    if (ax < 3.75F) {
        float y = x / 3.75F;
        y *= y;
        return 1.0F + y * (3.5156229F + y * (3.0899424F + y * (1.2067492F
            + y * (0.2659732F + y * (0.0360768F + y * 0.0045813F)))));
    }
    const float y = 3.75F / ax;
    return (std::exp(ax) / std::sqrt(ax)) * (0.39894228F + y * (0.01328592F
        + y * (0.00225319F + y * (-0.00157565F + y * (0.00916281F
        + y * (-0.02057706F + y * (0.02635537F
        + y * (-0.01647633F + y * 0.00392377F))))))));
}

std::vector<float> make_stft_window(const WindowParams& params, int size)
{
    if (size <= 0) {
        return {};
    }
    if (!params.custom_window.empty()) {
        if (static_cast<int>(params.custom_window.size()) != size) {
            throw std::invalid_argument(
                "spectral workspace: custom window length must equal nperseg");
        }
        return params.custom_window;
    }

    const int periodic_size = size + 1;
    const float denominator = static_cast<float>(periodic_size - 1);
    const std::string type = params.type.empty() ? "hann" : params.type;
    std::vector<float> window(size, 1.0F);
    if (type == "hamming") {
        for (int i = 0; i < size; ++i) {
            window[i] = 0.54F - 0.46F * std::cos(kTwoPi * i / denominator);
        }
    } else if (type == "hann" || type == "hanning") {
        for (int i = 0; i < size; ++i) {
            window[i] = 0.5F * (1.0F - std::cos(kTwoPi * i / denominator));
        }
    } else if (type == "kaiser") {
        const float beta = params.param > 0.0F ? params.param : 5.0F;
        const float alpha = 0.5F * denominator;
        const float inverse_i0 = 1.0F / bessel_i0(beta);
        for (int i = 0; i < size; ++i) {
            const float ratio = (static_cast<float>(i) - alpha) / alpha;
            window[i] = ratio * ratio <= 1.0F
                ? bessel_i0(beta * std::sqrt(1.0F - ratio * ratio)) * inverse_i0
                : 0.0F;
        }
    } else if (type == "boxcar" || type == "rectangular") {
        std::fill(window.begin(), window.end(), 1.0F);
    } else if (type == "blackman") {
        for (int i = 0; i < size; ++i) {
            window[i] = 0.42F - 0.5F * std::cos(kTwoPi * i / denominator)
                + 0.08F * std::cos(4.0F * kPi * i / denominator);
        }
    } else if (type == "tukey") {
        const float alpha = params.param;
        if (alpha <= 0.0F) {
            std::fill(window.begin(), window.end(), 1.0F);
        } else if (alpha >= 1.0F) {
            for (int i = 0; i < size; ++i) {
                window[i] = 0.5F * (1.0F - std::cos(kTwoPi * i / denominator));
            }
        } else {
            const float edge = alpha * denominator / 2.0F;
            for (int i = 0; i < size; ++i) {
                if (i < edge) {
                    window[i] = 0.5F * (1.0F + std::cos(
                        kPi * (-1.0F + 2.0F * i / (alpha * denominator))));
                } else if (i > denominator - edge) {
                    window[i] = 0.5F * (1.0F + std::cos(kPi * (
                        -2.0F / alpha + 1.0F
                        + 2.0F * i / (alpha * denominator))));
                }
            }
        }
    } else if (type == "bartlett" || type == "triang") {
        for (int i = 0; i < size; ++i) {
            const float fraction = static_cast<float>(i) / denominator;
            window[i] = 2.0F * (fraction < 0.5F ? fraction : 1.0F - fraction);
        }
    } else {
        throw std::invalid_argument("spectral workspace: unsupported window type");
    }
    return window;
}

void normalize_stft_sizes(
    const StftDeviceParams& params, int& nperseg, int& noverlap, int& nfft)
{
    nperseg = params.nperseg;
    if (!params.window.custom_window.empty() && nperseg <= 0) {
        nperseg = static_cast<int>(params.window.custom_window.size());
    }
    if (nperseg <= 0) {
        nperseg = 256;
    }
    noverlap = params.noverlap < 0 ? nperseg / 2 : params.noverlap;
    nfft = params.nfft <= 0 ? nperseg : params.nfft;
}

int positive_mod(int value, int modulus)
{
    const int remainder = value % modulus;
    return remainder < 0 ? remainder + modulus : remainder;
}

int boundary_mode(const std::string& boundary)
{
    if (boundary.empty() || boundary == "None" || boundary == "zeros") return 0;
    if (boundary == "constant") return 1;
    if (boundary == "even") return 2;
    if (boundary == "odd") return 3;
    throw std::invalid_argument("unsupported boundary");
}

int stft_frame_count(
    int input_size, int nperseg, int noverlap,
    const std::string& boundary, bool padded)
{
    if (input_size <= 0) {
        return 0;
    }
    const int step = nperseg - noverlap;
    const int boundary_pad = (boundary.empty() || boundary == "None") ? 0 : nperseg;
    int extended_size = input_size + boundary_pad;
    if (padded) {
        extended_size += positive_mod(-(extended_size - nperseg), step) % nperseg;
    }
    return extended_size < nperseg ? 0 : (extended_size - nperseg) / step + 1;
}

}  // namespace

void StftDeviceWorkspace::reset(
    int input_size_in, const StftDeviceParams& params, bool complex_input_in)
{
    if (input_size_in < 0) {
        throw std::invalid_argument("StftDeviceWorkspace: negative input size");
    }
    (void)boundary_mode(params.boundary);
    if (params.detrend != "constant" && !params.detrend.empty()) {
        throw std::invalid_argument("StftDeviceWorkspace: unsupported detrend");
    }
    if (complex_input_in && params.return_onesided) {
        throw std::invalid_argument("StftDeviceWorkspace: complex input must be two-sided");
    }

    input_size = input_size_in;
    const auto input_plan = spectral_layout::prepare_axis_plan(
        static_cast<std::size_t>(input_size), params.shape, params.axis,
        1, 0, false);
    input_shape = input_plan.input_shape;
    axis = input_plan.axis;
    axis_length = input_plan.axis_length;
    axis_stride = input_plan.axis_stride;
    inner_count = input_plan.inner_count;
    outer_count = input_plan.outer_count;
    line_count = input_plan.line_count;
    line_bases = DeviceArray<int>::from_host(input_plan.line_bases);
    normalize_stft_sizes(params, nperseg, noverlap, nfft);
    if (nfft < nperseg || noverlap < 0 || noverlap >= nperseg) {
        throw std::invalid_argument("StftDeviceWorkspace: invalid segment sizes");
    }
    return_onesided = params.return_onesided;
    complex_input = complex_input_in;
    boundary = params.boundary;
    padded = params.padded;
    detrend = params.detrend;
    nframes = stft_frame_count(axis_length, nperseg, noverlap, boundary, padded);
    nf = return_onesided ? nfft / 2 + 1 : nfft;
    output_shape = spectral_layout::prepare_axis_plan(
        static_cast<std::size_t>(input_size), params.shape, params.axis,
        nf, nframes, true).output_shape;

    const std::vector<float> host_window = make_stft_window(params.window, nperseg);
    compute_policy::CompensatedFloatAccumulator window_sum_acc;
    for (float value : host_window) {
        window_sum_acc.add(value);
    }
    const float window_sum = window_sum_acc.value();
    scale = std::abs(window_sum) > 1.0e-7F ? 1.0F / window_sum : 1.0F;
    window = DeviceArray<float>::from_host(host_window);

    const std::size_t scratch_size = fft_scratch_size();
    fft_input.reset(scratch_size);
    fft_output.reset(scratch_size);
    clear_fft_scratch_bindings();
    fft = nframes == 0 ? nullptr : std::make_unique<FFTInterface>(
        nfft, FFTInterface::BatchOnly{});
}

void StftDeviceWorkspace::bind_fft_scratch(
    DeviceArray<ComplexFloat>& input_scratch,
    DeviceArray<ComplexFloat>& output_scratch)
{
    const std::size_t required = fft_scratch_size();
    if (input_scratch.size() < required || output_scratch.size() < required) {
        throw std::invalid_argument("StftDeviceWorkspace: external scratch is too small");
    }
    fft_input_ptr = input_scratch.data();
    fft_output_ptr = output_scratch.data();
    fft_input_capacity = input_scratch.size();
    fft_output_capacity = output_scratch.size();
}

void StftDeviceWorkspace::clear_fft_scratch_bindings()
{
    fft_input_ptr = fft_input.data();
    fft_output_ptr = fft_output.data();
    fft_input_capacity = fft_input.size();
    fft_output_capacity = fft_output.size();
}

void SpectrogramDeviceWorkspace::reset(
    int input_size_in, const SpectrogramDeviceParams& params)
{
    if (params.mode != "psd" && params.mode != "complex"
        && params.mode != "magnitude" && params.mode != "angle"
        && params.mode != "phase") {
        throw std::invalid_argument("SpectrogramDeviceWorkspace: unsupported mode");
    }
    if (params.fs <= 0.0F) {
        throw std::invalid_argument("SpectrogramDeviceWorkspace: fs must be positive");
    }

    StftDeviceParams stft_params;
    stft_params.fs = params.fs;
    stft_params.window = params.window;
    stft_params.nperseg = params.nperseg;
    const int default_nperseg = params.nperseg > 0
        ? params.nperseg
        : (!params.window.custom_window.empty()
            ? static_cast<int>(params.window.custom_window.size()) : 256);
    stft_params.noverlap = params.noverlap < 0
        ? default_nperseg / 8
        : params.noverlap;
    stft_params.nfft = params.nfft;
    stft_params.detrend = params.detrend;
    stft_params.return_onesided = params.return_onesided;
    stft_params.boundary = "";
    stft_params.padded = false;
    stft_params.axis = params.axis;
    stft_params.shape = params.shape;
    stft.reset(input_size_in, stft_params, false);

    const std::vector<float> host_window = make_stft_window(params.window, stft.nperseg);
    compute_policy::CompensatedFloatAccumulator square_sum_acc;
    compute_policy::CompensatedFloatAccumulator sum_acc;
    for (float value : host_window) {
        square_sum_acc.add_product(value, value);
        sum_acc.add(value);
    }
    const float square_sum = square_sum_acc.value();
    const float sum = sum_acc.value();
    if (params.mode == "psd") {
        if (params.scaling == "density") {
            post_scale = 1.0F / (params.fs * square_sum);
        } else if (params.scaling == "spectrum") {
            post_scale = 1.0F / (sum * sum);
        } else {
            throw std::invalid_argument("SpectrogramDeviceWorkspace: unsupported scaling");
        }
        stft.scale = 1.0F;
    } else {
        if (params.scaling == "density") {
            stft.scale = std::sqrt(1.0F / (params.fs * square_sum));
        } else if (params.scaling == "spectrum") {
            stft.scale = std::abs(sum) > 1.0e-7F ? 1.0F / sum : 1.0F;
        } else {
            throw std::invalid_argument("SpectrogramDeviceWorkspace: unsupported scaling");
        }
        post_scale = 1.0F;
    }
}

void CsdDeviceWorkspace::reset(
    int input_size_in, const CsdDeviceParams& params)
{
    reset(input_size_in, input_size_in, params);
}

void CsdDeviceWorkspace::reset(
    int x_input_size_in, int y_input_size_in,
    const CsdDeviceParams& params)
{
    if (x_input_size_in < 0 || y_input_size_in < 0) {
        throw std::invalid_argument("CsdDeviceWorkspace: invalid input size");
    }
    if (params.average != "mean") {
        throw std::invalid_argument("CsdDeviceWorkspace: unsupported average");
    }
    if (params.detrend != "constant" && !params.detrend.empty()) {
        throw std::invalid_argument("CsdDeviceWorkspace: unsupported detrend");
    }
    if (params.fs <= 0.0F) {
        throw std::invalid_argument("CsdDeviceWorkspace: fs must be positive");
    }

    x_input_size = x_input_size_in;
    y_input_size = y_input_size_in;
    input_size = std::max(x_input_size, y_input_size);
    const auto input_plan = spectral_layout::prepare_csd_plan(
        static_cast<std::size_t>(x_input_size),
        static_cast<std::size_t>(y_input_size),
        params.x_shape, params.y_shape, params.axis, 1);
    x_shape = input_plan.x_shape;
    y_shape = input_plan.y_shape;
    x_axis_length = input_plan.x_axis_length;
    y_axis_length = input_plan.y_axis_length;
    x_axis_stride = input_plan.x_axis_stride;
    y_axis_stride = input_plan.y_axis_stride;
    line_count = input_plan.line_count;
    output_inner_count = input_plan.output_inner_count;
    x_line_bases = DeviceArray<int>::from_host(input_plan.x_line_bases);
    y_line_bases = DeviceArray<int>::from_host(input_plan.y_line_bases);
    nperseg = params.nperseg;
    if (!params.window.custom_window.empty() && nperseg <= 0) {
        nperseg = static_cast<int>(params.window.custom_window.size());
    }
    if (nperseg <= 0) {
        nperseg = 256;
    }
    noverlap = params.noverlap < 0 ? nperseg / 2 : params.noverlap;
    nfft = params.nfft <= 0 ? nperseg : params.nfft;
    if (nfft < nperseg || noverlap < 0 || noverlap >= nperseg) {
        throw std::invalid_argument("CsdDeviceWorkspace: invalid segment sizes");
    }
    const int spectral_length = std::max(x_axis_length, y_axis_length);
    nframes = spectral_length == 0 ? 0
        : (spectral_length < nperseg ? 1
            : (spectral_length - nperseg) / (nperseg - noverlap) + 1);
    nf = params.return_onesided ? nfft / 2 + 1 : nfft;
    const auto output_plan = spectral_layout::prepare_csd_plan(
        static_cast<std::size_t>(x_input_size),
        static_cast<std::size_t>(y_input_size),
        params.x_shape, params.y_shape, params.axis, nf);
    output_shape = output_plan.output_shape;
    output_inner_count = output_plan.output_inner_count;
    return_onesided = params.return_onesided;
    detrend = params.detrend;

    const std::vector<float> host_window = make_stft_window(params.window, nperseg);
    compute_policy::CompensatedFloatAccumulator square_sum_acc;
    compute_policy::CompensatedFloatAccumulator sum_acc;
    for (float value : host_window) {
        square_sum_acc.add_product(value, value);
        sum_acc.add(value);
    }
    const float square_sum = square_sum_acc.value();
    const float sum = sum_acc.value();
    if (params.scaling == "density") {
        scale = 1.0F / (params.fs * square_sum);
    } else if (params.scaling == "spectrum") {
        scale = 1.0F / (sum * sum);
    } else {
        throw std::invalid_argument("CsdDeviceWorkspace: unsupported scaling");
    }
    window = DeviceArray<float>::from_host(host_window);

    const std::size_t scratch_size = fft_scratch_size();
    x_fft_input.reset(scratch_size);
    y_fft_input.reset(scratch_size);
    x_fft_output.reset(scratch_size);
    y_fft_output.reset(scratch_size);
    clear_fft_scratch_bindings();
    fft = nframes == 0 ? nullptr : std::make_unique<FFTInterface>(
        nfft, FFTInterface::BatchOnly{});
}

void CsdDeviceWorkspace::bind_fft_scratch(
    DeviceArray<ComplexFloat>& x_input_scratch,
    DeviceArray<ComplexFloat>& y_input_scratch,
    DeviceArray<ComplexFloat>& x_output_scratch,
    DeviceArray<ComplexFloat>& y_output_scratch)
{
    const std::size_t required = fft_scratch_size();
    if (x_input_scratch.size() < required || y_input_scratch.size() < required
        || x_output_scratch.size() < required || y_output_scratch.size() < required) {
        throw std::invalid_argument("CsdDeviceWorkspace: external scratch is too small");
    }
    x_fft_input_ptr = x_input_scratch.data();
    y_fft_input_ptr = y_input_scratch.data();
    x_fft_output_ptr = x_output_scratch.data();
    y_fft_output_ptr = y_output_scratch.data();
    x_fft_input_capacity = x_input_scratch.size();
    y_fft_input_capacity = y_input_scratch.size();
    x_fft_output_capacity = x_output_scratch.size();
    y_fft_output_capacity = y_output_scratch.size();
}

void CsdDeviceWorkspace::clear_fft_scratch_bindings()
{
    x_fft_input_ptr = x_fft_input.data();
    y_fft_input_ptr = y_fft_input.data();
    x_fft_output_ptr = x_fft_output.data();
    y_fft_output_ptr = y_fft_output.data();
    x_fft_input_capacity = x_fft_input.size();
    y_fft_input_capacity = y_fft_input.size();
    x_fft_output_capacity = x_fft_output.size();
    y_fft_output_capacity = y_fft_output.size();
}

void csd_median_fp32_device(
    const CsdDeviceWorkspace& workspace,
    DeviceArray<ComplexFloat>& output)
{
    if (output.size() != workspace.output_size()) {
        throw std::invalid_argument("csd median output size");
    }
    if (workspace.nframes <= 0 || output.empty()) {
        return;
    }
    if (output.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || output.size() > std::numeric_limits<std::size_t>::max()
            / static_cast<std::size_t>(workspace.nframes)) {
        throw std::overflow_error("csd median scratch size");
    }
    const std::size_t value_count = output.size()
        * static_cast<std::size_t>(workspace.nframes);
    if (value_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("csd median kernel indexing");
    }
    DeviceArray<float> real_values(value_count);
    DeviceArray<float> imag_values(value_count);
    cuda_utils::launch_1d_kernel(
        spectral_detail::csd_median_materialize_kernel,
        value_count,
        workspace.x_fft_output_data(), workspace.y_fft_output_data(),
        workspace.nfft, workspace.nf, workspace.nframes,
        workspace.line_count, workspace.output_inner_count,
        workspace.scale, workspace.return_onesided,
        real_values.data(), imag_values.data());
    float bias = 1.0F;
    for (int index = 1; index <= (workspace.nframes - 1) / 2; ++index) {
        const float even = static_cast<float>(2 * index);
        bias += 1.0F / (even + 1.0F) - 1.0F / even;
    }
    cuda_utils::launch_1d_kernel(
        spectral_detail::csd_median_reduce_kernel,
        output.size(), real_values.data(), imag_values.data(),
        workspace.nframes, static_cast<int>(output.size()), bias,
        output.data());
}

void IstftDeviceWorkspace::reset(int frames_in,int bins_in,int hop_in){WindowParams window_params("hann");reset(frames_in,bins_in,bins_in,bins_in-hop_in,bins_in,false,false,window_params);}
void IstftDeviceWorkspace::reset(int frames_in,int frequency_count_in,int nperseg_in,int noverlap_in,int nfft_in,bool input_onesided_in,bool boundary_in,const WindowParams&window_params){if(frames_in<1||frequency_count_in<1||nperseg_in<1||nfft_in<nperseg_in||noverlap_in<0||noverlap_in>=nperseg_in)throw std::invalid_argument("istft workspace shape");const int expected_frequency_count=input_onesided_in?nfft_in/2+1:nfft_in;if(frequency_count_in!=expected_frequency_count)throw std::invalid_argument("istft frequency shape");frames=frames_in;bins=frequency_count_in;nperseg=nperseg_in;nfft=nfft_in;hop=nperseg-noverlap_in;input_onesided=input_onesided_in;boundary=boundary_in;output_length=nperseg+(frames-1)*hop;final_output_length=boundary?std::max(0,output_length-2*(nperseg/2)):output_length;spectrum.reset(fft_scratch_size());ifft_output.reset(fft_scratch_size());std::vector<float>host_window=make_stft_window(window_params,nperseg);window_sum=0.0F;for(float value:host_window)window_sum+=value;window=DeviceArray<float>::from_host(host_window);fft=std::make_unique<FFTInterface>(nfft,FFTInterface::BatchOnly{});}

template<class T>void csd_device(const DeviceArray<T>&x,const DeviceArray<T>&y,DeviceArray<ComplexFloat>&o,CsdDeviceWorkspace&workspace,const CsdDeviceParams&params){static_assert(detail::is_simple_signal_input_v<T>);if(params.average!="mean"||workspace.x_input_size!=(int)x.size()||workspace.y_input_size!=(int)y.size()||o.size()!=workspace.output_size())throw std::invalid_argument("csd shape/workspace");if(x.empty()||y.empty()||workspace.nframes==0)return;const int step=workspace.nperseg-workspace.noverlap;const std::size_t total=workspace.fft_scratch_size();cuda_utils::launch_1d_kernel(spectral_detail::stft_prepare_real_kernel<T,float,float>,total,x.data(),workspace.x_line_bases.data(),workspace.x_axis_length,workspace.x_axis_stride,workspace.line_count,workspace.window.data(),workspace.nperseg,workspace.nfft,step,workspace.nframes,0,0,workspace.detrend=="constant",1.0F,workspace.x_fft_input_data());cuda_utils::launch_1d_kernel(spectral_detail::stft_prepare_real_kernel<T,float,float>,total,y.data(),workspace.y_line_bases.data(),workspace.y_axis_length,workspace.y_axis_stride,workspace.line_count,workspace.window.data(),workspace.nperseg,workspace.nfft,step,workspace.nframes,0,0,workspace.detrend=="constant",1.0F,workspace.y_fft_input_data());const int batches=workspace.line_count*workspace.nframes;workspace.fft->fft_batch_device(workspace.x_fft_input_data(),workspace.x_fft_output_data(),batches,total);workspace.fft->fft_batch_device(workspace.y_fft_input_data(),workspace.y_fft_output_data(),batches,total);cuda_utils::launch_1d_kernel(spectral_detail::csd_reduce_kernel<float,ComplexFloat>,workspace.output_size(),workspace.x_fft_output_data(),workspace.y_fft_output_data(),workspace.nfft,workspace.nf,workspace.nframes,workspace.line_count,workspace.output_inner_count,static_cast<float>(workspace.scale),workspace.return_onesided,o.data());}
template<class T>void csd_device(const DeviceArray<T>&x,const DeviceArray<T>&y,DeviceArray<ComplexFloat>&o){if(x.size()<2||x.size()!=y.size())throw std::invalid_argument("csd shape");CsdDeviceParams params;params.window=WindowParams("hann");params.nperseg=(int)x.size();params.noverlap=0;params.nfft=(int)x.size();params.detrend="";params.return_onesided=false;params.scaling="spectrum";params.average="mean";CsdDeviceWorkspace workspace((int)x.size(),params);csd_device(x,y,o,workspace,params);}
template<class T>void stft_device(const DeviceArray<T>&x,DeviceArray<ComplexFloat>&o,StftDeviceWorkspace&workspace,const StftDeviceParams&params){static_assert(detail::is_simple_signal_input_v<T>);const int pn=params.nperseg<=0?256:params.nperseg;const int po=params.noverlap<0?pn/2:params.noverlap;const int pf=params.nfft<=0?pn:params.nfft;if(workspace.complex_input||workspace.input_size!=(int)x.size()||workspace.nperseg!=pn||workspace.noverlap!=po||workspace.nfft!=pf||workspace.return_onesided!=params.return_onesided||workspace.boundary!=params.boundary||workspace.padded!=params.padded||workspace.detrend!=params.detrend||o.size()!=workspace.output_size())throw std::invalid_argument("stft shape/workspace");if(x.empty())return;if(!workspace.fft)throw std::invalid_argument("stft FFT plan");const int step=workspace.nperseg-workspace.noverlap;const int boundary_left=(workspace.boundary.empty()||workspace.boundary=="None")?0:workspace.nperseg/2;cuda_utils::launch_1d_kernel(spectral_detail::stft_prepare_real_kernel<T,float,float>,workspace.fft_scratch_size(),x.data(),workspace.line_bases.data(),workspace.axis_length,workspace.axis_stride,workspace.line_count,workspace.window.data(),workspace.nperseg,workspace.nfft,step,workspace.nframes,boundary_left,boundary_mode(workspace.boundary),workspace.detrend=="constant",static_cast<float>(workspace.scale),workspace.fft_input_data());const int batches=workspace.line_count*workspace.nframes;workspace.fft->fft_batch_device(workspace.fft_input_data(),workspace.fft_output_data(),batches,workspace.fft_scratch_size());cuda_utils::launch_1d_kernel(spectral_detail::stft_pack_output_kernel<ComplexFloat>,workspace.output_size(),workspace.fft_output_data(),workspace.nfft,workspace.nf,workspace.nframes,workspace.line_count,workspace.inner_count,o.data());}
template<class T>void stft_device(const DeviceArray<T>&x,int frame,int hop,DeviceArray<ComplexFloat>&o){if(frame<2||hop<1||hop>frame)throw std::invalid_argument("stft shape");StftDeviceParams params;params.window=WindowParams("hann");params.nperseg=frame;params.noverlap=frame-hop;params.nfft=frame;params.detrend="";params.return_onesided=false;params.boundary="";params.padded=false;params.axis=-1;StftDeviceWorkspace workspace((int)x.size(),params,false);stft_device(x,o,workspace,params);}
template<class T>void istft_device(const DeviceArray<TypedComplex<T>>&z,DeviceArray<float>&o,IstftDeviceWorkspace&workspace){static_assert(detail::is_simple_signal_input_v<T>);if(z.size()!=size_t(workspace.bins*workspace.frames)||o.size()!=size_t(workspace.final_output_length)||!workspace.fft)throw std::invalid_argument("istft shape/workspace");cuda_utils::launch_1d_kernel(spectral_detail::istft_build_spectrum_kernel<TypedComplex<T>>,workspace.fft_scratch_size(),z.data(),workspace.bins,workspace.frames,workspace.bins,workspace.frames,workspace.nfft,workspace.input_onesided,false,workspace.spectrum.data());workspace.fft->ifft_batch_device(workspace.spectrum,workspace.ifft_output,workspace.frames);cuda_utils::launch_1d_kernel(spectral_detail::istft_finalize_output_kernel<float,float,float>,o.size(),workspace.ifft_output.data(),workspace.window.data(),workspace.nperseg,workspace.nperseg-workspace.hop,workspace.nfft,workspace.frames,workspace.input_onesided,workspace.boundary,workspace.window_sum,workspace.output_length,o.data());}
template<class T>void istft_device(const DeviceArray<T>&z,DeviceArray<float>&o,IstftDeviceWorkspace&workspace){static_assert(detail::is_simple_signal_input_v<T>);if(z.size()!=size_t(workspace.bins*workspace.frames)||o.size()!=size_t(workspace.final_output_length)||!workspace.fft)throw std::invalid_argument("istft shape/workspace");cuda_utils::launch_1d_kernel(spectral_detail::istft_build_spectrum_kernel<T>,workspace.fft_scratch_size(),z.data(),workspace.bins,workspace.frames,workspace.bins,workspace.frames,workspace.nfft,workspace.input_onesided,false,workspace.spectrum.data());workspace.fft->ifft_batch_device(workspace.spectrum,workspace.ifft_output,workspace.frames);cuda_utils::launch_1d_kernel(spectral_detail::istft_finalize_output_kernel<float,float,float>,o.size(),workspace.ifft_output.data(),workspace.window.data(),workspace.nperseg,workspace.nperseg-workspace.hop,workspace.nfft,workspace.frames,workspace.input_onesided,workspace.boundary,workspace.window_sum,workspace.output_length,o.data());}
template<class T>void istft_device(const DeviceArray<TypedComplex<T>>&z,int frames,int bins,int hop,DeviceArray<float>&o){IstftDeviceWorkspace workspace(frames,bins,hop);istft_device(z,o,workspace);}
template <class T>
void spectrogram_device(
    const DeviceArray<T>& x, DeviceArray<float>& output,
    SpectrogramDeviceWorkspace& workspace,
    const SpectrogramDeviceParams& params)
{
    static_assert(detail::is_simple_signal_input_v<T>);
    if (params.mode == "complex"
        || workspace.stft.input_size != static_cast<int>(x.size())
        || output.size() != workspace.output_size()) {
        throw std::invalid_argument("spectrogram shape/workspace");
    }
    if (x.empty()) {
        return;
    }
    const int step = workspace.stft.nperseg - workspace.stft.noverlap;
    cuda_utils::launch_1d_kernel(
        spectral_detail::stft_prepare_real_kernel<T, float, float>,
        workspace.stft.fft_scratch_size(), x.data(),
        workspace.stft.line_bases.data(), workspace.stft.axis_length,
        workspace.stft.axis_stride, workspace.stft.line_count,
        workspace.stft.window.data(), workspace.stft.nperseg,
        workspace.stft.nfft, step, workspace.stft.nframes, 0, 0,
        workspace.stft.detrend == "constant",
        static_cast<float>(workspace.stft.scale),
        workspace.stft.fft_input_data());
    const int batches = workspace.stft.line_count * workspace.stft.nframes;
    workspace.stft.fft->fft_batch_device(
        workspace.stft.fft_input_data(), workspace.stft.fft_output_data(),
        batches, workspace.stft.fft_scratch_size());
    const int mode = params.mode == "psd" ? 0
        : (params.mode == "magnitude" ? 1 : 2);
    cuda_utils::launch_1d_kernel(
        spectral_detail::spectrogram_real_post_kernel<float, float>,
        workspace.output_size(), workspace.stft.fft_output_data(),
        workspace.stft.nfft, workspace.stft.nf, workspace.stft.nframes,
        workspace.stft.line_count, workspace.stft.inner_count,
        static_cast<float>(workspace.post_scale),
        workspace.stft.return_onesided, mode, output.data());
    if (params.mode == "phase") {
        cuda_utils::launch_1d_kernel(
            spectral_detail::spectrogram_phase_unwrap_kernel,
            static_cast<std::size_t>(workspace.stft.line_count)
                * workspace.stft.nframes,
            output.data(), workspace.stft.nf, workspace.stft.nframes,
            workspace.stft.line_count, workspace.stft.inner_count);
    }
}
template<class T>void spectrogram_device(const DeviceArray<T>&x,int frame,int hop,DeviceArray<float>&o){if(frame<2||hop<1||hop>frame)throw std::invalid_argument("spectrogram shape");SpectrogramDeviceParams params;params.window=WindowParams("hann");params.nperseg=frame;params.noverlap=frame-hop;params.nfft=frame;params.detrend="";params.return_onesided=false;params.scaling="spectrum";params.mode="psd";SpectrogramDeviceWorkspace workspace((int)x.size(),params);spectrogram_device(x,o,workspace,params);}
namespace detail {
template<class T>void lombscargle_fp32_compute_device(const DeviceArray<T>&t,const DeviceArray<T>&y,const DeviceArray<T>&f,DeviceArray<float>&o,bool precenter,bool normalize){static_assert(is_simple_signal_input_v<T>);if(t.empty()||t.size()!=y.size()||f.size()!=o.size())throw std::invalid_argument("lombscargle_device: time and signal must be nonempty with equal lengths and output size must match frequencies");if(f.empty())return;spectral_detail::lombscargle_kernel<T,T,float,float><<<static_cast<unsigned int>(f.size()),1>>>(t.data(),y.data(),(int)t.size(),f.data(),(int)f.size(),precenter,normalize,o.data());CUDA_CHECK(cudaGetLastError());}
template<class T>void vectorstrength_fp32_compute_device(const DeviceArray<T>&e,const DeviceArray<T>&p,DeviceArray<float>&s,DeviceArray<float>&phase){static_assert(is_simple_signal_input_v<T>);if(p.size()!=s.size()||p.size()!=phase.size())throw std::invalid_argument("vectorstrength_device: strength and phase output sizes must match period count");if(p.empty())return;constexpr int threads=256;spectral_detail::vectorstrength_multi_kernel<T,T,float,float><<<static_cast<unsigned int>(p.size()),threads>>>(e.data(),(int)e.size(),p.data(),(int)p.size(),s.data(),phase.data());CUDA_CHECK(cudaGetLastError());}
}

#define INST(T) template void csd_device(const DeviceArray<T>&,const DeviceArray<T>&,DeviceArray<ComplexFloat>&);template void csd_device(const DeviceArray<T>&,const DeviceArray<T>&,DeviceArray<ComplexFloat>&,CsdDeviceWorkspace&,const CsdDeviceParams&);template void stft_device(const DeviceArray<T>&,int,int,DeviceArray<ComplexFloat>&);template void stft_device(const DeviceArray<T>&,DeviceArray<ComplexFloat>&,StftDeviceWorkspace&,const StftDeviceParams&);template void istft_device(const DeviceArray<TypedComplex<T>>&,int,int,int,DeviceArray<float>&);template void istft_device(const DeviceArray<TypedComplex<T>>&,DeviceArray<float>&,IstftDeviceWorkspace&);template void istft_device(const DeviceArray<T>&,DeviceArray<float>&,IstftDeviceWorkspace&);template void spectrogram_device(const DeviceArray<T>&,int,int,DeviceArray<float>&);template void spectrogram_device(const DeviceArray<T>&,DeviceArray<float>&,SpectrogramDeviceWorkspace&,const SpectrogramDeviceParams&);template void detail::lombscargle_fp32_compute_device(const DeviceArray<T>&,const DeviceArray<T>&,const DeviceArray<T>&,DeviceArray<float>&,bool,bool);template void detail::vectorstrength_fp32_compute_device(const DeviceArray<T>&,const DeviceArray<T>&,DeviceArray<float>&,DeviceArray<float>&)
INST(float);INST(__half);INST(std::int32_t);INST(std::int16_t);INST(std::int8_t);
#undef INST
} // namespace cusignal
