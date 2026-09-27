#include <cusignal/backends/fft/fft_interface_cpu.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cusignal {
namespace {

constexpr double PI = 3.141592653589793238462643383279502884;

bool is_power_of_two(std::size_t n)
{
    return n > 0 && (n & (n - 1)) == 0;
}

std::size_t next_power_of_two(std::size_t n)
{
    std::size_t p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

std::vector<complexd> resize_and_copy(const std::vector<complexd>& input, int n)
{
    std::vector<complexd> out(static_cast<std::size_t>(n), complexd(0.0, 0.0));
    const std::size_t count = std::min(out.size(), input.size());
    std::copy(input.begin(), input.begin() + static_cast<std::ptrdiff_t>(count), out.begin());
    return out;
}

void fft_radix2_inplace(std::vector<complexd>& a, bool inverse)
{
    const std::size_t n = a.size();
    std::size_t j = 0;
    for (std::size_t i = 1; i < n; ++i) {
        std::size_t bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }

    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double angle = 2.0 * PI / static_cast<double>(len) * (inverse ? 1.0 : -1.0);
        const complexd wlen(std::cos(angle), std::sin(angle));
        for (std::size_t i = 0; i < n; i += len) {
            complexd w(1.0, 0.0);
            const std::size_t half = len >> 1;
            for (std::size_t j2 = 0; j2 < half; ++j2) {
                const complexd u = a[i + j2];
                const complexd v = a[i + j2 + half] * w;
                a[i + j2] = u + v;
                a[i + j2 + half] = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse && n > 0) {
        const double scale = 1.0 / static_cast<double>(n);
        for (auto& v : a) {
            v *= scale;
        }
    }
}

std::vector<complexd> fft_bluestein(const std::vector<complexd>& input, bool inverse)
{
    const std::size_t n = input.size();
    if (n == 0) {
        return {};
    }
    if (n == 1) {
        return input;
    }

    const int sign = inverse ? 1 : -1;
    const std::size_t m = next_power_of_two(2 * n - 1);

    std::vector<complexd> a(m, complexd(0.0, 0.0));
    std::vector<complexd> b(m, complexd(0.0, 0.0));

    for (std::size_t k = 0; k < n; ++k) {
        const double angle = static_cast<double>(sign) * PI * static_cast<double>(k) * static_cast<double>(k) / static_cast<double>(n);
        const complexd chirp(std::cos(angle), std::sin(angle));
        a[k] = input[k] * chirp;
    }

    for (std::size_t k = 0; k < n; ++k) {
        const double angle = static_cast<double>(-sign) * PI * static_cast<double>(k) * static_cast<double>(k) / static_cast<double>(n);
        const complexd chirp(std::cos(angle), std::sin(angle));
        b[k] = chirp;
        if (k > 0) {
            b[m - k] = chirp;
        }
    }

    fft_radix2_inplace(a, false);
    fft_radix2_inplace(b, false);
    for (std::size_t i = 0; i < m; ++i) {
        a[i] *= b[i];
    }
    fft_radix2_inplace(a, true);

    std::vector<complexd> out(n, complexd(0.0, 0.0));
    for (std::size_t k = 0; k < n; ++k) {
        const double angle = static_cast<double>(sign) * PI * static_cast<double>(k) * static_cast<double>(k) / static_cast<double>(n);
        const complexd chirp(std::cos(angle), std::sin(angle));
        out[k] = a[k] * chirp;
        if (inverse) {
            out[k] /= static_cast<double>(n);
        }
    }
    return out;
}

std::vector<complexd> fft_1d(std::vector<complexd> data, bool inverse)
{
    if (data.empty()) {
        return data;
    }
    if (is_power_of_two(data.size())) {
        fft_radix2_inplace(data, inverse);
        return data;
    }
    return fft_bluestein(data, inverse);
}

} // namespace

FFTInterface_cpu::FFTInterface_cpu(int n) : n_(n)
{
    if (n_ <= 0) {
        throw std::invalid_argument("FFTInterface_cpu: n must be positive");
    }
}

std::vector<complexd> FFTInterface_cpu::fft(const std::vector<complexd>& input)
{
    return fft_1d(resize_and_copy(input, n_), false);
}

std::vector<complexd> FFTInterface_cpu::ifft(const std::vector<complexd>& input)
{
    return fft_1d(resize_and_copy(input, n_), true);
}

void FFTInterface_cpu::fft_inplace(std::vector<complexd>& input_output)
{
    input_output = fft(input_output);
}

void FFTInterface_cpu::ifft_inplace(std::vector<complexd>& input_output)
{
    input_output = ifft(input_output);
}

std::vector<std::vector<complexd>> FFTInterface_cpu::fft_batch(const std::vector<std::vector<complexd>>& input)
{
    std::vector<std::vector<complexd>> out;
    out.reserve(input.size());
    for (const auto& row : input) {
        out.push_back(fft(row));
    }
    return out;
}

std::vector<std::vector<complexd>> FFTInterface_cpu::ifft_batch(const std::vector<std::vector<complexd>>& input)
{
    std::vector<std::vector<complexd>> out;
    out.reserve(input.size());
    for (const auto& row : input) {
        out.push_back(ifft(row));
    }
    return out;
}

std::vector<ComplexFloat> FFTInterface_cpu::ifft_batch_float(const std::vector<ComplexFloat>& input, int batch)
{
    if (batch <= 0) {
        return {};
    }

    std::vector<ComplexFloat> out(static_cast<std::size_t>(batch) * static_cast<std::size_t>(n_));
    for (int b = 0; b < batch; ++b) {
        std::vector<complexd> row(static_cast<std::size_t>(n_), complexd(0.0, 0.0));
        for (int i = 0; i < n_; ++i) {
            const std::size_t idx = static_cast<std::size_t>(b) * static_cast<std::size_t>(n_) + static_cast<std::size_t>(i);
            if (idx < input.size()) {
                row[static_cast<std::size_t>(i)] = complexd(input[idx].re, input[idx].im);
            }
        }
        const auto transformed = ifft(row);
        for (int i = 0; i < n_; ++i) {
            const std::size_t idx = static_cast<std::size_t>(b) * static_cast<std::size_t>(n_) + static_cast<std::size_t>(i);
            out[idx].re = static_cast<float>(transformed[static_cast<std::size_t>(i)].real());
            out[idx].im = static_cast<float>(transformed[static_cast<std::size_t>(i)].imag());
        }
    }
    return out;
}

std::vector<complexd> FFTInterface_cpu::rfft(const std::vector<complexd>& input)
{
    auto full = fft(input);
    const int output_len = static_cast<int>(full.size()) / 2 + 1;
    return std::vector<complexd>(full.begin(), full.begin() + output_len);
}

std::vector<complexd> FFTInterface_cpu::irfft(const std::vector<complexd>& input, int output_len)
{
    if (output_len <= 0) {
        output_len = 2 * (static_cast<int>(input.size()) - 1);
    }
    std::vector<complexd> full(static_cast<std::size_t>(output_len), complexd(0.0, 0.0));
    const int half = static_cast<int>(input.size());
    for (int i = 0; i < half && i < output_len; ++i) {
        full[static_cast<std::size_t>(i)] = input[static_cast<std::size_t>(i)];
    }
    for (int i = 1; i < half - 1 && i < output_len; ++i) {
        const int mirrored = output_len - i;
        if (mirrored >= half && mirrored < output_len) {
            full[static_cast<std::size_t>(mirrored)] = std::conj(input[static_cast<std::size_t>(i)]);
        }
    }
    auto time = FFTInterface_cpu(output_len).ifft(full);
    for (auto& v : time) {
        v = complexd(v.real(), 0.0);
    }
    return time;
}

FFTInterface2D_cpu::FFTInterface2D_cpu(int nx, int ny) : nx_(nx), ny_(ny)
{
    if (nx_ <= 0 || ny_ <= 0) {
        throw std::invalid_argument("FFTInterface2D_cpu: nx and ny must be positive");
    }
}

std::vector<std::vector<complexd>> FFTInterface2D_cpu::fft2d(const std::vector<std::vector<complexd>>& input)
{
    std::vector<std::vector<complexd>> temp(static_cast<std::size_t>(nx_), std::vector<complexd>(static_cast<std::size_t>(ny_), complexd(0.0, 0.0)));
    FFTInterface_cpu row_fft(ny_);
    FFTInterface_cpu col_fft(nx_);

    for (int r = 0; r < nx_; ++r) {
        const std::vector<complexd> row = (r < static_cast<int>(input.size())) ? input[static_cast<std::size_t>(r)] : std::vector<complexd>{};
        temp[static_cast<std::size_t>(r)] = row_fft.fft(row);
    }

    std::vector<std::vector<complexd>> out(static_cast<std::size_t>(nx_), std::vector<complexd>(static_cast<std::size_t>(ny_)));
    for (int c = 0; c < ny_; ++c) {
        std::vector<complexd> column(static_cast<std::size_t>(nx_));
        for (int r = 0; r < nx_; ++r) {
            column[static_cast<std::size_t>(r)] = temp[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
        }
        const auto transformed = col_fft.fft(column);
        for (int r = 0; r < nx_; ++r) {
            out[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = transformed[static_cast<std::size_t>(r)];
        }
    }
    return out;
}

std::vector<std::vector<complexd>> FFTInterface2D_cpu::ifft2d(const std::vector<std::vector<complexd>>& input)
{
    std::vector<std::vector<complexd>> temp(static_cast<std::size_t>(nx_), std::vector<complexd>(static_cast<std::size_t>(ny_), complexd(0.0, 0.0)));
    FFTInterface_cpu row_ifft(ny_);
    FFTInterface_cpu col_ifft(nx_);

    for (int r = 0; r < nx_; ++r) {
        const std::vector<complexd> row = (r < static_cast<int>(input.size())) ? input[static_cast<std::size_t>(r)] : std::vector<complexd>{};
        temp[static_cast<std::size_t>(r)] = row_ifft.ifft(row);
    }

    std::vector<std::vector<complexd>> out(static_cast<std::size_t>(nx_), std::vector<complexd>(static_cast<std::size_t>(ny_)));
    for (int c = 0; c < ny_; ++c) {
        std::vector<complexd> column(static_cast<std::size_t>(nx_));
        for (int r = 0; r < nx_; ++r) {
            column[static_cast<std::size_t>(r)] = temp[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
        }
        const auto transformed = col_ifft.ifft(column);
        for (int r = 0; r < nx_; ++r) {
            out[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = transformed[static_cast<std::size_t>(r)];
        }
    }
    return out;
}

void FFTInterface2D_cpu::fft2d_inplace(std::vector<std::vector<complexd>>& input_output)
{
    input_output = fft2d(input_output);
}

void FFTInterface2D_cpu::ifft2d_inplace(std::vector<std::vector<complexd>>& input_output)
{
    input_output = ifft2d(input_output);
}

} // namespace cusignal
