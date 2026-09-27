#pragma once

#include <cusignal/backends/fft/fft_interface.h>

#include <vector>

namespace cusignal {

class FFTInterface_cpu {
public:
    explicit FFTInterface_cpu(int n);
    ~FFTInterface_cpu() = default;

    // 归一化约定与 FFTInterface 保持一致：
    // - fft / fft_inplace: forward，不归一化
    // - ifft / ifft_inplace: inverse，自动除以 N
    std::vector<complexd> fft(const std::vector<complexd>& input);
    std::vector<complexd> ifft(const std::vector<complexd>& input);
    void fft_inplace(std::vector<complexd>& input_output);
    void ifft_inplace(std::vector<complexd>& input_output);
    std::vector<std::vector<complexd>> fft_batch(const std::vector<std::vector<complexd>>& input);
    std::vector<std::vector<complexd>> ifft_batch(const std::vector<std::vector<complexd>>& input);
    std::vector<ComplexFloat> ifft_batch_float(const std::vector<ComplexFloat>& input, int batch);
    std::vector<complexd> rfft(const std::vector<complexd>& input);
    std::vector<complexd> irfft(const std::vector<complexd>& input, int output_len = 0);

private:
    int n_;
};

class FFTInterface2D_cpu {
public:
    FFTInterface2D_cpu(int nx, int ny);
    ~FFTInterface2D_cpu() = default;

    std::vector<std::vector<complexd>> fft2d(const std::vector<std::vector<complexd>>& input);
    std::vector<std::vector<complexd>> ifft2d(const std::vector<std::vector<complexd>>& input);
    void fft2d_inplace(std::vector<std::vector<complexd>>& input_output);
    void ifft2d_inplace(std::vector<std::vector<complexd>>& input_output);

private:
    int nx_;
    int ny_;
};

} // namespace cusignal
