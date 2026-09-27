/**
 * @file fft_dlfft.cu
 * @brief ZQ500 dlfft 后端入口
 *
 * ZQ500 的 dlfft 提供 cuFFT 兼容 API。这里复用 fft_cufft.cu 中的
 * FFTInterface C API 实现，但通过独立源文件和 CMake 目标明确选择
 * 中科芯 dlfft 链接路径，避免上层业务代码直接包含或调用平台库。
 */

#define FFT_INTERFACE_DLFFT_BACKEND 1
#include "fft_cufft.cu"
