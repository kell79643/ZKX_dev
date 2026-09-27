/**
 * @file signal_processing.h
 * @brief cusignal_cpp 公共总头文件
 *
 * 用户可通过本头文件访问项目当前公开 API：
 * - FP32、FP16、INT32、INT16、INT8 的 `*_cpu` CPU reference
 * - FP32、FP16、INT32、INT16、INT8 的 `*_device` 正式 GPU interface
 * - `FFTInterface` GPU FFT 抽象层
 *
 * 说明：`FFTInterface_cpu` 主要用于 FFT-based CPU 路径与测试比较，
 * 若需要直接使用请包含 `fft_interface/fft_interface_cpu.h`。
 */

#pragma once

#include <cusignal/backends/fft/fft_interface.h>

#include <cusignal/operators/bsplines/bsplines_typed.h>
#include <cusignal/operators/convolution/convolution_typed.h>
#include <cusignal/operators/demod/demod_typed.h>
#include <cusignal/operators/estimation/estimation_typed.h>
#include <cusignal/operators/filter_design/filter_design_typed.h>
#include <cusignal/operators/filtering/filtering_typed.h>
#include <cusignal/operators/peak_finding/peak_finding_typed.h>
#include <cusignal/operators/radartools/radartools_typed.h>
#include <cusignal/operators/spectral_analysis/spectral_analysis_typed.h>
#include <cusignal/operators/waveforms/waveforms_typed.h>
#include <cusignal/operators/wavelets/wavelets_typed.h>
#include <cusignal/operators/windows/windows_typed.h>
