/**
 * @file fft_cufft.cu
 * @brief ZQ500 dlfft 的 cuFFT-compatible 共享接口实现
 *
 * 当前正式构建不把本文件作为独立后端：fft_dlfft.cu 定义
 * FFT_INTERFACE_DLFFT_BACKEND 后文本包含本文件，使用 ZQ500 SDK 的
 * <cufftXt.h>，并链接 dlfft/curt。文件中的 cufft* 名称来自 dlfft 提供的
 * cuFFT-compatible API，不表示当前程序链接 NVIDIA cuFFT。
 * DLFFT 后端不注入项目替换层：Plan、Exec、Destroy 和版本查询均直接调用供应商
 * /zq500/sdk/lib/libdlfft.so。需要无已观测 CPU/GPU 持续增长的项目后端时选择
 * USE_THRUST=ON。
 *
 * 未定义 FFT_INTERFACE_DLFFT_BACKEND 时仍保留 <cufft.h> 源码兼容分支，
 * 但当前仓库没有正式、经过验证的 NVIDIA CMake 目标，不将它声明为当前支持后端。
 * test_all 中名为 cufft-compatible 的目标也仍链接 ZQ500 dlfft，只验证这份共享
 * 封装在平台兼容 API 下可用。
 *
 * 归一化约定（与 fft_thrust.cu 一致）：
 *   - 正向变换（FFT_FORWARD）：不归一化
 *   - 逆向变换（FFT_INVERSE）：自动 1/N 缩放
 *     （与 NumPy numpy.fft.ifft 行为一致）
 *
 * @note 当前项目事实来源以 ZQ500 dlcc、curt、dlfft 的远程构建和运行结果为准。
 *
 * @author cusignal_cpp
 * @date 2024-2026
 */

#include <cusignal/backends/fft/fft_interface.h>
#include <cuda_runtime.h>
#if defined(FFT_INTERFACE_DLFFT_BACKEND)
#include <cufftXt.h>
#else
#include <cufft.h>
#endif
#include <iostream>
#include <cstring>

namespace {

__global__ void normalize_complex_kernel(cufftComplex* data, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

__global__ void normalize_real_kernel(float* data, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

}

struct FFTPlan {
    cufftHandle handle{};
    FFTType type;
    int nx;
    int ny;
    int batch;
    FFTDirection direction;
};

extern "C" {

FFTResult fft_plan_create_1d(
    FFTPlanHandle* plan,
    int nx,
    int batch,
    FFTType type,
    FFTDirection direction
) {
    if (!plan || nx <= 0 || batch <= 0 ||
        (direction != FFT_FORWARD && direction != FFT_INVERSE)) {
        return FFT_INVALID_VALUE;
    }

    FFTPlan* p = new FFTPlan();
    p->type = type;
    p->nx = nx;
    p->ny = 0;
    p->batch = batch;
    p->direction = direction;

    cufftType cufft_type;
    switch (type) {
        case FFT_TYPE_C2C:
            cufft_type = CUFFT_C2C;
            break;
        case FFT_TYPE_R2C:
            cufft_type = CUFFT_R2C;
            break;
        case FFT_TYPE_C2R:
            cufft_type = CUFFT_C2R;
            break;
        default:
            delete p;
            return FFT_INVALID_VALUE;
    }

    cufftResult result = cufftPlan1d(&p->handle, nx, cufft_type, batch);
    if (result != CUFFT_SUCCESS) {
        delete p;
        return FFT_ALLOC_FAILED;
    }

    *plan = reinterpret_cast<FFTPlanHandle>(p);
    return FFT_SUCCESS;
}

FFTResult fft_plan_create_2d(
    FFTPlanHandle* plan,
    int nx,
    int ny,
    FFTType type,
    FFTDirection direction
) {
    if (!plan || nx <= 0 || ny <= 0 ||
        (direction != FFT_FORWARD && direction != FFT_INVERSE)) {
        return FFT_INVALID_VALUE;
    }

    FFTPlan* p = new FFTPlan();
    p->type = type;
    p->nx = nx;
    p->ny = ny;
    p->batch = 1;
    p->direction = direction;

    cufftType cufft_type;
    switch (type) {
        case FFT_TYPE_C2C:
            cufft_type = CUFFT_C2C;
            break;
        case FFT_TYPE_R2C:
            cufft_type = CUFFT_R2C;
            break;
        case FFT_TYPE_C2R:
            cufft_type = CUFFT_C2R;
            break;
        default:
            delete p;
            return FFT_INVALID_VALUE;
    }

    cufftResult result = cufftPlan2d(&p->handle, ny, nx, cufft_type);
    if (result != CUFFT_SUCCESS) {
        delete p;
        return FFT_ALLOC_FAILED;
    }

    *plan = reinterpret_cast<FFTPlanHandle>(p);
    return FFT_SUCCESS;
}

FFTResult fft_execute(
    FFTPlanHandle plan,
    void* output,
    const void* input
) {
    if (!plan) return FFT_INVALID_VALUE;

    FFTPlan* p = reinterpret_cast<FFTPlan*>(plan);
    cufftResult result;
    
    int cufft_direction = (p->direction == FFT_FORWARD) ? CUFFT_FORWARD : CUFFT_INVERSE;

    switch (p->type) {
        case FFT_TYPE_C2C: {
            cufftComplex* in = const_cast<cufftComplex*>(
                reinterpret_cast<const cufftComplex*>(input)
            );
            result = cufftExecC2C(
                p->handle,
                in,
                reinterpret_cast<cufftComplex*>(output),
                cufft_direction
            );
            
            if (result == CUFFT_SUCCESS && p->direction == FFT_INVERSE) {
                int fft_size = (p->ny > 0) ? (p->nx * p->ny) : p->nx;
                int total_size = fft_size * p->batch;
                float scale = 1.0f / fft_size;
                int threads = 256;
                int blocks = (total_size + threads - 1) / threads;
                normalize_complex_kernel<<<blocks, threads>>>(
                    reinterpret_cast<cufftComplex*>(output),
                    total_size,
                    scale
                );
            }
            break;
        }
        case FFT_TYPE_R2C: {
            cufftReal* in = const_cast<cufftReal*>(
                reinterpret_cast<const cufftReal*>(input)
            );
            result = cufftExecR2C(
                p->handle,
                in,
                reinterpret_cast<cufftComplex*>(output)
            );
            break;
        }
        case FFT_TYPE_C2R: {
            cufftComplex* in = const_cast<cufftComplex*>(
                reinterpret_cast<const cufftComplex*>(input)
            );
            result = cufftExecC2R(
                p->handle,
                in,
                reinterpret_cast<cufftReal*>(output)
            );
            
            if (result == CUFFT_SUCCESS) {
                int total_size = p->nx * p->batch;
                float scale = 1.0f / p->nx;
                int threads = 256;
                int blocks = (total_size + threads - 1) / threads;
                normalize_real_kernel<<<blocks, threads>>>(
                    reinterpret_cast<float*>(output),
                    total_size,
                    scale
                );
            }
            break;
        }
        default:
            return FFT_INVALID_VALUE;
    }

    if (result != CUFFT_SUCCESS) {
        return FFT_EXEC_FAILED;
    }

    return FFT_SUCCESS;
}

FFTResult fft_execute_inplace(
    FFTPlanHandle plan,
    void* data
) {
    if (!plan) return FFT_INVALID_VALUE;

    FFTPlan* p = reinterpret_cast<FFTPlan*>(plan);
    
    int cufft_direction = (p->direction == FFT_FORWARD) ? CUFFT_FORWARD : CUFFT_INVERSE;

    if (p->type == FFT_TYPE_C2C) {
        cufftResult result = cufftExecC2C(
            p->handle,
            reinterpret_cast<cufftComplex*>(data),
            reinterpret_cast<cufftComplex*>(data),
            cufft_direction
        );
        if (result != CUFFT_SUCCESS) {
            return FFT_EXEC_FAILED;
        }
        
        if (p->direction == FFT_INVERSE) {
            const int fft_size = p->ny > 0 ? p->nx * p->ny : p->nx;
            const int total_size = fft_size * p->batch;
            const float scale = 1.0f / fft_size;
            int threads = 256;
            int blocks = (total_size + threads - 1) / threads;
            normalize_complex_kernel<<<blocks, threads>>>(
                reinterpret_cast<cufftComplex*>(data),
                total_size,
                scale
            );
        }
    }

    return FFT_SUCCESS;
}

FFTResult fft_plan_destroy(FFTPlanHandle plan) {
    if (!plan) return FFT_INVALID_VALUE;

    FFTPlan* p = reinterpret_cast<FFTPlan*>(plan);
    cufftResult result = cufftDestroy(p->handle);
    delete p;

    if (result != CUFFT_SUCCESS) {
        return FFT_INTERNAL_ERROR;
    }

    return FFT_SUCCESS;
}

FFTResult fft_shift(
    void* output,
    const void* input,
    int nx,
    int batch
) {
    int half = nx / 2;
    size_t elem_size = sizeof(cufftComplex);

    cudaError_t cuda_err = cudaMemcpy(
        output,
        reinterpret_cast<const char*>(input) + half * elem_size,
        (nx - half) * elem_size * batch,
        cudaMemcpyDeviceToDevice
    );
    if (cuda_err != cudaSuccess) return FFT_INTERNAL_ERROR;

    cuda_err = cudaMemcpy(
        reinterpret_cast<char*>(output) + (nx - half) * elem_size,
        input,
        half * elem_size * batch,
        cudaMemcpyDeviceToDevice
    );
    if (cuda_err != cudaSuccess) return FFT_INTERNAL_ERROR;

    return FFT_SUCCESS;
}

FFTResult fft_inverse_shift(
    void* output,
    const void* input,
    int nx,
    int batch
) {
    int half = nx / 2;
    size_t elem_size = sizeof(cufftComplex);

    cudaError_t cuda_err = cudaMemcpy(
        output,
        reinterpret_cast<const char*>(input) + half * elem_size,
        (nx - half) * elem_size * batch,
        cudaMemcpyDeviceToDevice
    );
    if (cuda_err != cudaSuccess) return FFT_INTERNAL_ERROR;

    cuda_err = cudaMemcpy(
        reinterpret_cast<char*>(output) + (nx - half) * elem_size,
        input,
        half * elem_size * batch,
        cudaMemcpyDeviceToDevice
    );
    if (cuda_err != cudaSuccess) return FFT_INTERNAL_ERROR;

    return FFT_SUCCESS;
}

const char* fft_error_string(FFTResult result) {
    switch (result) {
        case FFT_SUCCESS: return "Success";
        case FFT_INVALID_VALUE: return "Invalid value";
        case FFT_ALLOC_FAILED: return "Allocation failed";
        case FFT_EXEC_FAILED: return "Execution failed";
        case FFT_INTERNAL_ERROR: return "Internal error";
        default: return "Unknown error";
    }
}

int fft_version() {
#if defined(FFT_INTERFACE_DLFFT_BACKEND)
    int dlfft_version = 0;
    return cufftGetVersion(&dlfft_version) == CUFFT_SUCCESS ? 3 : -1;
#else
    return 2;
#endif
}

FFTResult fft_freq(float* freqs, int n, float d) {
    if (!freqs || n <= 0) return FFT_INVALID_VALUE;

    float inv_n_d = 1.0f / (n * d);
    int num_pos = (n + 1) / 2;

    for (int i = 0; i < num_pos; i++) {
        freqs[i] = i * inv_n_d;
    }

    for (int i = 0; i < n - num_pos; i++) {
        freqs[num_pos + i] = -(n - num_pos - i) * inv_n_d;
    }

    return FFT_SUCCESS;
}

FFTResult fft_rfftfreq(float* freqs, int n, float d) {
    if (!freqs || n <= 0) return FFT_INVALID_VALUE;

    int len = n / 2 + 1;
    float inv_n_d = 1.0f / (n * d);

    for (int i = 0; i < len; i++) {
        freqs[i] = i * inv_n_d;
    }

    return FFT_SUCCESS;
}

FFTResult fft_shift_2d(
    void* output,
    const void* input,
    int nx,
    int ny
) {
    if (!output || !input) return FFT_INVALID_VALUE;
    if (nx <= 0 || ny <= 0) return FFT_INVALID_VALUE;

    int shift_x = nx / 2;
    int shift_y = ny / 2;
    size_t elem_size = sizeof(cufftComplex);

    cufftComplex* out = reinterpret_cast<cufftComplex*>(output);
    const cufftComplex* in = reinterpret_cast<const cufftComplex*>(input);

    for (int y = 0; y < ny; y++) {
        int new_y = (y + shift_y) % ny;
        cudaError_t err = cudaMemcpy(
            out + y * nx,
            in + new_y * nx + shift_x,
            (nx - shift_x) * elem_size,
            cudaMemcpyDeviceToDevice
        );
        if (err != cudaSuccess) return FFT_INTERNAL_ERROR;

        err = cudaMemcpy(
            out + y * nx + (nx - shift_x),
            in + new_y * nx,
            shift_x * elem_size,
            cudaMemcpyDeviceToDevice
        );
        if (err != cudaSuccess) return FFT_INTERNAL_ERROR;
    }

    return FFT_SUCCESS;
}

FFTResult fft_inverse_shift_2d(
    void* output,
    const void* input,
    int nx,
    int ny
) {
    if (!output || !input) return FFT_INVALID_VALUE;
    if (nx <= 0 || ny <= 0) return FFT_INVALID_VALUE;

    int shift_x = (nx + 1) / 2;
    int shift_y = (ny + 1) / 2;
    size_t elem_size = sizeof(cufftComplex);

    cufftComplex* out = reinterpret_cast<cufftComplex*>(output);
    const cufftComplex* in = reinterpret_cast<const cufftComplex*>(input);

    for (int y = 0; y < ny; y++) {
        int new_y = (y + shift_y) % ny;
        cudaError_t err = cudaMemcpy(
            out + y * nx,
            in + new_y * nx + shift_x,
            (nx - shift_x) * elem_size,
            cudaMemcpyDeviceToDevice
        );
        if (err != cudaSuccess) return FFT_INTERNAL_ERROR;

        err = cudaMemcpy(
            out + y * nx + (nx - shift_x),
            in + new_y * nx,
            shift_x * elem_size,
            cudaMemcpyDeviceToDevice
        );
        if (err != cudaSuccess) return FFT_INTERNAL_ERROR;
    }

    return FFT_SUCCESS;
}

} // extern "C"
