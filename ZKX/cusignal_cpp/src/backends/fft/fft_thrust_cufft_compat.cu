/**
 * @file fft_thrust_cufft_compat.cu
 * @brief libzkx_fft_thrust.so 对当前项目所用 ZQ500 DLFFT/cuFFT API 的兼容导出
 *
 * 本文件只增加官方 API 兼容入口，不改变项目既有 fft_* API。兼容入口复用
 * fft_thrust.cu 的计划与执行实现，并补偿项目逆变换自动归一化，从而保持
 * cufftExecC2C/cufftExecC2R 的官方未归一化逆变换语义。ZQ500 libdlfft 的
 * 1D R2C/C2R batch 契约使用每批 N 个复数的完整共轭频谱，本层只在 cufft*
 * 边界转换该布局，项目既有 fft_* 紧凑半谱契约保持不变。
 */

#include <cusignal/backends/fft/fft_interface.h>

#include <cufftXt.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace {

constexpr int kFftThrustCufftCompatibilityVersion = 10000;

struct CufftCompatibilityPlan {
    FFTPlanHandle forward = nullptr;
    FFTPlanHandle inverse = nullptr;
    cufftType type = CUFFT_C2C;
    int nx = 0;
    int ny = 0;
    int batch = 1;
    cufftComplex* compact_real_spectrum = nullptr;
};

std::mutex g_plan_mutex;
std::unordered_map<cufftHandle, std::shared_ptr<CufftCompatibilityPlan>> g_plans;
std::atomic<int> g_next_handle{1};

__global__ void cufft_compat_scale_complex(cufftComplex* values, int count, float scale)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        values[index].x *= scale;
        values[index].y *= scale;
    }
}

__global__ void cufft_compat_scale_real(cufftReal* values, int count, float scale)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) values[index] *= scale;
}

__global__ void cufft_compat_expand_real_spectrum(
    cufftComplex* full, const cufftComplex* compact, int length, int batch)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = length * batch;
    if (index >= total) return;
    const int batch_index = index / length;
    const int frequency = index - batch_index * length;
    const int compact_length = length / 2 + 1;
    const int compact_frequency = frequency <= length / 2 ? frequency : length - frequency;
    cufftComplex value = compact[batch_index * compact_length + compact_frequency];
    if (frequency > length / 2) value.y = -value.y;
    full[index] = value;
}

__global__ void cufft_compat_compact_real_spectrum(
    cufftComplex* compact, const cufftComplex* full, int length, int batch)
{
    const int compact_length = length / 2 + 1;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = compact_length * batch;
    if (index >= total) return;
    const int batch_index = index / compact_length;
    const int frequency = index - batch_index * compact_length;
    compact[index] = full[batch_index * length + frequency];
}

bool is_power_of_two(int value)
{
    return value > 0 && (value & (value - 1)) == 0;
}

cufftResult map_fft_result(FFTResult result)
{
    switch (result) {
        case FFT_SUCCESS: return CUFFT_SUCCESS;
        case FFT_INVALID_VALUE: return CUFFT_INVALID_VALUE;
        case FFT_ALLOC_FAILED: return CUFFT_ALLOC_FAILED;
        case FFT_EXEC_FAILED: return CUFFT_EXEC_FAILED;
        default: return CUFFT_INTERNAL_ERROR;
    }
}

FFTType map_fft_type(cufftType type)
{
    switch (type) {
        case CUFFT_C2C: return FFT_TYPE_C2C;
        case CUFFT_R2C: return FFT_TYPE_R2C;
        case CUFFT_C2R: return FFT_TYPE_C2R;
        default: return static_cast<FFTType>(-1);
    }
}

void destroy_plan_handles(CufftCompatibilityPlan& plan)
{
    if (plan.forward) {
        fft_plan_destroy(plan.forward);
        plan.forward = nullptr;
    }
    if (plan.inverse) {
        fft_plan_destroy(plan.inverse);
        plan.inverse = nullptr;
    }
    if (plan.compact_real_spectrum) {
        cudaFree(plan.compact_real_spectrum);
        plan.compact_real_spectrum = nullptr;
    }
}

std::shared_ptr<CufftCompatibilityPlan> find_plan(cufftHandle handle)
{
    std::lock_guard<std::mutex> lock(g_plan_mutex);
    const auto found = g_plans.find(handle);
    return found == g_plans.end() ? nullptr : found->second;
}

cufftResult publish_plan(cufftHandle* output, std::shared_ptr<CufftCompatibilityPlan> plan)
{
    if (!output || !plan) return CUFFT_INVALID_VALUE;
    int candidate = g_next_handle.fetch_add(1);
    if (candidate <= 0) return CUFFT_INTERNAL_ERROR;
    const cufftHandle handle = static_cast<cufftHandle>(candidate);
    try {
        std::lock_guard<std::mutex> lock(g_plan_mutex);
        if (!g_plans.emplace(handle, std::move(plan)).second) return CUFFT_INTERNAL_ERROR;
    } catch (...) {
        return CUFFT_ALLOC_FAILED;
    }
    *output = handle;
    return CUFFT_SUCCESS;
}

cufftResult create_1d_handles(CufftCompatibilityPlan& plan)
{
    const FFTType type = map_fft_type(plan.type);
    if (static_cast<int>(type) < 0) return CUFFT_INVALID_TYPE;
    if (type == FFT_TYPE_C2C) {
        FFTResult status = fft_plan_create_1d(
            &plan.forward, plan.nx, plan.batch, type, FFT_FORWARD);
        if (status != FFT_SUCCESS) return map_fft_result(status);
        status = fft_plan_create_1d(
            &plan.inverse, plan.nx, plan.batch, type, FFT_INVERSE);
        if (status != FFT_SUCCESS) {
            destroy_plan_handles(plan);
            return map_fft_result(status);
        }
        return CUFFT_SUCCESS;
    }
    FFTPlanHandle* selected = type == FFT_TYPE_R2C ? &plan.forward : &plan.inverse;
    const FFTDirection direction = type == FFT_TYPE_R2C ? FFT_FORWARD : FFT_INVERSE;
    const FFTResult status = fft_plan_create_1d(
        selected, plan.nx, plan.batch, type, direction);
    if (status != FFT_SUCCESS) return map_fft_result(status);
    const std::size_t compact_count =
        static_cast<std::size_t>(plan.nx / 2 + 1) * plan.batch;
    if (cudaMalloc(
            reinterpret_cast<void**>(&plan.compact_real_spectrum),
            compact_count * sizeof(cufftComplex)) != cudaSuccess) {
        destroy_plan_handles(plan);
        return CUFFT_ALLOC_FAILED;
    }
    return CUFFT_SUCCESS;
}

cufftResult create_2d_handles(CufftCompatibilityPlan& plan)
{
    if (plan.type != CUFFT_C2C) return CUFFT_INVALID_TYPE;
    FFTResult status = fft_plan_create_2d(
        &plan.forward, plan.ny, plan.nx, FFT_TYPE_C2C, FFT_FORWARD);
    if (status != FFT_SUCCESS) return map_fft_result(status);
    status = fft_plan_create_2d(
        &plan.inverse, plan.ny, plan.nx, FFT_TYPE_C2C, FFT_INVERSE);
    if (status != FFT_SUCCESS) {
        destroy_plan_handles(plan);
        return map_fft_result(status);
    }
    return CUFFT_SUCCESS;
}

}  // namespace

extern "C" {

cufftResult cufftPlan1d(cufftHandle* plan, int nx, cufftType type, int batch)
{
    if (!plan || !is_power_of_two(nx) || batch <= 0) return CUFFT_INVALID_VALUE;
    std::shared_ptr<CufftCompatibilityPlan> compatibility_plan;
    try {
        compatibility_plan = std::make_shared<CufftCompatibilityPlan>();
    } catch (...) {
        return CUFFT_ALLOC_FAILED;
    }
    compatibility_plan->type = type;
    compatibility_plan->nx = nx;
    compatibility_plan->batch = batch;
    const cufftResult status = create_1d_handles(*compatibility_plan);
    if (status != CUFFT_SUCCESS) return status;
    const cufftResult publish_status = publish_plan(plan, compatibility_plan);
    if (publish_status != CUFFT_SUCCESS) destroy_plan_handles(*compatibility_plan);
    return publish_status;
}

cufftResult cufftPlan2d(cufftHandle* plan, int nx, int ny, cufftType type)
{
    if (!plan || !is_power_of_two(nx) || !is_power_of_two(ny)) return CUFFT_INVALID_VALUE;
    std::shared_ptr<CufftCompatibilityPlan> compatibility_plan;
    try {
        compatibility_plan = std::make_shared<CufftCompatibilityPlan>();
    } catch (...) {
        return CUFFT_ALLOC_FAILED;
    }
    compatibility_plan->type = type;
    compatibility_plan->nx = nx;
    compatibility_plan->ny = ny;
    const cufftResult status = create_2d_handles(*compatibility_plan);
    if (status != CUFFT_SUCCESS) return status;
    const cufftResult publish_status = publish_plan(plan, compatibility_plan);
    if (publish_status != CUFFT_SUCCESS) destroy_plan_handles(*compatibility_plan);
    return publish_status;
}

cufftResult cufftExecC2C(
    cufftHandle handle, cufftComplex* input, cufftComplex* output, int direction)
{
    const auto plan = find_plan(handle);
    if (!plan || plan->type != CUFFT_C2C || !input || !output) return CUFFT_INVALID_VALUE;
    FFTPlanHandle selected = nullptr;
    if (direction == CUFFT_FORWARD) selected = plan->forward;
    else if (direction == CUFFT_INVERSE) selected = plan->inverse;
    else return CUFFT_INVALID_VALUE;
    const FFTResult result = fft_execute(selected, output, input);
    if (result != FFT_SUCCESS) return map_fft_result(result);
    if (direction == CUFFT_INVERSE) {
        const int transform_size = plan->ny > 0 ? plan->nx * plan->ny : plan->nx;
        const int count = transform_size * plan->batch;
        cufft_compat_scale_complex<<<(count + 255) / 256, 256>>>(
            output, count, static_cast<float>(transform_size));
    }
    return cudaGetLastError() == cudaSuccess ? CUFFT_SUCCESS : CUFFT_EXEC_FAILED;
}

cufftResult cufftExecR2C(cufftHandle handle, cufftReal* input, cufftComplex* output)
{
    const auto plan = find_plan(handle);
    if (!plan || plan->type != CUFFT_R2C || !plan->forward || !input || !output)
        return CUFFT_INVALID_VALUE;
    const FFTResult result = fft_execute(plan->forward, plan->compact_real_spectrum, input);
    if (result != FFT_SUCCESS) return map_fft_result(result);
    const int count = plan->nx * plan->batch;
    cufft_compat_expand_real_spectrum<<<(count + 255) / 256, 256>>>(
        output, plan->compact_real_spectrum, plan->nx, plan->batch);
    return cudaGetLastError() == cudaSuccess ? CUFFT_SUCCESS : CUFFT_EXEC_FAILED;
}

cufftResult cufftExecC2R(cufftHandle handle, cufftComplex* input, cufftReal* output)
{
    const auto plan = find_plan(handle);
    if (!plan || plan->type != CUFFT_C2R || !plan->inverse || !input || !output)
        return CUFFT_INVALID_VALUE;
    const int compact_count = (plan->nx / 2 + 1) * plan->batch;
    cufft_compat_compact_real_spectrum<<<(compact_count + 255) / 256, 256>>>(
        plan->compact_real_spectrum, input, plan->nx, plan->batch);
    if (cudaGetLastError() != cudaSuccess) return CUFFT_EXEC_FAILED;
    const FFTResult result = fft_execute(plan->inverse, output, plan->compact_real_spectrum);
    if (result != FFT_SUCCESS) return map_fft_result(result);
    const int count = plan->nx * plan->batch;
    cufft_compat_scale_real<<<(count + 255) / 256, 256>>>(
        output, count, static_cast<float>(plan->nx));
    return cudaGetLastError() == cudaSuccess ? CUFFT_SUCCESS : CUFFT_EXEC_FAILED;
}

cufftResult cufftDestroy(cufftHandle handle)
{
    std::shared_ptr<CufftCompatibilityPlan> plan;
    {
        std::lock_guard<std::mutex> lock(g_plan_mutex);
        const auto found = g_plans.find(handle);
        if (found == g_plans.end()) return CUFFT_INVALID_PLAN;
        plan = found->second;
        g_plans.erase(found);
    }
    FFTResult first = FFT_SUCCESS;
    if (plan->forward) first = fft_plan_destroy(plan->forward);
    if (plan->inverse) {
        const FFTResult second = fft_plan_destroy(plan->inverse);
        if (first == FFT_SUCCESS) first = second;
    }
    cudaError_t free_status = cudaSuccess;
    if (plan->compact_real_spectrum) {
        free_status = cudaFree(plan->compact_real_spectrum);
        plan->compact_real_spectrum = nullptr;
    }
    plan->forward = nullptr;
    plan->inverse = nullptr;
    if (first != FFT_SUCCESS) return map_fft_result(first);
    return free_status == cudaSuccess ? CUFFT_SUCCESS : CUFFT_INTERNAL_ERROR;
}

cufftResult cufftGetVersion(int* version)
{
    if (!version) return CUFFT_INVALID_VALUE;
    *version = kFftThrustCufftCompatibilityVersion;
    return CUFFT_SUCCESS;
}

}  // extern "C"
