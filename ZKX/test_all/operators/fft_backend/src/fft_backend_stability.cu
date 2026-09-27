#include <cusignal/backends/fft/fft_interface.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>
#include <malloc.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kStabilizationMaxBlocks = 10;
constexpr int kStabilizationExecutionsPerBlock = 5000;
constexpr int kRequiredStableBlocks = 3;
constexpr int kExecuteCheckpoints = 10;
constexpr int kExecutePerCheckpoint = 5000;
constexpr int kPlanWarmup = 512;
constexpr int kPlanCheckpoints = 8;
constexpr int kPlanPerCheckpoint = 512;
constexpr long long kAllowedCpuNoiseBytes = 4096;

struct DeviceCase {
    const char* name;
    int n;
    int batch;
    FFTType type = FFT_TYPE_C2C;
    FFTDirection direction = FFT_FORWARD;
    void* input = nullptr;
    void* output = nullptr;
};

void require_cuda(cudaError_t status, const char* step)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(step) + ": " + cudaGetErrorString(status));
    }
}

void require_fft(FFTResult status, const char* step)
{
    if (status != FFT_SUCCESS) {
        throw std::runtime_error(
            std::string(step) + ": status=" + std::to_string(static_cast<int>(status)));
    }
}

const char* backend_name()
{
#if defined(USE_DLFFT)
    return "dlfft";
#elif defined(USE_THRUST)
    return "thrust";
#else
    return "unknown";
#endif
}

std::size_t gpu_used_bytes()
{
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    require_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    return total_bytes - free_bytes;
}

void allocate_case(DeviceCase& test_case)
{
    const std::size_t real_count = static_cast<std::size_t>(test_case.n) * test_case.batch;
    const std::size_t spectrum_count =
        static_cast<std::size_t>(test_case.n / 2 + 1) * test_case.batch;
    const std::size_t input_bytes = test_case.type == FFT_TYPE_R2C
        ? real_count * sizeof(float)
        : (test_case.type == FFT_TYPE_C2R
            ? spectrum_count * sizeof(ComplexFloat)
            : real_count * sizeof(ComplexFloat));
    const std::size_t output_bytes = test_case.type == FFT_TYPE_C2R
        ? real_count * sizeof(float)
        : (test_case.type == FFT_TYPE_R2C
            ? spectrum_count * sizeof(ComplexFloat)
            : real_count * sizeof(ComplexFloat));
    require_cuda(cudaMalloc(&test_case.input, input_bytes), "allocate input");
    require_cuda(cudaMalloc(&test_case.output, output_bytes), "allocate output");
    require_cuda(cudaMemset(test_case.input, 0, input_bytes), "initialize input");
}

void release_case(DeviceCase& test_case)
{
    require_cuda(cudaFree(test_case.output), "free output");
    require_cuda(cudaFree(test_case.input), "free input");
}

bool execute_stability(DeviceCase& test_case)
{
    FFTPlanHandle plan = nullptr;
    require_fft(fft_plan_create_1d(
        &plan, test_case.n, test_case.batch, test_case.type, test_case.direction), "create plan");
    std::size_t previous_gpu = gpu_used_bytes();
    std::size_t previous_cpu = mallinfo2().uordblks;
    int stable_blocks = 0;
    int stabilization_blocks = 0;
    for (int block = 1; block <= kStabilizationMaxBlocks; ++block) {
        for (int i = 0; i < kStabilizationExecutionsPerBlock; ++i) {
            require_fft(fft_execute(plan, test_case.output, test_case.input), "stabilization execute");
            require_cuda(cudaDeviceSynchronize(), "stabilization execute synchronize");
        }
        const std::size_t current_gpu = gpu_used_bytes();
        const std::size_t current_cpu = mallinfo2().uordblks;
        const long long cpu_delta = static_cast<long long>(current_cpu) -
            static_cast<long long>(previous_cpu);
        const bool cpu_flat = cpu_delta >= -kAllowedCpuNoiseBytes &&
            cpu_delta <= kAllowedCpuNoiseBytes;
        const bool gpu_flat = current_gpu == previous_gpu;
        stable_blocks = cpu_flat && gpu_flat ? stable_blocks + 1 : 0;
        previous_cpu = current_cpu;
        previous_gpu = current_gpu;
        stabilization_blocks = block;
        if (stable_blocks >= kRequiredStableBlocks) break;
    }
    if (stable_blocks < kRequiredStableBlocks) {
        std::cout << "[FFT-STABILITY][STABILIZATION] backend=" << backend_name()
                  << " case=" << test_case.name
                  << " blocks=" << stabilization_blocks
                  << " status=failed\n";
        require_fft(fft_plan_destroy(plan), "destroy unstable plan");
        return false;
    }

    std::vector<std::size_t> cpu_samples(kExecuteCheckpoints + 1);
    std::vector<std::size_t> gpu_samples(kExecuteCheckpoints + 1);
    gpu_samples[0] = gpu_used_bytes();
    cpu_samples[0] = mallinfo2().uordblks;
    for (int checkpoint = 1; checkpoint <= kExecuteCheckpoints; ++checkpoint) {
        for (int iteration = 0; iteration < kExecutePerCheckpoint; ++iteration) {
            require_fft(fft_execute(plan, test_case.output, test_case.input), "measured execute");
            require_cuda(cudaDeviceSynchronize(), "measured execute synchronize");
        }
        gpu_samples[checkpoint] = gpu_used_bytes();
        cpu_samples[checkpoint] = mallinfo2().uordblks;
    }
    for (int checkpoint = 0; checkpoint <= kExecuteCheckpoints; ++checkpoint) {
        std::cout << "[FFT-STABILITY][EXECUTE_SAMPLE] backend=" << backend_name()
                  << " case=" << test_case.name
                  << " checkpoint=" << checkpoint
                  << " executions=" << checkpoint * kExecutePerCheckpoint
                  << " cpu_live_heap_bytes=" << cpu_samples[checkpoint]
                  << " gpu_used_bytes=" << gpu_samples[checkpoint] << '\n';
    }

    const long long cpu_growth = static_cast<long long>(cpu_samples.back()) -
        static_cast<long long>(cpu_samples.front());
    const long long gpu_growth = static_cast<long long>(gpu_samples.back()) -
        static_cast<long long>(gpu_samples.front());
    const bool passed = cpu_growth <= kAllowedCpuNoiseBytes && gpu_growth <= 0;
    std::cout << "[FFT-STABILITY][EXECUTE] backend=" << backend_name()
              << " case=" << test_case.name
              << " stabilization_blocks=" << stabilization_blocks
              << " iterations=" << kExecuteCheckpoints * kExecutePerCheckpoint
              << " cpu_growth_bytes=" << cpu_growth
              << " gpu_growth_bytes=" << gpu_growth
              << " status=" << (passed ? "passed" : "failed") << '\n';
    require_fft(fft_plan_destroy(plan), "destroy execute plan");
    return passed;
}

void run_plan_cycle(std::vector<DeviceCase>& cases)
{
    for (DeviceCase& test_case : cases) {
        FFTPlanHandle plan = nullptr;
        require_fft(fft_plan_create_1d(
            &plan, test_case.n, test_case.batch, test_case.type, test_case.direction),
            "create lifecycle plan");
        require_fft(fft_execute(plan, test_case.output, test_case.input),
            "lifecycle execute");
        require_cuda(cudaDeviceSynchronize(), "lifecycle synchronize");
        require_fft(fft_plan_destroy(plan), "destroy lifecycle plan");
    }
}

bool plan_lifecycle_stability(std::vector<DeviceCase>& cases)
{
    for (int i = 0; i < kPlanWarmup; ++i) run_plan_cycle(cases);
    std::vector<std::size_t> cpu_samples(kPlanCheckpoints + 1);
    std::vector<std::size_t> gpu_samples(kPlanCheckpoints + 1);
    gpu_samples[0] = gpu_used_bytes();
    cpu_samples[0] = mallinfo2().uordblks;
    for (int checkpoint = 1; checkpoint <= kPlanCheckpoints; ++checkpoint) {
        for (int i = 0; i < kPlanPerCheckpoint; ++i) run_plan_cycle(cases);
        gpu_samples[checkpoint] = gpu_used_bytes();
        cpu_samples[checkpoint] = mallinfo2().uordblks;
    }
    for (int checkpoint = 0; checkpoint <= kPlanCheckpoints; ++checkpoint) {
        std::cout << "[FFT-STABILITY][PLAN_SAMPLE] backend=" << backend_name()
                  << " checkpoint=" << checkpoint
                  << " cycles=" << checkpoint * kPlanPerCheckpoint
                  << " cpu_live_heap_bytes=" << cpu_samples[checkpoint]
                  << " gpu_used_bytes=" << gpu_samples[checkpoint] << '\n';
    }
    const long long cpu_growth = static_cast<long long>(cpu_samples.back()) -
        static_cast<long long>(cpu_samples.front());
    const long long gpu_growth = static_cast<long long>(gpu_samples.back()) -
        static_cast<long long>(gpu_samples.front());
    const bool passed = cpu_growth <= kAllowedCpuNoiseBytes && gpu_growth <= 0;
    std::cout << "[FFT-STABILITY][PLAN] backend=" << backend_name()
              << " warmup_cycles=" << kPlanWarmup
              << " measured_cycles=" << kPlanCheckpoints * kPlanPerCheckpoint
              << " cases_per_cycle=" << cases.size()
              << " cpu_growth_bytes=" << cpu_growth
              << " gpu_growth_bytes=" << gpu_growth
              << " status=" << (passed ? "passed" : "failed") << '\n';
    return passed;
}

}  // namespace

int main()
{
    try {
        std::vector<DeviceCase> cases{
            {"c2c_power2_8x1", 8, 1, FFT_TYPE_C2C, FFT_FORWARD},
            {"c2c_power2_64x15", 64, 15, FFT_TYPE_C2C, FFT_FORWARD},
            {"c2c_non_power_127x1", 127, 1, FFT_TYPE_C2C, FFT_FORWARD},
            {"c2c_power2_1024x16", 1024, 16, FFT_TYPE_C2C, FFT_FORWARD},
            {"r2c_power2_64x15", 64, 15, FFT_TYPE_R2C, FFT_FORWARD},
            {"c2r_power2_64x15", 64, 15, FFT_TYPE_C2R, FFT_INVERSE},
        };
        for (DeviceCase& test_case : cases) allocate_case(test_case);
        bool passed = true;
        for (DeviceCase& test_case : cases) passed = execute_stability(test_case) && passed;
        passed = plan_lifecycle_stability(cases) && passed;
        for (DeviceCase& test_case : cases) release_case(test_case);
        std::cout << "[FFT-STABILITY][SUMMARY] backend=" << backend_name()
                  << " status=" << (passed ? "passed" : "failed") << '\n';
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "[FFT-STABILITY][FAIL] reason=" << error.what() << '\n';
        return 1;
    }
}
