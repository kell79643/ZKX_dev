#include <cusignal/backends/fft/fft_interface.h>

#include <cuda_runtime.h>

#include <malloc.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

constexpr int kFftLength = 1024;
constexpr int kBatch = 1;
constexpr int kDefaultWarmupRounds = 100;
constexpr int kDefaultMeasuredRounds = 2000;
constexpr const char* kDtype = "complex64";
constexpr std::size_t kElementCount =
    static_cast<std::size_t>(kFftLength) * kBatch;

struct Options {
    std::string scenario;
    std::string output_path;
    int run_id = 0;
    int warmup_rounds = kDefaultWarmupRounds;
    int measured_rounds = kDefaultMeasuredRounds;
};

struct MemorySample {
    std::uint64_t cpu_live_bytes = 0;
    std::uint64_t vm_rss_bytes = 0;
    std::uint64_t gpu_used_bytes = 0;
    int cuda_status = static_cast<int>(cudaSuccess);
    bool proc_status_ok = false;
};

struct LifecycleRow {
    int run_id = 0;
    int iteration = 0;
    MemorySample before;
    MemorySample after_alloc;
    MemorySample after_execute;
    MemorySample after_free;
    std::int64_t cpu_growth = 0;
    std::int64_t cpu_step = 0;
    std::int64_t gpu_growth = 0;
    std::int64_t gpu_step = 0;
    int create_status = -1;
    int input_alloc_status = -1;
    int output_alloc_status = -1;
    int buffer_alloc_status = -1;
    int input_copy_status = -1;
    int initialization_sync_status = -1;
    int execute_status = -1;
    int execution_sync_status = -1;
    int destroy_status = -1;
    int input_free_status = -1;
    int output_free_status = -1;
    int buffer_free_status = -1;
    int post_free_sync_status = -1;
    int allocator_trim_status = -1;
    int memory_read_status = -1;
    std::string error_message;
};

struct ResidentRow {
    int run_id = 0;
    int iteration = 0;
    MemorySample sample;
    std::int64_t cpu_growth = 0;
    std::int64_t cpu_step = 0;
    std::int64_t gpu_growth = 0;
    std::int64_t gpu_step = 0;
    int execute_status = -1;
    int synchronize_status = -1;
    int memory_read_status = -1;
    std::string error_message;
};

const char* backend_name()
{
#if defined(USE_DLFFT)
    return "dlfft";
#elif defined(USE_THRUST)
    return "fft_thrust";
#else
    return "unknown";
#endif
}

std::int64_t difference(std::uint64_t end, std::uint64_t begin)
{
    if (end >= begin) {
        return static_cast<std::int64_t>(end - begin);
    }
    return -static_cast<std::int64_t>(begin - end);
}

std::string csv_escape(const std::string& value)
{
    if (value.find_first_of(",\"\r\n") == std::string::npos) {
        return value;
    }
    std::string escaped = "\"";
    for (char character : value) {
        escaped += character == '\"' ? "\"\"" : std::string(1, character);
    }
    escaped += '\"';
    return escaped;
}

void append_error(std::string& target, const char* operation, int status)
{
    if (!target.empty()) {
        target += "; ";
    }
    target += std::string(operation) + " status=" + std::to_string(status);
}

bool read_vm_rss(std::uint64_t& bytes)
{
    FILE* status_file = std::fopen("/proc/self/status", "r");
    if (status_file == nullptr) {
        return false;
    }
    char line[256];
    bool found = false;
    while (std::fgets(line, sizeof(line), status_file) != nullptr) {
        unsigned long long kib = 0;
        if (std::sscanf(line, "VmRSS: %llu kB", &kib) == 1) {
            bytes = static_cast<std::uint64_t>(kib) * 1024;
            found = true;
            break;
        }
    }
    std::fclose(status_file);
    return found;
}

MemorySample sample_memory()
{
    MemorySample sample;
    const struct mallinfo2 heap = mallinfo2();
    sample.cpu_live_bytes = static_cast<std::uint64_t>(heap.uordblks);
    sample.proc_status_ok = read_vm_rss(sample.vm_rss_bytes);
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    sample.cuda_status = static_cast<int>(status);
    if (status == cudaSuccess) {
        sample.gpu_used_bytes = static_cast<std::uint64_t>(total_bytes - free_bytes);
    }
    return sample;
}

bool sample_ok(const MemorySample& sample)
{
    return sample.cuda_status == static_cast<int>(cudaSuccess) &&
           sample.proc_status_ok;
}

std::array<ComplexFloat, kElementCount> make_input()
{
    std::array<ComplexFloat, kElementCount> input{};
    for (int batch_index = 0; batch_index < kBatch; ++batch_index) {
        for (int index = 0; index < kFftLength; ++index) {
            const std::uint32_t seed = static_cast<std::uint32_t>(
                (index + 1) * 1664525U + (batch_index + 7) * 1013904223U);
            const int real_code = static_cast<int>((seed >> 8U) & 0x3ffU) - 512;
            const int imag_code = static_cast<int>((seed >> 18U) & 0x3ffU) - 512;
            const std::size_t offset =
                static_cast<std::size_t>(batch_index) * kFftLength + index;
            input[offset].re = static_cast<float>(real_code) / 512.0F;
            input[offset].im = static_cast<float>(imag_code) / 512.0F;
        }
    }
    return input;
}

FFTResult create_execution_object(FFTPlanHandle* plan)
{
    return fft_plan_create_1d(
        plan, kFftLength, kBatch, FFT_TYPE_C2C, FFT_FORWARD);
}

FFTResult execute_fft(
    FFTPlanHandle plan,
    ComplexFloat* output,
    const ComplexFloat* input)
{
    return fft_execute(plan, output, input);
}

FFTResult destroy_execution_object(FFTPlanHandle plan)
{
    return fft_plan_destroy(plan);
}

void write_lifecycle_header(std::ofstream& output)
{
    output
        << "run_id,iteration,backend,fft_length,batch,dtype,"
        << "cpu_before_create_bytes,cpu_after_alloc_bytes,cpu_after_execute_bytes,cpu_after_free_bytes,"
        << "cpu_after_free_growth_from_baseline_bytes,cpu_after_free_step_growth_bytes,"
        << "gpu_before_create_bytes,gpu_after_alloc_bytes,gpu_after_execute_bytes,gpu_after_free_bytes,"
        << "gpu_after_free_growth_from_baseline_bytes,gpu_after_free_step_growth_bytes,"
        << "vmrss_before_create_bytes,vmrss_after_alloc_bytes,vmrss_after_execute_bytes,vmrss_after_free_bytes,"
        << "create_status,input_alloc_status,output_alloc_status,buffer_alloc_status,input_copy_status,"
        << "initialization_sync_status,execute_status,execution_sync_status,destroy_status,"
        << "input_free_status,output_free_status,buffer_free_status,post_free_sync_status,"
        << "allocator_trim_status,memory_read_status,error_message\n";
}

void write_lifecycle_row(std::ofstream& output, const LifecycleRow& row)
{
    output
        << row.run_id << ',' << row.iteration << ',' << backend_name() << ','
        << kFftLength << ',' << kBatch << ',' << kDtype << ','
        << row.before.cpu_live_bytes << ',' << row.after_alloc.cpu_live_bytes << ','
        << row.after_execute.cpu_live_bytes << ',' << row.after_free.cpu_live_bytes << ','
        << row.cpu_growth << ',' << row.cpu_step << ','
        << row.before.gpu_used_bytes << ',' << row.after_alloc.gpu_used_bytes << ','
        << row.after_execute.gpu_used_bytes << ',' << row.after_free.gpu_used_bytes << ','
        << row.gpu_growth << ',' << row.gpu_step << ','
        << row.before.vm_rss_bytes << ',' << row.after_alloc.vm_rss_bytes << ','
        << row.after_execute.vm_rss_bytes << ',' << row.after_free.vm_rss_bytes << ','
        << row.create_status << ',' << row.input_alloc_status << ','
        << row.output_alloc_status << ',' << row.buffer_alloc_status << ','
        << row.input_copy_status << ',' << row.initialization_sync_status << ','
        << row.execute_status << ',' << row.execution_sync_status << ','
        << row.destroy_status << ',' << row.input_free_status << ','
        << row.output_free_status << ',' << row.buffer_free_status << ','
        << row.post_free_sync_status << ',' << row.allocator_trim_status << ','
        << row.memory_read_status << ',' << csv_escape(row.error_message) << '\n';
    output.flush();
}

bool run_iteration(
    LifecycleRow& row,
    const MemorySample& baseline,
    const MemorySample& previous_after_free)
{
    const auto host_input = make_input();
    const std::size_t bytes = host_input.size() * sizeof(ComplexFloat);
    FFTPlanHandle plan = nullptr;
    ComplexFloat* device_input = nullptr;
    ComplexFloat* device_output = nullptr;
    bool ok = true;

    row.before = sample_memory();
    if (!sample_ok(row.before)) {
        append_error(row.error_message, "before_create memory read", row.before.cuda_status);
        ok = false;
    }
    if (ok) {
        const FFTResult status = create_execution_object(&plan);
        row.create_status = static_cast<int>(status);
        if (status != FFT_SUCCESS) {
            append_error(row.error_message, "fft_plan_create_1d", row.create_status);
            ok = false;
        }
    }
    if (ok) {
        const cudaError_t status = cudaMalloc(
            reinterpret_cast<void**>(&device_input), bytes);
        row.input_alloc_status = static_cast<int>(status);
        if (status != cudaSuccess) {
            append_error(row.error_message, "cudaMalloc input", row.input_alloc_status);
            ok = false;
        }
    }
    if (ok) {
        const cudaError_t status = cudaMalloc(
            reinterpret_cast<void**>(&device_output), bytes);
        row.output_alloc_status = static_cast<int>(status);
        if (status != cudaSuccess) {
            append_error(row.error_message, "cudaMalloc output", row.output_alloc_status);
            ok = false;
        }
    }
    row.buffer_alloc_status =
        row.input_alloc_status == 0 && row.output_alloc_status == 0 ? 0 : -1;
    if (ok) {
        const cudaError_t status = cudaMemcpy(
            device_input, host_input.data(), bytes, cudaMemcpyHostToDevice);
        row.input_copy_status = static_cast<int>(status);
        if (status != cudaSuccess) {
            append_error(row.error_message, "cudaMemcpy input", row.input_copy_status);
            ok = false;
        }
    }
    if (ok) {
        const cudaError_t status = cudaDeviceSynchronize();
        row.initialization_sync_status = static_cast<int>(status);
        if (status != cudaSuccess) {
            append_error(row.error_message, "initialization synchronize", row.initialization_sync_status);
            ok = false;
        }
    }
    if (ok) {
        row.after_alloc = sample_memory();
        if (!sample_ok(row.after_alloc)) {
            append_error(row.error_message, "after_create_and_alloc memory read", row.after_alloc.cuda_status);
            ok = false;
        }
    }
    if (ok) {
        const FFTResult status = execute_fft(plan, device_output, device_input);
        row.execute_status = static_cast<int>(status);
        if (status != FFT_SUCCESS) {
            append_error(row.error_message, "fft_execute", row.execute_status);
            ok = false;
        }
    }
    if (ok) {
        const cudaError_t status = cudaDeviceSynchronize();
        row.execution_sync_status = static_cast<int>(status);
        if (status != cudaSuccess) {
            append_error(row.error_message, "execution synchronize", row.execution_sync_status);
            ok = false;
        }
    }
    if (ok) {
        row.after_execute = sample_memory();
        if (!sample_ok(row.after_execute)) {
            append_error(row.error_message, "after_execute_before_free memory read", row.after_execute.cuda_status);
            ok = false;
        }
    }

    if (plan != nullptr) {
        const FFTResult status = destroy_execution_object(plan);
        row.destroy_status = static_cast<int>(status);
        if (status != FFT_SUCCESS) {
            append_error(row.error_message, "fft_plan_destroy", row.destroy_status);
            ok = false;
        }
        plan = nullptr;
    }
    if (device_output != nullptr) {
        const cudaError_t status = cudaFree(device_output);
        row.output_free_status = static_cast<int>(status);
        if (status != cudaSuccess) {
            append_error(row.error_message, "cudaFree output", row.output_free_status);
            ok = false;
        }
    }
    if (device_input != nullptr) {
        const cudaError_t status = cudaFree(device_input);
        row.input_free_status = static_cast<int>(status);
        if (status != cudaSuccess) {
            append_error(row.error_message, "cudaFree input", row.input_free_status);
            ok = false;
        }
    }
    row.buffer_free_status =
        row.input_free_status == 0 && row.output_free_status == 0 ? 0 : -1;
    const cudaError_t post_free_status = cudaDeviceSynchronize();
    row.post_free_sync_status = static_cast<int>(post_free_status);
    if (post_free_status != cudaSuccess) {
        append_error(row.error_message, "post-free synchronize", row.post_free_sync_status);
        ok = false;
    }
    row.allocator_trim_status = malloc_trim(0);
    row.after_free = sample_memory();
    if (!sample_ok(row.after_free)) {
        append_error(row.error_message, "after_destroy_and_free memory read", row.after_free.cuda_status);
        ok = false;
    }
    row.memory_read_status =
        sample_ok(row.before) && sample_ok(row.after_alloc) &&
                sample_ok(row.after_execute) && sample_ok(row.after_free)
            ? 0
            : -1;
    row.cpu_growth = difference(row.after_free.cpu_live_bytes, baseline.cpu_live_bytes);
    row.cpu_step = difference(
        row.after_free.cpu_live_bytes, previous_after_free.cpu_live_bytes);
    row.gpu_growth = difference(row.after_free.gpu_used_bytes, baseline.gpu_used_bytes);
    row.gpu_step = difference(
        row.after_free.gpu_used_bytes, previous_after_free.gpu_used_bytes);
    return ok;
}

void write_failure_sidecar(
    const Options& options,
    const char* suffix,
    LifecycleRow& row)
{
    std::ofstream output(options.output_path + suffix);
    if (output) {
        write_lifecycle_header(output);
        write_lifecycle_row(output, row);
    }
}

int run_lifecycle(const Options& options)
{
    std::ofstream output(options.output_path);
    if (!output) {
        std::cerr << "cannot open output CSV: " << options.output_path << '\n';
        return 73;
    }
    write_lifecycle_header(output);
    MemorySample dummy{};
    for (int round = 0; round < options.warmup_rounds; ++round) {
        LifecycleRow row;
        row.run_id = options.run_id;
        row.iteration = -(round + 1);
        if (!run_iteration(row, dummy, dummy)) {
            write_failure_sidecar(options, ".warmup_error.csv", row);
            std::cerr << "warmup failed: " << row.error_message << '\n';
            return 1;
        }
    }
    const cudaError_t baseline_sync = cudaDeviceSynchronize();
    if (baseline_sync != cudaSuccess) {
        std::cerr << "baseline synchronize failed\n";
        return 1;
    }
    (void)malloc_trim(0);
    const MemorySample baseline = sample_memory();
    if (!sample_ok(baseline)) {
        LifecycleRow row;
        row.run_id = options.run_id;
        row.iteration = 0;
        row.before = baseline;
        row.after_free = baseline;
        row.memory_read_status = -1;
        append_error(row.error_message, "formal baseline memory read", baseline.cuda_status);
        write_failure_sidecar(options, ".baseline_error.csv", row);
        return 1;
    }
    MemorySample previous = baseline;
    for (int iteration = 1; iteration <= options.measured_rounds; ++iteration) {
        LifecycleRow row;
        row.run_id = options.run_id;
        row.iteration = iteration;
        const bool ok = run_iteration(row, baseline, previous);
        write_lifecycle_row(output, row);
        if (!ok) {
            std::cerr << "formal iteration failed: " << row.error_message << '\n';
            return 1;
        }
        previous = row.after_free;
    }
    std::cout << "[FFT-BACKEND-MEMORY][LIFECYCLE] backend=" << backend_name()
              << " run_id=" << options.run_id
              << " warmup=" << options.warmup_rounds
              << " iterations=" << options.measured_rounds
              << " status=complete\n";
    return 0;
}

void write_resident_header(std::ofstream& output)
{
    output
        << "run_id,iteration,backend,fft_length,batch,dtype,cpu_current_bytes,"
        << "cpu_growth_from_baseline_bytes,cpu_step_growth_bytes,vmrss_current_bytes,"
        << "gpu_current_bytes,gpu_growth_from_baseline_bytes,gpu_step_growth_bytes,"
        << "execute_status,synchronize_status,memory_read_status,error_message\n";
}

void write_resident_row(std::ofstream& output, const ResidentRow& row)
{
    output << row.run_id << ',' << row.iteration << ',' << backend_name() << ','
           << kFftLength << ',' << kBatch << ',' << kDtype << ','
           << row.sample.cpu_live_bytes << ',' << row.cpu_growth << ',' << row.cpu_step << ','
           << row.sample.vm_rss_bytes << ',' << row.sample.gpu_used_bytes << ','
           << row.gpu_growth << ',' << row.gpu_step << ',' << row.execute_status << ','
           << row.synchronize_status << ',' << row.memory_read_status << ','
           << csv_escape(row.error_message) << '\n';
    output.flush();
}

int run_resident(const Options& options)
{
    std::ofstream output(options.output_path);
    if (!output) {
        return 73;
    }
    write_resident_header(output);
    const auto host_input = make_input();
    const std::size_t bytes = host_input.size() * sizeof(ComplexFloat);
    FFTPlanHandle plan = nullptr;
    ComplexFloat* device_input = nullptr;
    ComplexFloat* device_output = nullptr;
    std::string setup_error;
    int create_status = static_cast<int>(create_execution_object(&plan));
    int input_alloc_status = static_cast<int>(cudaMalloc(
        reinterpret_cast<void**>(&device_input), bytes));
    int output_alloc_status = static_cast<int>(cudaMalloc(
        reinterpret_cast<void**>(&device_output), bytes));
    int copy_status = static_cast<int>(cudaMemcpy(
        device_input, host_input.data(), bytes, cudaMemcpyHostToDevice));
    int sync_status = static_cast<int>(cudaDeviceSynchronize());
    if (create_status != 0) append_error(setup_error, "create", create_status);
    if (input_alloc_status != 0) append_error(setup_error, "allocate input", input_alloc_status);
    if (output_alloc_status != 0) append_error(setup_error, "allocate output", output_alloc_status);
    if (copy_status != 0) append_error(setup_error, "copy input", copy_status);
    if (sync_status != 0) append_error(setup_error, "setup synchronize", sync_status);
    if (!setup_error.empty()) {
        std::ofstream sidecar(options.output_path + ".setup_error.csv");
        sidecar << "backend,create_status,input_alloc_status,output_alloc_status,input_copy_status,"
                   "synchronize_status,error_message\n"
                << backend_name() << ',' << create_status << ',' << input_alloc_status << ','
                << output_alloc_status << ',' << copy_status << ',' << sync_status << ','
                << csv_escape(setup_error) << '\n';
        return 1;
    }
    for (int round = 0; round < options.warmup_rounds; ++round) {
        if (execute_fft(plan, device_output, device_input) != FFT_SUCCESS ||
            cudaDeviceSynchronize() != cudaSuccess) {
            std::cerr << "resident warmup failed\n";
            return 1;
        }
    }
    (void)malloc_trim(0);
    const MemorySample baseline = sample_memory();
    if (!sample_ok(baseline)) {
        std::cerr << "resident baseline memory read failed\n";
        return 1;
    }
    MemorySample previous = baseline;
    for (int iteration = 1; iteration <= options.measured_rounds; ++iteration) {
        ResidentRow row;
        row.run_id = options.run_id;
        row.iteration = iteration;
        const FFTResult execute_status = execute_fft(plan, device_output, device_input);
        row.execute_status = static_cast<int>(execute_status);
        if (execute_status != FFT_SUCCESS) {
            append_error(row.error_message, "fft_execute", row.execute_status);
        }
        const cudaError_t synchronize_status = cudaDeviceSynchronize();
        row.synchronize_status = static_cast<int>(synchronize_status);
        if (synchronize_status != cudaSuccess) {
            append_error(row.error_message, "execute synchronize", row.synchronize_status);
        }
        row.sample = sample_memory();
        row.memory_read_status = sample_ok(row.sample) ? 0 : -1;
        if (!sample_ok(row.sample)) {
            append_error(row.error_message, "memory read", row.sample.cuda_status);
        }
        row.cpu_growth = difference(row.sample.cpu_live_bytes, baseline.cpu_live_bytes);
        row.cpu_step = difference(row.sample.cpu_live_bytes, previous.cpu_live_bytes);
        row.gpu_growth = difference(row.sample.gpu_used_bytes, baseline.gpu_used_bytes);
        row.gpu_step = difference(row.sample.gpu_used_bytes, previous.gpu_used_bytes);
        write_resident_row(output, row);
        if (!row.error_message.empty()) {
            return 1;
        }
        previous = row.sample;
    }
    const int destroy_status = static_cast<int>(destroy_execution_object(plan));
    const int output_free_status = static_cast<int>(cudaFree(device_output));
    const int input_free_status = static_cast<int>(cudaFree(device_input));
    const int cleanup_sync_status = static_cast<int>(cudaDeviceSynchronize());
    std::ofstream cleanup(options.output_path + ".cleanup.csv");
    cleanup << "backend,destroy_status,output_free_status,input_free_status,"
               "synchronize_status,error_message\n";
    std::string cleanup_error;
    if (destroy_status != 0) append_error(cleanup_error, "destroy", destroy_status);
    if (output_free_status != 0) append_error(cleanup_error, "free output", output_free_status);
    if (input_free_status != 0) append_error(cleanup_error, "free input", input_free_status);
    if (cleanup_sync_status != 0) append_error(cleanup_error, "cleanup synchronize", cleanup_sync_status);
    cleanup << backend_name() << ',' << destroy_status << ',' << output_free_status << ','
            << input_free_status << ',' << cleanup_sync_status << ','
            << csv_escape(cleanup_error) << '\n';
    if (!cleanup_error.empty()) {
        return 1;
    }
    std::cout << "[FFT-BACKEND-MEMORY][RESIDENT] backend=" << backend_name()
              << " run_id=" << options.run_id
              << " warmup=" << options.warmup_rounds
              << " iterations=" << options.measured_rounds
              << " status=complete\n";
    return 0;
}

int write_verification_output(const Options& options)
{
    const auto host_input = make_input();
    std::array<ComplexFloat, kElementCount> host_output{};
    const std::size_t bytes = host_input.size() * sizeof(ComplexFloat);
    FFTPlanHandle plan = nullptr;
    ComplexFloat* device_input = nullptr;
    ComplexFloat* device_output = nullptr;
    bool ok = create_execution_object(&plan) == FFT_SUCCESS &&
              cudaMalloc(reinterpret_cast<void**>(&device_input), bytes) == cudaSuccess &&
              cudaMalloc(reinterpret_cast<void**>(&device_output), bytes) == cudaSuccess &&
              cudaMemcpy(device_input, host_input.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaDeviceSynchronize() == cudaSuccess &&
              execute_fft(plan, device_output, device_input) == FFT_SUCCESS &&
              cudaDeviceSynchronize() == cudaSuccess &&
              cudaMemcpy(host_output.data(), device_output, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    if (plan != nullptr) ok = destroy_execution_object(plan) == FFT_SUCCESS && ok;
    if (device_output != nullptr) ok = cudaFree(device_output) == cudaSuccess && ok;
    if (device_input != nullptr) ok = cudaFree(device_input) == cudaSuccess && ok;
    ok = cudaDeviceSynchronize() == cudaSuccess && ok;
    if (!ok) {
        return 1;
    }
    std::ofstream output(options.output_path);
    output << "index,input_real,input_imag,output_real,output_imag\n";
    for (std::size_t index = 0; index < host_input.size(); ++index) {
        output << index << ',' << host_input[index].re << ',' << host_input[index].im << ','
               << host_output[index].re << ',' << host_output[index].im << '\n';
    }
    return output ? 0 : 73;
}

bool parse_positive(const char* text, int& value)
{
    try {
        std::size_t consumed = 0;
        const std::string input(text);
        value = std::stoi(input, &consumed);
        return consumed == input.size() && value > 0;
    } catch (...) {
        return false;
    }
}

bool parse_options(int argc, char** argv, Options& options)
{
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--scenario" && index + 1 < argc) {
            options.scenario = argv[++index];
        } else if (argument == "--output" && index + 1 < argc) {
            options.output_path = argv[++index];
        } else if (argument == "--run-id" && index + 1 < argc) {
            if (!parse_positive(argv[++index], options.run_id)) return false;
        } else if (argument == "--warmup" && index + 1 < argc) {
            if (!parse_positive(argv[++index], options.warmup_rounds)) return false;
        } else if (argument == "--iterations" && index + 1 < argc) {
            if (!parse_positive(argv[++index], options.measured_rounds)) return false;
        } else {
            return false;
        }
    }
    if (options.output_path.empty()) return false;
    if (options.scenario == "verify") return true;
    return options.run_id > 0 &&
           (options.scenario == "lifecycle" || options.scenario == "resident");
}

}  // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!parse_options(argc, argv, options)) {
        std::cerr << "usage: fft_backend_memory_unified --scenario verify|lifecycle|resident "
                     "--output FILE [--run-id N] [--warmup N] [--iterations N]\n";
        return 64;
    }
    if (options.scenario == "verify") return write_verification_output(options);
    if (options.scenario == "lifecycle") return run_lifecycle(options);
    return run_resident(options);
}
