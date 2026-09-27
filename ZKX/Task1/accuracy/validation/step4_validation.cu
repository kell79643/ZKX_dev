#include "accuracy/validation/step4_validation.h"

#include <cusignal/runtime/device_array.h>

namespace task1::accuracy {
namespace {

std::uint32_t noise_bits(std::uint32_t index)
{
    std::uint32_t value = 0x9e3779b9U ^ (index * 747796405U + 2891336453U);
    value ^= value >> 16;
    value *= 2246822519U;
    value ^= value >> 13;
    value *= 3266489917U;
    return value ^ (value >> 16);
}

}  // namespace

Step4ValidationData generate_step4_cpu_reference(const PipelineState& state)
{
    const auto& config = state.config;
    Step4ValidationData data;
    const int count = config.num_pulses * config.samples_per_pulse;
    cusignal::CaCfarOptions options;
    options.guard_cells = {config.cfar_guard_doppler, config.cfar_guard_range};
    options.reference_cells = {config.cfar_reference_doppler, config.cfar_reference_range};
    options.pfa = config.pfa;
    const int od = config.cfar_guard_doppler + config.cfar_reference_doppler;
    const int orng = config.cfar_guard_range + config.cfar_reference_range;
    const int reference_count = (2 * od + 1) * (2 * orng + 1) -
        (2 * config.cfar_guard_doppler + 1) * (2 * config.cfar_guard_range + 1);
    data.power_cpu.resize(count);
    for (int index = 0; index < count; ++index) {
        const auto value = state.range_doppler[index];
        data.power_cpu[index] = value.real * value.real + value.imag * value.imag;
    }
    const auto cpu = cusignal::ca_cfar_typed_cpu<float>(
        data.power_cpu, {config.num_pulses, config.samples_per_pulse}, options);
    data.threshold_cpu = cpu.threshold;
    data.detections_cpu = cpu.detections;
    data.detections_same_power_cpu = cpu.detections;
    data.alpha_cpu = cusignal::cfar_alpha_typed_cpu<float>(config.pfa, reference_count);

    return data;
}

Step4ValidationData collect_step4_validation(const PipelineState& state)
{
    const auto& config = state.config;
    Step4ValidationData data = generate_step4_cpu_reference(state);
    const int count = config.num_pulses * config.samples_per_pulse;
    cusignal::CaCfarOptions options;
    options.guard_cells = {config.cfar_guard_doppler, config.cfar_guard_range};
    options.reference_cells = {config.cfar_reference_doppler, config.cfar_reference_range};
    options.pfa = config.pfa;
    data.detections_same_power_cpu =
        generate_step4_same_power_detections_cpu(state);

    std::vector<float> noise_only(count);
    for (int index = 0; index < count; ++index)
        noise_only[index] = 0.75F +
            static_cast<float>(noise_bits(static_cast<std::uint32_t>(index)) & 0xffffU) /
                65535.0F * 0.5F;
    auto d_noise_only = cusignal::DeviceArray<float>::from_host(noise_only);
    data.noise_detections_gpu = cusignal::ca_cfar_device<float>(
        d_noise_only, {config.num_pulses, config.samples_per_pulse}, options).detections.to_host();
    const int interference_doppler = config.num_pulses * 3 / 8;
    const int interference_range = config.samples_per_pulse * 2 / 3;
    noise_only[static_cast<std::size_t>(interference_doppler) *
        config.samples_per_pulse + interference_range] = 100.0F;
    auto d_interference = cusignal::DeviceArray<float>::from_host(noise_only);
    data.interference_detections_gpu = cusignal::ca_cfar_device<float>(
        d_interference, {config.num_pulses, config.samples_per_pulse}, options).detections.to_host();
    options.pfa = 1.0e-5;
    const auto stricter = cusignal::ca_cfar_typed_cpu<float>(
        data.power_cpu, {config.num_pulses, config.samples_per_pulse}, options);
    const int target_doppler_index = config.doppler_bin >= 0
        ? config.doppler_bin : config.num_pulses + config.doppler_bin;
    const std::size_t target_index = static_cast<std::size_t>(target_doppler_index) *
        config.samples_per_pulse + config.target_delay_samples;
    data.stricter_target_threshold_cpu = stricter.threshold[target_index];
    return data;
}

std::vector<cusignal::CfarDetection> generate_step4_same_power_detections_cpu(
    const PipelineState& state)
{
    const auto& config = state.config;
    cusignal::CaCfarOptions options;
    options.guard_cells = {config.cfar_guard_doppler, config.cfar_guard_range};
    options.reference_cells = {
        config.cfar_reference_doppler, config.cfar_reference_range};
    options.pfa = config.pfa;
    return cusignal::ca_cfar_typed_cpu<float>(
        state.power, {config.num_pulses, config.samples_per_pulse}, options).detections;
}

}  // namespace task1::accuracy
