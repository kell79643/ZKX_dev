#include <cusignal/backends/random/rand_interface.h>

#include <curand.h>

#include <stdexcept>
#include <string>

namespace cusignal {
namespace {

void require_curand(curandStatus_t status, const char* step)
{
    if (status != CURAND_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(step) + ": curand status " + std::to_string(static_cast<int>(status)));
    }
}

curandGenerator_t create_generator(unsigned long long seed, cudaStream_t stream)
{
    curandGenerator_t generator = nullptr;
    require_curand(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32), "curandCreateGenerator");
    if (stream != nullptr) {
        require_curand(curandSetStream(generator, stream), "curandSetStream");
    }
    require_curand(
        curandSetPseudoRandomGeneratorSeed(generator, seed),
        "curandSetPseudoRandomGeneratorSeed");
    return generator;
}

}  // namespace

void rand_generate_uniform_device(
    float* d_output,
    std::size_t count,
    unsigned long long seed,
    cudaStream_t stream)
{
    if (d_output == nullptr && count != 0U) {
        throw std::invalid_argument("rand_generate_uniform_device requires non-null output");
    }
    if (count == 0U) {
        return;
    }

    curandGenerator_t generator = create_generator(seed, stream);
    require_curand(curandGenerateUniform(generator, d_output, count), "curandGenerateUniform");
    require_curand(curandDestroyGenerator(generator), "curandDestroyGenerator");
}

void rand_generate_normal_device(
    float* d_output,
    std::size_t count,
    float mean,
    float stddev,
    unsigned long long seed,
    cudaStream_t stream)
{
    if (d_output == nullptr && count != 0U) {
        throw std::invalid_argument("rand_generate_normal_device requires non-null output");
    }
    if (count == 0U) {
        return;
    }

    curandGenerator_t generator = create_generator(seed, stream);
    require_curand(curandGenerateNormal(generator, d_output, count, mean, stddev), "curandGenerateNormal");
    require_curand(curandDestroyGenerator(generator), "curandDestroyGenerator");
}

}  // namespace cusignal
