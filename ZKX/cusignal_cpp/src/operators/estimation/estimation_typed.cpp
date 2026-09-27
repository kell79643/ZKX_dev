#include <cusignal/operators/estimation/estimation_typed.h>

#include <limits>
#include <stdexcept>
#include <utility>

namespace cusignal {
namespace {

std::size_t checked_product(std::size_t left, std::size_t right, const char* message)
{
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::invalid_argument(message);
    }
    return left * right;
}

bool same_layout(const KalmanLayout& left, const KalmanLayout& right)
{
    return left.dim_x == right.dim_x && left.dim_z == right.dim_z &&
           left.points == right.points && left.x_count == right.x_count &&
           left.p_count == right.p_count && left.h_count == right.h_count &&
           left.r_count == right.r_count && left.z_count == right.z_count;
}

KalmanLayout validated_layout(const KalmanLayout& layout)
{
    KalmanLayout validated(layout.dim_x, layout.dim_z, layout.points);
    if (!same_layout(layout, validated)) {
        throw std::invalid_argument("Kalman layout was mutated");
    }
    return validated;
}

void validate_host_state(const KalmanHostState& state)
{
    const KalmanLayout layout = validated_layout(state.layout);
    if (state.x.size() != layout.x_count || state.p.size() != layout.p_count) {
        throw std::invalid_argument("Kalman Host state shape mismatch");
    }
}

template <typename T>
float load(T value)
{
    return detail::SimpleSignalTypePolicy<T>::load(value);
}

template <typename T>
std::vector<float> load_vector(const std::vector<T>& input)
{
    std::vector<float> output(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        output[index] = load(input[index]);
    }
    return output;
}

void inverse_fixed_cusignal(std::vector<float>& matrix, int size)
{
    std::vector<float> inverse(static_cast<std::size_t>(size) * size, 0.0F);
    for (int row = 0; row < size; ++row) inverse[row * size + row] = 1.0F;

    // 固定版raw kernel只执行一次从末行到首行的相邻比较，使最大signed首列值冒泡到第0行。
    for (int row = size - 1; row > 0; --row) {
        if (matrix[(row - 1) * size] < matrix[row * size]) {
            for (int col = 0; col < size; ++col) {
                std::swap(matrix[(row - 1) * size + col], matrix[row * size + col]);
                std::swap(inverse[(row - 1) * size + col], inverse[row * size + col]);
            }
        }
    }

    for (int pivot = 0; pivot < size; ++pivot) {
        for (int row = 0; row < size; ++row) {
            if (row == pivot) continue;
            const float factor =
                matrix[row * size + pivot] / matrix[pivot * size + pivot];
            for (int col = 0; col < size; ++col) {
                matrix[row * size + col] -= matrix[pivot * size + col] * factor;
                inverse[row * size + col] -= inverse[pivot * size + col] * factor;
            }
        }
    }
    for (int row = 0; row < size; ++row) {
        const float diagonal = matrix[row * size + row];
        for (int col = 0; col < size; ++col) {
            matrix[row * size + col] /= diagonal;
            inverse[row * size + col] /= diagonal;
        }
    }
    matrix = std::move(inverse);
}

}  // namespace

void KalmanLayout::reset(int dim_x_in, int dim_z_in, int points_in)
{
    if (dim_x_in < 1 || dim_z_in < 1 || points_in < 0) {
        throw std::invalid_argument("Kalman dimensions require dim_x>=1, dim_z>=1, points>=0");
    }
    dim_x = dim_x_in;
    dim_z = dim_z_in;
    points = points_in;
    const std::size_t batch = static_cast<std::size_t>(points);
    x_count = checked_product(batch, static_cast<std::size_t>(dim_x),
                              "Kalman x shape exceeds size_t capacity");
    p_count = checked_product(x_count, static_cast<std::size_t>(dim_x),
                              "Kalman P shape exceeds size_t capacity");
    h_count = checked_product(
        checked_product(batch, static_cast<std::size_t>(dim_z),
                        "Kalman H shape exceeds size_t capacity"),
        static_cast<std::size_t>(dim_x), "Kalman H shape exceeds size_t capacity");
    r_count = checked_product(
        checked_product(batch, static_cast<std::size_t>(dim_z),
                        "Kalman R shape exceeds size_t capacity"),
        static_cast<std::size_t>(dim_z), "Kalman R shape exceeds size_t capacity");
    z_count = checked_product(batch, static_cast<std::size_t>(dim_z),
                              "Kalman z shape exceeds size_t capacity");
}

void KalmanDeviceState::reset(const KalmanLayout& layout_in)
{
    layout = validated_layout(layout_in);
    x.reset(layout.x_count);
    p.reset(layout.p_count);
}

void KalmanDeviceWorkspace::reset(const KalmanLayout& layout_in)
{
    layout = validated_layout(layout_in);
    f.reset(layout.p_count);
    q.reset(layout.p_count);
    alpha_sq.reset(static_cast<std::size_t>(layout.points));
    h.reset(layout.h_count);
    r.reset(layout.r_count);
    z.reset(layout.z_count);
    next_x.reset(layout.x_count);
    next_p.reset(layout.p_count);
    fp.reset(layout.p_count);
    pht.reset(layout.h_count);
    innovation.reset(layout.z_count);
    s.reset(layout.r_count);
    augmented.reset(checked_product(layout.r_count, 2, "Kalman inverse workspace exceeds size_t capacity"));
    gain.reset(layout.h_count);
    i_kh.reset(layout.p_count);
    left_covariance.reset(layout.p_count);
}

template <typename T>
KalmanHostState make_kalman_host_state(
    const std::vector<T>& x,
    const std::vector<T>& p,
    const KalmanLayout& layout)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported Kalman business dtype");
    const KalmanLayout validated = validated_layout(layout);
    if (x.size() != validated.x_count || p.size() != validated.p_count) {
        throw std::invalid_argument("make_kalman_host_state shape mismatch");
    }
    KalmanHostState state;
    state.layout = validated;
    state.x = load_vector(x);
    state.p = load_vector(p);
    return state;
}

template <typename T>
void kalman_predict_typed_cpu(
    KalmanHostState& state,
    const std::vector<T>& f,
    const std::vector<T>& q,
    const std::vector<T>& alpha_sq,
    const KalmanPredictOptions& options)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported Kalman business dtype");
    if (options.control_provided) {
        throw std::logic_error("Kalman control input is not implemented by fixed cuSignal");
    }
    validate_host_state(state);
    const KalmanLayout& layout = state.layout;
    if (f.size() != layout.p_count ||
        (!options.process_noise_scalar && q.size() != layout.p_count) ||
        alpha_sq.size() != static_cast<std::size_t>(layout.points)) {
        throw std::invalid_argument("kalman_predict_typed_cpu shape mismatch");
    }

    const std::vector<float> ff = load_vector(f);
    const std::vector<float> qf = options.process_noise_scalar ? std::vector<float>{} : load_vector(q);
    const std::vector<float> af = load_vector(alpha_sq);
    std::vector<float> next_x(layout.x_count, 0.0F);
    std::vector<float> fp(layout.p_count, 0.0F);
    std::vector<float> next_p(layout.p_count, 0.0F);
    const int nx = layout.dim_x;

    for (int point = 0; point < layout.points; ++point) {
        const std::size_t x_base = static_cast<std::size_t>(point) * nx;
        const std::size_t p_base = x_base * nx;
        for (int row = 0; row < nx; ++row) {
            for (int inner = 0; inner < nx; ++inner) {
                next_x[x_base + row] +=
                    ff[p_base + row * nx + inner] * state.x[x_base + inner];
            }
        }
        for (int row = 0; row < nx; ++row) {
            for (int col = 0; col < nx; ++col) {
                for (int inner = 0; inner < nx; ++inner) {
                    fp[p_base + row * nx + col] +=
                        ff[p_base + row * nx + inner] *
                        state.p[p_base + inner * nx + col];
                }
            }
        }
        for (int row = 0; row < nx; ++row) {
            for (int col = 0; col < nx; ++col) {
                float value = 0.0F;
                for (int inner = 0; inner < nx; ++inner) {
                    value += fp[p_base + row * nx + inner] *
                             ff[p_base + col * nx + inner];
                }
                const float noise = options.process_noise_scalar
                    ? (row == col ? options.q_scalar : 0.0F)
                    : qf[p_base + row * nx + col];
                next_p[p_base + row * nx + col] = af[point] * value + noise;
            }
        }
    }
    state.x = std::move(next_x);
    state.p = std::move(next_p);
}

template <typename T>
void kalman_update_typed_cpu(
    KalmanHostState& state,
    const std::vector<T>& h,
    const std::vector<T>& r,
    const std::vector<T>& z,
    const KalmanUpdateOptions& options)
{
    static_assert(detail::is_simple_signal_input_v<T>, "unsupported Kalman business dtype");
    validate_host_state(state);
    if (!options.measurement_present) return;
    const KalmanLayout& layout = state.layout;
    if (h.size() != layout.h_count || z.size() != layout.z_count ||
        (!options.measurement_noise_scalar && r.size() != layout.r_count)) {
        throw std::invalid_argument("kalman_update_typed_cpu shape mismatch");
    }

    const std::vector<float> hf = load_vector(h);
    const std::vector<float> rf = options.measurement_noise_scalar ? std::vector<float>{} : load_vector(r);
    const std::vector<float> zf = load_vector(z);
    const int nx = layout.dim_x;
    const int nz = layout.dim_z;

    for (int point = 0; point < layout.points; ++point) {
        const std::size_t x_base = static_cast<std::size_t>(point) * nx;
        const std::size_t p_base = x_base * nx;
        const std::size_t h_base = static_cast<std::size_t>(point) * nz * nx;
        const std::size_t r_base = static_cast<std::size_t>(point) * nz * nz;
        const std::size_t z_base = static_cast<std::size_t>(point) * nz;
        std::vector<float> innovation(nz, 0.0F);
        std::vector<float> pht(static_cast<std::size_t>(nx) * nz, 0.0F);
        std::vector<float> s(static_cast<std::size_t>(nz) * nz, 0.0F);
        std::vector<float> gain(static_cast<std::size_t>(nx) * nz, 0.0F);
        std::vector<float> i_kh(static_cast<std::size_t>(nx) * nx, 0.0F);
        std::vector<float> left(static_cast<std::size_t>(nx) * nx, 0.0F);
        std::vector<float> next_p(static_cast<std::size_t>(nx) * nx, 0.0F);

        for (int row = 0; row < nz; ++row) {
            innovation[row] = zf[z_base + row];
            for (int inner = 0; inner < nx; ++inner) {
                innovation[row] -=
                    hf[h_base + row * nx + inner] * state.x[x_base + inner];
            }
        }
        for (int row = 0; row < nx; ++row) {
            for (int col = 0; col < nz; ++col) {
                for (int inner = 0; inner < nx; ++inner) {
                    pht[row * nz + col] += state.p[p_base + row * nx + inner] *
                        hf[h_base + col * nx + inner];
                }
            }
        }
        for (int row = 0; row < nz; ++row) {
            for (int col = 0; col < nz; ++col) {
                for (int inner = 0; inner < nx; ++inner) {
                    s[row * nz + col] += hf[h_base + row * nx + inner] *
                        pht[inner * nz + col];
                }
                s[row * nz + col] += options.measurement_noise_scalar
                    ? (row == col ? options.r_scalar : 0.0F)
                    : rf[r_base + row * nz + col];
            }
        }
        inverse_fixed_cusignal(s, nz);
        for (int row = 0; row < nx; ++row) {
            for (int col = 0; col < nz; ++col) {
                for (int inner = 0; inner < nz; ++inner) {
                    // 固定raw kernel按inverse[col,inner]读取；对理想对称S等价于常规乘法。
                    gain[row * nz + col] +=
                        pht[row * nz + inner] * s[col * nz + inner];
                }
            }
        }
        for (int row = 0; row < nx; ++row) {
            for (int inner = 0; inner < nz; ++inner) {
                state.x[x_base + row] += gain[row * nz + inner] * innovation[inner];
            }
        }
        for (int row = 0; row < nx; ++row) {
            for (int col = 0; col < nx; ++col) {
                float value = row == col ? 1.0F : 0.0F;
                for (int inner = 0; inner < nz; ++inner) {
                    value -= gain[row * nz + inner] * hf[h_base + inner * nx + col];
                }
                i_kh[row * nx + col] = value;
            }
        }
        for (int row = 0; row < nx; ++row) {
            for (int col = 0; col < nx; ++col) {
                for (int inner = 0; inner < nx; ++inner) {
                    left[row * nx + col] +=
                        i_kh[row * nx + inner] * state.p[p_base + inner * nx + col];
                }
            }
        }
        for (int row = 0; row < nx; ++row) {
            for (int col = 0; col < nx; ++col) {
                for (int inner = 0; inner < nx; ++inner) {
                    next_p[row * nx + col] +=
                        left[row * nx + inner] * i_kh[col * nx + inner];
                }
                for (int left_z = 0; left_z < nz; ++left_z) {
                    for (int right_z = 0; right_z < nz; ++right_z) {
                        const float noise = options.measurement_noise_scalar
                            ? (left_z == right_z ? options.r_scalar : 0.0F)
                            : rf[r_base + left_z * nz + right_z];
                        next_p[row * nx + col] += gain[row * nz + left_z] *
                            noise * gain[col * nz + right_z];
                    }
                }
            }
        }
        const std::size_t matrix_size = static_cast<std::size_t>(nx) * nx;
        for (std::size_t index = 0; index < matrix_size; ++index) {
            state.p[p_base + index] = next_p[index];
        }
    }
}

#define INSTANTIATE_KALMAN_CPU(T) \
    template KalmanHostState make_kalman_host_state(const std::vector<T>&, const std::vector<T>&, const KalmanLayout&); \
    template void kalman_predict_typed_cpu(KalmanHostState&, const std::vector<T>&, const std::vector<T>&, const std::vector<T>&, const KalmanPredictOptions&); \
    template void kalman_update_typed_cpu(KalmanHostState&, const std::vector<T>&, const std::vector<T>&, const std::vector<T>&, const KalmanUpdateOptions&)

INSTANTIATE_KALMAN_CPU(float);
INSTANTIATE_KALMAN_CPU(__half);
INSTANTIATE_KALMAN_CPU(std::int32_t);
INSTANTIATE_KALMAN_CPU(std::int16_t);
INSTANTIATE_KALMAN_CPU(std::int8_t);

#undef INSTANTIATE_KALMAN_CPU

}  // namespace cusignal
