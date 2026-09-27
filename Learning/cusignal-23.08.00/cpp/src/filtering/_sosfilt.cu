// Copyright (c) 2019-2020, NVIDIA CORPORATION.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

///////////////////////////////////////////////////////////////////////////////
//                                SOSFILT                                    //
///////////////////////////////////////////////////////////////////////////////

constexpr int sos_width = 6;

template<typename T>
// <学习注释：device 模板为一条独立信号执行多节 DF-II-T 流水线，T 为 float 或 double。>
__device__ void _cupy_sosfilt( const int n_signals,
                               const int n_samples,
                               const int n_sections,
                               const int zi_width,
                               const T *__restrict__ sos,
                               const T *__restrict__ zi,
                               T *__restrict__ x_in,
                               T *s_buffer ) {

    // <学习注释：动态共享内存前 n_sections 个元素存节间输出，后面存每节六个系数。>
    T *s_out { s_buffer };
    T *s_sos { reinterpret_cast<T *>( &s_out[n_sections] ) };

    // <学习注释：线程索引 tx 对应二阶节索引，block 索引 bx 对应独立信号索引。>
    const int tx { static_cast<int>( threadIdx.x ) };
    const int bx { static_cast<int>( blockIdx.x ) };

    // Reset shared memory
    s_out[tx] = 0;

    // Load SOS
    // b is in s_sos[tx * sos_width + [0-2]]
    // a is in s_sos[tx * sos_width + [3-5]]
    // <学习注释：每个线程从全局 SOS 矩阵复制自己一节的六个系数到共享内存。>
#pragma unroll sos_width
    for ( int i = 0; i < sos_width; i++ ) {
        s_sos[tx * sos_width + i] = sos[tx * sos_width + i];
    }

    // __syncthreads( );

    // <学习注释：从全局 zi 读取当前信号、当前节的两个 DF-II-T 初态到寄存器。>
    T zi0 = zi[bx * n_sections * zi_width + tx * zi_width + 0];
    T zi1 = zi[bx * n_sections * zi_width + tx * zi_width + 1];

    // <学习注释：前 n_sections-1 次用于填满级联流水线，剩余次数产生可写回的输出。>
    const int load_size { n_sections - 1 };
    const int unload_size { n_samples - load_size };

    T temp {};
    T x_n {};

    if ( bx < n_signals ) {
        // Loading phase
        for ( int n = 0; n < load_size; n++ ) {
            __syncthreads( );
            // <学习注释：第 0 节读取原输入，其余节读取前一节放在共享内存中的输出。>
            if ( tx == 0 ) {
                x_n = x_in[bx * n_samples + n];
            } else {
                x_n = s_out[tx - 1];
            }

            // Use direct II transposed structure
            // <学习注释：temp=b0*x+zi0 是本节当前输出。>
            temp = s_sos[tx * sos_width + 0] * x_n + zi0;
            // <学习注释：zi0=b1*x-a1*temp+zi1 更新第一个状态。>
            zi0  = s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + zi1;
            // <学习注释：zi1=b2*x-a2*temp 更新第二个状态。>
            zi1  = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;

            s_out[tx] = temp;
        }

        // Processing phase
        for ( int n = load_size; n < n_samples; n++ ) {
            __syncthreads( );
            if ( tx == 0 ) {
                x_n = x_in[bx * n_samples + n];
            } else {
                x_n = s_out[tx - 1];
            }

            // Use direct II transposed structure
            // <学习注释：稳定处理阶段重复同一 DF-II-T 三式递推。>
            temp = s_sos[tx * sos_width + 0] * x_n + zi0;
            zi0  = s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + zi1;
            zi1  = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;

            // <学习注释：非末节继续写共享内存；末节把成熟流水输出写回 x_in。>
            if ( tx < load_size ) {
                s_out[tx] = temp;
            } else {
                x_in[bx * n_samples + ( n - load_size )] = temp;
            }
        }

        // Unloading phase
        // <学习注释：输入读完后继续推进流水线，排出仍停留在后续节中的最后若干样本。>
        for ( int n = 0; n < n_sections; n++ ) {
            __syncthreads( );
            // retire threads that are less than n
            if ( tx > n ) {
                x_n = s_out[tx - 1];

                // Use direct II transposed structure
                // <学习注释：排空阶段仍使用完全相同的 DF-II-T 递推公式。>
                temp = s_sos[tx * sos_width + 0] * x_n + zi0;
                zi0  = s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + zi1;
                zi1  = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;

                if ( tx < load_size ) {
                    s_out[tx] = temp;
                } else {
                    x_in[bx * n_samples + ( n + unload_size )] = temp;
                }
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__( 1024 ) _cupy_sosfilt_float32( const int n_signals,
                                                                            const int n_samples,
                                                                            const int n_sections,
                                                                            const int zi_width,
                                                                            const float *__restrict__ sos,
                                                                            const float *__restrict__ zi,
                                                                            float *__restrict__ x_in ) {

    // <学习注释：float32 入口声明动态共享内存并实例化 T=float 的 device 模板。>
    extern __shared__ float s_buffer_f[];

    _cupy_sosfilt<float>( n_signals, n_samples, n_sections, zi_width, sos, zi, x_in, s_buffer_f );
}

extern "C" __global__ void __launch_bounds__( 1024 ) _cupy_sosfilt_float64( const int n_signals,
                                                                            const int n_samples,
                                                                            const int n_sections,
                                                                            const int zi_width,
                                                                            const double *__restrict__ sos,
                                                                            const double *__restrict__ zi,
                                                                            double *__restrict__ x_in ) {

    // <学习注释：float64 入口声明 double 共享内存并实例化 T=double 的同一算法。>
    extern __shared__ double s_buffer_d[];

    _cupy_sosfilt<double>( n_signals, n_samples, n_sections, zi_width, sos, zi, x_in, s_buffer_d );
}
