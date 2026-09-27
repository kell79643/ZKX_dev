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

#include <nvfunctional>

///////////////////////////////////////////////////////////////////////////////
//                            FUNCTION POINTERS                              //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
// <学习注释：严格小于比较器；模板 T 让同一逻辑复用于四种预编译数值类型。>
__device__ __forceinline__ bool less( const T &a, const T &b ) {
    return ( a < b );
}

template<typename T>
// <学习注释：严格大于比较器，对应严格局部最大值判据。>
__device__ __forceinline__ bool greater( const T &a, const T &b ) {
    return ( a > b );
}

template<typename T>
// <学习注释：非严格小于等于比较器，可用于逐样本非严格局部最小判据。>
__device__ __forceinline__ bool less_equal( const T &a, const T &b ) {
    return ( a <= b );
}

template<typename T>
// <学习注释：非严格大于等于比较器，可让平台中的多个样本同时通过比较。>
__device__ __forceinline__ bool greater_equal( const T &a, const T &b ) {
    return ( a >= b );
}

template<typename T>
// <学习注释：相等谓词属于通用邻域筛选语义，本身不表示局部极值。>
__device__ __forceinline__ bool equal( const T &a, const T &b ) {
    return ( a == b );
}

template<typename T>
// <学习注释：不等谓词同样只是邻域模式，不应自动解释为极值。>
__device__ __forceinline__ bool not_equal( const T &a, const T &b ) {
    return ( a != b );
}

template<typename T>
// <学习注释：op_func 是指向“接收两个 T 常量引用并返回 bool”的 device 函数指针类型。>
using op_func = bool ( * )( const T &, const T & );

// <学习注释：四张函数指针表的元素顺序必须与 Python _modedict 的操作码 0 到 5 完全一致。>
__device__ op_func<int> const func_i[6]      = { less, greater, less_equal, greater_equal, equal, not_equal };
__device__ op_func<long int> const func_l[6] = { less, greater, less_equal, greater_equal, equal, not_equal };
__device__ op_func<float> const func_f[6]    = { less, greater, less_equal, greater_equal, equal, not_equal };
__device__ op_func<double> const func_d[6]   = { less, greater, less_equal, greater_equal, equal, not_equal };

///////////////////////////////////////////////////////////////////////////////
//                              HELPER FUNCTIONS                             //
///////////////////////////////////////////////////////////////////////////////

// <学习注释：修正正向邻居索引；clip=true 时复制末端，false 时减去一次轴长实现单周期回绕。>
__device__ __forceinline__ void clip_plus( const bool &clip, const int &n, int &plus ) {
    if ( clip ) {
        if ( plus >= n ) {
            plus = n - 1;
        }
    } else {
        if ( plus >= n ) {
            plus -= n;
        }
    }
}

// <学习注释：修正负向邻居索引；clip=true 时复制首端，false 时加上一次轴长。>
__device__ __forceinline__ void clip_minus( const bool &clip, const int &n, int &minus ) {
    if ( clip ) {
        if ( minus < 0 ) {
            minus = 0;
        }
    } else {
        if ( minus < 0 ) {
            minus += n;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
//                          BOOLRELEXTREMA 1D                                //
///////////////////////////////////////////////////////////////////////////////

template<typename T, class U>
// <学习注释：一维 device 核心让每个线程以 grid-stride loop 处理一个或多个中心样本。>
__device__ void _cupy_boolrelextrema_1D( const int  n,
                                         const int  order,
                                         const bool clip,
                                         const T *__restrict__ inp,
                                         bool *__restrict__ results,
                                         U func ) {

    // <学习注释：tx 是线程在整个一维 grid 中的全局起始索引。>
    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    // <学习注释：stride 是整个 grid 的线程总数，供线程跨步覆盖更长输入。>
    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

    // <学习注释：每个线程从 tx 开始，每次前进 stride，直到覆盖轴长 n。>
    for ( int tid = tx; tid < n; tid += stride ) {

        // <学习注释：把中心值读入寄存器 data，并把当前候选状态初始化为 true。>
        const T data { inp[tid] };
        bool    temp { true };

        // <学习注释：o 从 1 到 order，逐一访问中心两侧相同距离的邻居。>
        for ( int o = 1; o < ( order + 1 ); o++ ) {
            int plus { tid + o };
            int minus { tid - o };

            clip_plus( clip, n, plus );
            clip_minus( clip, n, minus );

            // <学习注释：把中心与正向邻居的比较结果逻辑与到 temp。>
            temp &= func( data, inp[plus] );
            // <学习注释：再累积负向邻居，最终 temp 仅在全部 2×order 次比较为真时保持 true。>
            temp &= func( data, inp[minus] );
        }
        // <学习注释：将该中心位置的最终候选状态写入输出布尔数组。>
        results[tid] = temp;
    }
}

// <学习注释：以下四个 extern "C" 一维入口把模板核心分别实例化为 int32、int64、float32、float64，并提供稳定 fatbin 符号名。>
extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_1D_int32( const int  n,
                                                                                   const int  order,
                                                                                   const bool clip,
                                                                                   const int  comp,
                                                                                   const int *__restrict__ inp,
                                                                                   bool *__restrict__ results ) {
    _cupy_boolrelextrema_1D<int, op_func<int>>( n, order, clip, inp, results, func_i[comp] );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_1D_int64( const int  n,
                                                                                   const int  order,
                                                                                   const bool clip,
                                                                                   const int  comp,
                                                                                   const long int *__restrict__ inp,
                                                                                   bool *__restrict__ results ) {
    _cupy_boolrelextrema_1D<long int, op_func<long int>>( n, order, clip, inp, results, func_l[comp] );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_1D_float32( const int  n,
                                                                                     const int  order,
                                                                                     const bool clip,
                                                                                     const int  comp,
                                                                                     const float *__restrict__ inp,
                                                                                     bool *__restrict__ results ) {
    _cupy_boolrelextrema_1D<float, op_func<float>>( n, order, clip, inp, results, func_f[comp] );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_1D_float64( const int  n,
                                                                                     const int  order,
                                                                                     const bool clip,
                                                                                     const int  comp,
                                                                                     const double *__restrict__ inp,
                                                                                     bool *__restrict__ results ) {
    _cupy_boolrelextrema_1D<double, op_func<double>>( n, order, clip, inp, results, func_d[comp] );
}

///////////////////////////////////////////////////////////////////////////////
//                          BOOLRELEXTREMA 2D                                //
///////////////////////////////////////////////////////////////////////////////

template<typename T, class U>
// <学习注释：二维 device 核心仍只沿 axis 指定的一个维度比较，不是二维全邻域极值检测。>
__device__ void _cupy_boolrelextrema_2D( const int  in_x,
                                         const int  in_y,
                                         const int  order,
                                         const bool clip,
                                         const int  axis,
                                         const T *__restrict__ inp,
                                         bool *__restrict__ results,
                                         U func ) {

    // <学习注释：ty 与 tx 分别计算列、行方向的全局线程坐标；命名与常见 x/y 直觉相反，后续线性地址给出真实含义。>
    const int ty { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int tx { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };

    // <学习注释：二维 grid 向上取整后会有越界线程，只有有效行列坐标才参与计算。>
    if ( ( tx < in_y ) && ( ty < in_x ) ) {
        // <学习注释：C 连续布局下线性地址等于 row×列数+column。>
        int tid { tx * in_x + ty };

        const T data { inp[tid] };
        bool    temp { true };

        for ( int o = 1; o < ( order + 1 ); o++ ) {

            int plus {};
            int minus {};

            // <学习注释：axis=0 时改变行坐标 tx，保持列坐标 ty 不变。>
            if ( axis == 0 ) {
                plus  = tx + o;
                minus = tx - o;

                clip_plus( clip, in_y, plus );
                clip_minus( clip, in_y, minus );

                plus  = plus * in_x + ty;
                minus = minus * in_x + ty;
            } else {
                // <学习注释：其他 axis 值走列方向分支，改变 ty 并保持 tx 不变。>
                plus  = ty + o;
                minus = ty - o;

                clip_plus( clip, in_x, plus );
                clip_minus( clip, in_x, minus );

                plus  = tx * in_x + plus;
                minus = tx * in_x + minus;
            }

            // <学习注释：与一维核心相同，二维核心也把每个正负邻居比较累积为全称逻辑与。>
            temp &= func( data, inp[plus] );
            temp &= func( data, inp[minus] );
        }
        results[tid] = temp;
    }
}

// <学习注释：以下四个二维入口同样只做类型实例化和函数表选择，真正的邻域循环位于 _cupy_boolrelextrema_2D。>
extern "C" __global__ void __launch_bounds__( 256 ) _cupy_boolrelextrema_2D_int32( const int  in_x,
                                                                                   const int  in_y,
                                                                                   const int  order,
                                                                                   const bool clip,
                                                                                   const int  comp,
                                                                                   const int  axis,
                                                                                   const int *__restrict__ inp,
                                                                                   bool *__restrict__ results ) {
    _cupy_boolrelextrema_2D<int, op_func<int>>( in_x, in_y, order, clip, axis, inp, results, func_i[comp] );
}

extern "C" __global__ void __launch_bounds__( 256 ) _cupy_boolrelextrema_2D_int64( const int  in_x,
                                                                                   const int  in_y,
                                                                                   const int  order,
                                                                                   const bool clip,
                                                                                   const int  comp,
                                                                                   const int  axis,
                                                                                   const long int *__restrict__ inp,
                                                                                   bool *__restrict__ results ) {
    _cupy_boolrelextrema_2D<long int, op_func<long int>>( in_x, in_y, order, clip, axis, inp, results, func_l[comp] );
}

extern "C" __global__ void __launch_bounds__( 256 ) _cupy_boolrelextrema_2D_float32( const int  in_x,
                                                                                     const int  in_y,
                                                                                     const int  order,
                                                                                     const bool clip,
                                                                                     const int  comp,
                                                                                     const int  axis,
                                                                                     const float *__restrict__ inp,
                                                                                     bool *__restrict__ results ) {
    _cupy_boolrelextrema_2D<float, op_func<float>>( in_x, in_y, order, clip, axis, inp, results, func_f[comp] );
}

extern "C" __global__ void __launch_bounds__( 256 ) _cupy_boolrelextrema_2D_float64( const int  in_x,
                                                                                     const int  in_y,
                                                                                     const int  order,
                                                                                     const bool clip,
                                                                                     const int  comp,
                                                                                     const int  axis,
                                                                                     const double *__restrict__ inp,
                                                                                     bool *__restrict__ results ) {
    _cupy_boolrelextrema_2D<double, op_func<double>>( in_x, in_y, order, clip, axis, inp, results, func_d[comp] );
}
