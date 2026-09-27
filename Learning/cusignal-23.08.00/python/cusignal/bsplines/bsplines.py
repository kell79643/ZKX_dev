# Copyright (c) 2019-2020, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# <学习注释：导入 CuPy 并使用别名 cp；quadratic 通过它把输入转成 GPU 数组并创建逐元素 kernel。>
import cupy as cp

_gauss_spline_kernel = cp.ElementwiseKernel(
    "T x, int32 n",
    "T output",
    """
    output = 1 / sqrt( 2.0 * M_PI * signsq ) * exp( -( x * x ) * r_signsq );
    """,
    "_gauss_spline_kernel",
    options=("-std=c++11",),
    loop_prep="const double signsq { ( n + 1 ) / 12.0 }; \
               const double r_signsq { 0.5 / signsq };",
)


def gauss_spline(x, n):
    """Gaussian approximation to B-spline basis function of order n.

    Parameters
    ----------
    n : int
        The order of the spline. Must be nonnegative, i.e. n >= 0

    References
    ----------
    .. [1] Bouma H., Vilanova A., Bescos J.O., ter Haar Romeny B.M., Gerritsen
       F.A. (2007) Fast and Accurate Gaussian Derivatives Based on B-Splines.
       In: Sgallari F., Murli A., Paragios N. (eds) Scale Space and Variational
       Methods in Computer Vision. SSVM 2007. Lecture Notes in Computer
       Science, vol 4485. Springer, Berlin, Heidelberg
    """
    x = cp.asarray(x)

    return _gauss_spline_kernel(x, n)


# <学习注释：创建并保存一个 CuPy 逐元素 kernel；Python 变量 _cubic_kernel 后续可像函数一样调用。>
_cubic_kernel = cp.ElementwiseKernel(
# <学习注释：声明一个名为 x、类型由模板占位符 T 推导的输入参数。>
    "T x",
# <学习注释：声明一个名为 res、类型同为 T 的输出参数，每个输入元素对应一个输出元素。>
    "T res",
# <学习注释：开始传给 ElementwiseKernel 的多行 C/C++ 风格 operation 字符串。>
    """
// <学习注释：在当前 GPU 元素线程中计算 x 的绝对值，并用 const T 局部变量 ax 保存。>
    const T ax { abs( x ) };

// <学习注释：若绝对值 ax 小于 1，则进入三次 B 样条的中心区间分支。>
    if( ax < 1 ) {
// <学习注释：计算中心段公式 2/3-(1/2)ax^2(2-ax)，并写入当前元素的输出 res。>
        res =  2.0 / 3 - 1.0 / 2  * ax * ax * ( 2.0 - ax );
// <学习注释：若中心分支不成立且 ax 小于 2，则进入支撑边缘区间；前半个条件显式表达 ax 不小于 1。>
    } else if( !( ax < 1 ) && ( ax < 2 ) ) {
// <学习注释：计算外侧段公式 (2-ax)^3/6，并写入当前元素的输出 res。>
        res = 1.0 / 6 * ( 2.0 - ax ) *  ( 2.0 - ax ) * ( 2.0 - ax );
// <学习注释：当前两个区间条件均不成立时进入兜底分支。>
    } else {
// <学习注释：把支撑区间之外以及未被前两分支接纳的输入映射为 0。>
        res = 0.0;
// <学习注释：结束 if/else 条件结构。>
    }
# <学习注释：结束传给 ElementwiseKernel 的多行 operation 字符串，末尾逗号分隔下一个实参。>
    """,
# <学习注释：把生成的 CUDA kernel 命名为 _cubic_kernel，供 CuPy 编译缓存和诊断使用。>
    "_cubic_kernel",
# <学习注释：用单元素元组传入编译选项 -std=c++11；尾随逗号使括号内容保持元组类型。>
    options=("-std=c++11",),
# <学习注释：结束 cp.ElementwiseKernel 调用，并把返回的 kernel 对象赋给 _cubic_kernel。>
)


# <学习注释：定义公开 Python 函数 cubic，只接收一个位置或位置数组 x。>
def cubic(x):
# <学习注释：下面的三引号 docstring 第一行说明函数计算三次 B 样条。>
# <学习注释：docstring 第二个非空行说明 cubic 是一般 bspline 在次数参数为 3 时的特例。>
# <学习注释：末尾三引号结束 docstring；不在字符串内部插注释，以免改变 cubic.__doc__。>
    """A cubic B-spline.

    This is a special case of `bspline`, and equivalent to ``bspline(x, 3)``.
    """
# <学习注释：用 cp.asarray 把 x 转为 CuPy 数组；若 x 已是合适的 CuPy 数组，通常可避免复制。>
    x = cp.asarray(x)

# <学习注释：调用逐元素 GPU kernel 并直接返回与输入逐元素对应的三次 B 样条函数值。>
    return _cubic_kernel(x)


# <学习注释：创建并保存 quadratic 专用的 CuPy 逐元素 kernel；调用时每个输出元素只依赖对应输入元素。>
_quadratic_kernel = cp.ElementwiseKernel(
# <学习注释：输入参数声明中 T 是由 CuPy 推导的占位 dtype，x 是当前线程处理的单个输入元素。>
    "T x",
# <学习注释：输出参数声明复用同一个 T，res 是当前输入元素对应的输出值。>
    "T res",
# <学习注释：三引号开始一个作为字符串传给 CuPy 的 C/C++ kernel 操作代码块。>
    """
// <学习注释：计算 x 的绝对值并保存为只读局部量 ax，从而利用二次 B 样条的偶对称性。>
    const T ax { abs( x ) };

// <学习注释：中央分段覆盖 |x|<0.5；花括号开始该条件成立时执行的代码块。>
    if( ax < 0.5 ) {
// <学习注释：计算中心段公式 beta^2(x)=3/4-|x|^2。>
        res = 0.75 - ax * ax;
// <学习注释：外侧分段要求前一条件不成立且 |x|<1.5，因此实际区间是 0.5<=|x|<1.5。>
    } else if( !( ax < 0.5 ) && ( ax < 1.5 ) ) {
// <学习注释：计算外侧段公式 beta^2(x)=0.5(|x|-1.5)^2。>
        res = ( ( ax - 1.5 ) * ( ax - 1.5 ) ) * 0.5 ;
// <学习注释：else 覆盖 |x|>=1.5，即紧支撑之外及支撑端点。>
    } else {
// <学习注释：在支撑外把基函数值设为零。>
        res = 0.0;
// <学习注释：结束 if/else if/else 分支。>
    }
// <学习注释：下一行结束传给 ElementwiseKernel 的操作代码字符串；逗号表示后面还有位置参数。>
    """,
# <学习注释：为生成的 kernel 指定内部名称 _quadratic_kernel。>
    "_quadratic_kernel",
# <学习注释：以单元素元组传入 C++11 编译选项；末尾逗号是单元素元组语法所必需的。>
    options=("-std=c++11",),
# <学习注释：结束 ElementwiseKernel 构造调用，并把得到的可调用 kernel 赋给左侧变量。>
)


# <学习注释：定义公开 Python 包装函数 quadratic，唯一形参 x 表示待求值的坐标。>
def quadratic(x):
# <学习注释：下一行开始函数 docstring；两句原文说明它是二次 B 样条并在数学上等价于 bspline(x, 2)。>
# <学习注释：docstring 结束后，运行时可由 help(quadratic) 或 quadratic.__doc__ 读取。>
    """A quadratic B-spline.

    This is a special case of `bspline`, and equivalent to ``bspline(x, 2)``.
    """
# <学习注释：把标量、列表、NumPy/CuPy 兼容对象统一转换为 CuPy 数组并重新绑定到局部变量 x。>
    x = cp.asarray(x)

# <学习注释：调用逐元素 kernel 计算数组中每个坐标的二次 B 样条值，并把结果返回给调用者。>
    return _quadratic_kernel(x)
