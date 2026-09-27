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

# <学习注释：把 bsplines.py 中的 quadratic 绑定到 cusignal.bsplines 命名空间，同时该原始导入语句也导出另外两个 B 样条符号。>
# <学习注释：把实现模块中的 cubic 等符号重新导出到 cusignal.bsplines 包命名空间。>
from cusignal.bsplines.bsplines import cubic, gauss_spline, quadratic
