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

from cusignal.windows.windows import (
    barthann,
    bartlett,
    blackman,
    blackmanharris,
    bohman,
    boxcar,
    chebwin,
    cosine,
    exponential,
    flattop,
    gaussian,
    general_cosine,
    general_gaussian,
    general_hamming,
    get_window,
    hamming,
    hann,
    # <学习注释：从 windows.py 导出 kaiser，使其可通过 cusignal.windows.kaiser 访问。>
    kaiser,
    nuttall,
    # <学习注释：把 windows.py 中的 parzen 导入包命名空间，使用户可通过 cusignal.windows.parzen 访问。>
    parzen,
    # <学习注释：把 windows.py 中定义的 taylor 暴露为 cusignal.windows.taylor。>
    taylor,
    triang,
    tukey,
)
