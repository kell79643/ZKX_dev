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

from cusignal.filtering.filtering import (
    channelize_poly,
    detrend,
    filtfilt,
    # <学习注释：把 filtering.py 中的 firfilter 导入 filtering 子包命名空间，因而可用 cusignal.filtering.firfilter 访问。>
    firfilter,
    # <学习注释：把 filtering.py 中的 firfilter2 导出到 cusignal.filtering 命名空间。>
    firfilter2,
    firfilter_zi,
    freq_shift,
    # <学习注释：把 filtering.py 中的 hilbert 暴露为 cusignal.filtering.hilbert。>
    hilbert,
    # <学习注释：把 filtering.py 中的 hilbert2 导出为 cusignal.filtering.hilbert2。>
    hilbert2,
    lfilter,
    lfilter_zi,
    # <学习注释：把 filtering.py 中的 sosfilt 暴露为 cusignal.filtering.sosfilt。>
    sosfilt,
    wiener,
)
from cusignal.filtering.resample import decimate, resample, resample_poly, upfirdn
