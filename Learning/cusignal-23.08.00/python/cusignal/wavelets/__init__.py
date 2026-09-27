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

# <学习注释：从实现模块导入 qmf，使用户可以通过 cusignal.wavelets.qmf 访问该函数。>
# <学习注释：从 wavelets.py 导出 ricker，使它可以通过 cusignal.wavelets.ricker 访问；同一原始语句还导出其他小波 API。>
# <学习注释：把 wavelets.py 中的 cwt 暴露为 cusignal.wavelets.cwt。>
from cusignal.wavelets.wavelets import cwt, morlet, morlet2, qmf, ricker
