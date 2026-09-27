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

# <学习注释：该导入语句把实现文件中的 firwin2 暴露为 cusignal.filter_design.firwin2。>
from cusignal.filter_design.fir_filter_design import (
    cmplx_sort,
    # <学习注释：把 firwin 导出到 cusignal.filter_design 命名空间，使调用者无需直接导入实现文件。>
    firwin,
    # <学习注释：这一项选择性导入 firwin2，逗号表示它是多行导入列表中的一个元素。>
    firwin2,
    kaiser_atten,
    kaiser_beta,
)
