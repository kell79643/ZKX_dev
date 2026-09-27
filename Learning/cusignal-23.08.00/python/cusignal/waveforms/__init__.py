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

from cusignal.waveforms.waveforms import (
    # <学习注释：把 waveforms.py 中定义的 chirp 导入 cusignal.waveforms 命名空间。>
    chirp,
    gausspulse,
    sawtooth,
    # <学习注释：把 waveforms.py 中定义的 square 导入当前包，使用户能够通过 cusignal.waveforms.square 访问它。>
    square,
    # <学习注释：把 waveforms.py 中定义的 unit_impulse 导入 cusignal.waveforms 命名空间。>
    unit_impulse,
)
