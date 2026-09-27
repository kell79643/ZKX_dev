# 开发环境

## 每次会话必读

- 只要任务涉及 ZQ500 的同步、构建、运行、测试或环境判断，必须先读
  `代码规范.txt`，再读活跃仓库中的
  `ZKX/AGENTS.md` 和 `ZKX/docs/development/ZQ500_RUNTIME.md` 的“标准操作流程”。
- `代码规范.txt` 是服务器操作基准；`ZQ500_RUNTIME.md` 是同步进源码
  仓库的完整技术流程；本文件只保留 AI 必须每次看到的硬约束。三者冲突时先停止并按
  前两份文档核对，不自行发明新的 sudo、Docker 或 shell 包装方式。

## 远程执行

- Codex 在 Windows 本地负责阅读、编辑和适配代码。开发时使用聚焦的本地 Git
  分支和提交管理改动；项目代码验证必须放到 ZKX 远程服务器执行。
- 本项目必须运行在远程主机 `usr_02@10.110.12.10`。
- 远程项目目录是 `/home/usr_02/ZKX_dev`。
- ZQ500 SDK 和硬件运行环境位于该主机上的 Docker 容器 `gpu_02` 内。
- 一切适配、构建、运行、测试和文档判断都以中科芯 ZQ500 服务器
  `usr_02@10.110.12.10` 为主。本地 Windows、个人机器路径、通用
  NVIDIA CUDA 示例或更高版本工具链只能作为参考，不能覆盖 `usr_02`
  上的实际环境。
- CMake 版本必须服从 `gpu_02` 实际构建环境。2026-07-12 在初始化 SDK 的容器内
  实测为 `3.22.1`；此前宿主机曾观察到 `3.16.3`，不能再把宿主机版本写成当前
  容器版本。项目仍以 `cmake_minimum_required(VERSION 3.16)` 保持向下兼容，除非
  官方构建要求明确需要新版特性，不因当前 3.22.1 就随意提高最低版本。
- 活跃代码仓库通常是嵌套的 `ZKX/` 目录。优先从 `ZKX_dev/ZKX` 或该共享工作区的本地等价路径启动 Codex。
- 两名开发者使用同一份共享 `ZKX_dev/` 文件树。不要把个人电脑绝对路径写入仓库文档、脚本或代码注释；使用相对 `ZKX_dev/` 的路径。
- 如果在 `ZKX/` 内工作，遵守 `ZKX/AGENTS.md`；它是更具体的项目指令文件。
- 不要在本地运行 Python、MATLAB、CUDA 或硬件测试。
- 先进入活跃仓库，再使用 `.\remote_scripts\remote-run.ps1` 执行远程 smoke test
  和程序运行。传入精确远程命令，例如：
  `.\remote_scripts\remote-run.ps1 "<python> path/to/script.py"`。
- 远程执行脚本优先使用 `rsync`；如果本地没有 `rsync`，会回退到通过 SSH 传输 `tar`。
- 远程硬件、CUDA 环境、数据集和运行依赖在本地不可用。
- 在 `gpu_02` 内使用 `dlsmi` 查看 GPU 状态；它是 ZQ500 平台上的
  `nvidia-smi` 等价工具。
- 平台每天 `01:00-08:00` 可能维护，不适合安排长时间实验。

## 标准远程工作流

必须严格分成“本地核对 → 同步宿主机 → 精确复制 → 交互进入容器 → 容器内执行”五个
阶段，不得跳步或把容器命令拼进一次非交互 `docker exec`。

在 `ZKX_dev/` 的本地 PowerShell 中执行：

```powershell
Set-Location .\ZKX
git rev-parse HEAD
git status --short
.\remote_scripts\remote-run.ps1 "pwd"
.\remote_scripts\remote-run.ps1 --no-sync "sudo docker cp /home/usr_02/ZKX_dev/. gpu_02:/tmp/ZKX_dev"
```

- 只有同步返回 0，且输出同时包含 `/home/usr_02/ZKX_dev` 和 `Sync done`，才允许执行
  `docker cp`。
- sudoers 只保证上面的精确 `docker cp` 和下面的精确交互式 `docker exec` 入口可用。
  不得尝试 `sudo docker exec gpu_02 <命令>`、`sudo docker ps`、`sudo -n`、
  `bash -c` 或 `bash -lc`；这些变体会越过既定入口并触发 sudo 密码验证。
- AI 不得等待、索取、代输或记录 sudo 密码。出现密码提示说明命令形态错误，应立即终止，
  回到本节的精确命令。

人工操作在 PowerShell 中继续执行，并在打开的容器 shell 内逐条运行后续命令：

```powershell
.\remote_scripts\remote-run.ps1 --no-sync "sudo docker exec -it gpu_02 bash"
```

```sh
source /zq500/sdk/env.sh
cd /tmp/ZKX_dev
pwd
dlsmi
cmake --version
```

AI/自动化不能维持上述人工终端时，必须使用仓库现有的持久交互辅助脚本；该脚本内部仍只
启动精确的 `sudo docker exec -it gpu_02 bash`：

```powershell
.\remote_scripts\remote-run.ps1 --no-sync "python3 remote_scripts/gpu02_interactive.py start --timeout 30"
.\remote_scripts\remote-run.ps1 --no-sync "python3 remote_scripts/gpu02_interactive.py send 'source /zq500/sdk/env.sh' --timeout 30"
.\remote_scripts\remote-run.ps1 --no-sync "python3 remote_scripts/gpu02_interactive.py send 'cd /tmp/ZKX_dev && pwd && dlsmi && cmake --version' --timeout 60"
```

后续构建、运行和检查也通过 `gpu02_interactive.py send` 逐条发送。每条必须等待
`[gpu02-interactive] status=prompt`；`sudo_password_required`、`timeout`、非零退出或成功
标志缺失时立即停止。`source`、`cd`、CMake 和程序运行只能发生在这个容器会话内。

### rsync 23 与 root-owned 目录恢复

- `remote-run.ps1` 的标准同步是“本地打包 -> 上传 tar -> 由 `usr_02` 解压到
  staging -> rsync 镜像”。由 `usr_02` 解压产生的文件 owner 应为
  `usr_02:usr_02`。使用 sudo 从容器直接复制目录回宿主机，可能留下
  `root:root` 文件；两者不能混用为同一个可写源码树。
- 若同步出现 `rsync error ... code 23`、`Permission denied`、`chgrp failed` 或
  `cannot delete non-empty directory`，先停止同步，不得把部分传输结果当作成功，
  也不得继续执行 `docker cp`。使用 `--no-sync` 和 `stat -c %U:%G:%n` 比较
  `/tmp/zkx-sync-stage-user` 与 `/home/usr_02/ZKX_dev` 的 owner。
- `usr_02` 没有任意 `chown` sudo 权限；不要反复尝试 `sudo chown`。如果项目根目录
  由 `usr_02` 持有，但内部固定依赖树为 `root:root`，采用非破坏性恢复：把整个
  `/home/usr_02/ZKX_dev` 改名为带时间戳的备份，创建新的同名目录，再重新运行
  `remote-run.ps1 "pwd"`。旧目录未经用户明确批准不得删除。
- `cupy-13.6.0` 位于外层共享工作区，不在 `ZKX/` 同步源内，但 CMake 契约需要它。
  同步脚本必须排除并保留该目录。若重建宿主项目根目录，应由 `usr_02` 对旧 CuPy
  目录打 tar，再以 `tar --no-same-owner -xf ...` 解到新项目根目录；不要直接移动
  root-owned 子目录，也不要从同步镜像中删除它。
- 只有同步命令返回 0，同时输出远程 `pwd` 和 `Sync done`，并确认活动任务目录、
  泄漏门禁源码及 owner 正确，才能执行 `sudo docker cp`。

## 如何发现运行命令

- 先检查 `README.md`、任务专属文档、`remote_scripts/`、`run*.sh`、
  `test*.sh`、`CMakeLists.txt`，以及带 `if __name__ == "__main__"` 的
  Python 文件。
- 优先使用已有项目脚本，不要重新发明运行命令。
- 做 ZQ500 CMake 工作前，由用户进入 `gpu_02`，初始化 SDK，再查看
  `/zq500/sdk/samples/` 下的平台样例。
- ZQ500 样例通常使用 `project(... LANGUAGES CXX)`，从 `$DLICC_PATH`
  设置 C++ 编译器，将 `.cu` 文件标记为 `LANGUAGE CXX`，链接 `curt`，
  并用 `-x cuda` 和平台 GPU 架构参数编译 CUDA 源文件。
- 不确定入口时，优先通过不含 sudo、`-n` 或 `bash` 的 `--no-sync` 只读命令检查。

## 已知入口

- 任务入口必须先从 `ZKX/README.md`、任务专属 README、CMake 和现有运行脚本确认。
- 使用 Python 入口前，先确认远程 Python 命令及项目依赖；不要假设宿主机默认存在
  `python` 或已经安装项目依赖。

## 安全

- 未经明确要求，不要修改或同步 `data/`、`results/` 或大型生成产物。
- 不要同步 `.git/`、`.venv/`、`__pycache__/`、`build/`、`dist/` 或 `*.egg-info/`。
- 不要同步大型二进制输出，例如 `*.mat`、`*.npy`、`*.npz`、`*.png`、`*.jpg` 或 `*.jpeg`。
- 未经明确批准，不要在远程主机安装包。
- 不要把平台密码、VPN 密码、sudo 密码或其他凭据写入仓库文档、脚本、代码注释
  或命令输出。
- 不要杀掉远程主机上的无关进程。
- 优先跑 smoke test，再跑完整实验。

## AI 验证与重复运行交接

- 需要重复运行多轮的测试、实验或性能采样，必须在代码或现有运行脚本中预先实现结果
  落盘；不得把持续读取、筛选终端标准输出作为 AI 获取实验结果的工作方式。
- AI 只负责运行一轮代表性验证，确认入口、参数、结果文件写入和关键结果格式均正确。
  单轮验证通过后，AI 停止代跑其余重复轮次，由开发者按已确认的命令执行剩余次数。
- 后续轮次所需的关键信息必须由程序自动写入任务约定的文本、CSV、JSON 或其他可审阅
  文件。结果至少应包含运行轮次、关键参数、代码版本或提交 SHA、成功/失败状态以及任务
  所需核心指标；不能要求开发者或后续 AI 从完整控制台输出中人工摘录。
- AI 在交接时必须说明剩余运行命令、运行次数、结果文件路径和完成判据。除非开发者明确
  要求排查某一轮异常，否则 AI 不继续代替开发者执行或监看重复实验。

## 文档语言与编码

- 面向仓库使用者、评委和后续 AI 会话的 Markdown 文档默认使用中文写作。
- 只有代码标识符、API 名称、命令名、文件路径、状态标签，以及已经约定俗成的技术术语，才保留英文。
- 新增或修改文档时，应延续现有中文项目文档的表达风格，不要把中文文档改成英文说明。
- 中文 Markdown/text 文件按 UTF-8 读取和写入。Windows PowerShell 读取时优先使用：
  `Get-Content -Raw -Encoding UTF8 <path>`。
- 如果终端中中文显示为乱码，不要先假设文件已损坏；应按 UTF-8 重新读取。必要时先把控制台切到 UTF-8：
  `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; chcp 65001`。
- 除非用户明确要求，不要把项目文档转换为其他编码。

## Git 同步要求

- 每完成一个明确的小任务或任务切片，都要同步进本地 Git 仓库，形成可追踪提交。
- 提交前只暂存本次小任务相关文件，不要把无关工作区改动混入同一个提交。
- 每次执行 `git commit` 都必须显式提供能够区分版本的提交信息，例如
  `git commit -m "<类型>: <本次改动特点和影响范围>"`。提交信息必须简短说明本次改动
  的具体特点和影响范围；禁止使用空消息、默认消息、纯编号/日期、`update`、`修改` 等
  无法辨识内容，或任何乱码作为提交信息。
- 除非用户明确要求，不要执行 `git push`、`git pull` 或其他远端同步操作。

## 本地前置条件

- Windows OpenSSH 必须可用，命令名为 `ssh`。
- 推荐本地 PATH 上有 `rsync`，例如通过 MSYS2、Cygwin、WSL 或 cwRsync 提供。
- Windows 自带的 `tar.exe` 足以支持回退同步路径。
