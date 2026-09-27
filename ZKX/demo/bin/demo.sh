#!/usr/bin/env bash
set -euo pipefail
repository_root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repository_root"
export LD_LIBRARY_PATH="$repository_root/build/demo/fft_thrust/lib:$repository_root/build/demo/dlfft/lib:$repository_root/build/demo/not_applicable/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec python3 demo/app/unified_demo_runner.py "$@"
