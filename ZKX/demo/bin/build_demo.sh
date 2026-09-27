#!/usr/bin/env bash
set -euo pipefail
repository_root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repository_root"
exec python3 demo/build/build_demo.py "$@"
