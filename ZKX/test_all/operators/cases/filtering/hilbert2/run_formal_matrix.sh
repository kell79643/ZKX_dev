#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/../channelize_poly/run_formal_matrix.sh" "$@"
