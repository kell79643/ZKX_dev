#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 7 || $# -eq 8 ]] || { echo "usage: run_case.sh OP BINARY SCALE RUN_TYPE GIT_COMMIT DIRTY_STATE RESULTS_ROOT [CONFIG]" >&2; exit 2; }
op=$1; binary=$2; scale=$3; run_type=$4; commit=$5; dirty=$6; results=$7
case "$op" in hamming|firwin|firfilter) ;; *) exit 2;; esac
case "$dirty" in true|false) ;; *) exit 2;; esac
case "$run_type" in smoke|formal) ;; *) exit 2;; esac
support_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$support_dir/../../../../" && pwd)
config=${8:-"$repo_root/test_all/config/operators/suites/task2_filter_chain/operator_scales.json"}
test_time=$(date -u +%Y%m%dT%H%M%SZ)
random=$(printf '%08x' $((16#$(printf '%s' "$op:$scale" | sha256sum | cut -c1-8) ^ 16#${commit:0:8})))
exec python3 "$support_dir/run_operator.py" --binary "$binary" --operator "$op" \
  --config "$config" \
  --thresholds "$repo_root/test_all/config/operators/suites/task2_filter_chain/accuracy_thresholds.json" \
  --results-root "$results" --scale "$scale" --run-type "$run_type" \
  --git-commit "$commit" --git-dirty "$dirty" --test-time-utc "$test_time" --random "$random"
