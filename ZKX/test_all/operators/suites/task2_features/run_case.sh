#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 8 || $# -eq 9 ]] || {
  echo "usage: run_case.sh OPERATOR BINARY SCALE RUN_TYPE GIT_COMMIT DIRTY_STATE RESULTS_ROOT BACKEND [CONFIG]" >&2
  exit 2
}
operator=$1
binary=$2
scale=$3
run_type=$4
git_commit=$5
dirty_state=$6
results_root=$7
backend=$8
case "$operator" in fm_demod|correlate|spectrogram) ;; *) exit 2 ;; esac
case "$run_type" in smoke|formal) ;; *) exit 2 ;; esac
case "$dirty_state" in true|false) ;; *) exit 2 ;; esac
case "$backend" in fft_thrust|dlfft) ;; *) exit 2 ;; esac
support_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$support_dir/../../../../" && pwd)
config=${9:-"$repo_root/test_all/config/operators/suites/task2_features/operator_scales.json"}
test_time=$(date -u +%Y%m%dT%H%M%SZ)
random=$(printf '%08x' $((16#$(printf '%s' "$operator:$scale" | sha256sum | cut -c1-8) ^ 16#${git_commit:0:8})))
exec python3 "$support_dir/run_operator.py" \
  --binary "$binary" --operator "$operator" --backend "$backend" \
  --config "$config" \
  --thresholds "$repo_root/test_all/config/operators/suites/task2_features/accuracy_thresholds.json" \
  --results-root "$results_root" --scale "$scale" --run-type "$run_type" \
  --git-commit "$git_commit" --git-dirty "$dirty_state" \
  --test-time-utc "$test_time" --random "$random"
