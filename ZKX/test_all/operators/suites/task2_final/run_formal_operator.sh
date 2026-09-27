#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 6 ]] || { echo "usage: run_formal_operator.sh OPERATOR BINARY GIT_COMMIT DIRTY_STATE RESULTS_ROOT CASE_DIR" >&2; exit 2; }
operator=$1; binary=$2; git_commit=$3; dirty_state=$4; results_root=$5; case_dir=$6
case "$operator" in cubic|argrelextrema|kalman_filter) ;; *) exit 2 ;; esac
driver_root="$results_root/formal_driver_logs/$operator"
mkdir -p "$driver_root"
count=0
for dtype in fp32 fp16 int32 int16 int8; do
  for magnitude in 10e2 10e3 10e4; do
    scale_id="${operator}_${dtype}_${magnitude}"
    log="$driver_root/${dtype}_${magnitude}.log"
    [[ ! -e "$log" ]] || { echo "refusing to overwrite $log" >&2; exit 2; }
    set +e; set -o pipefail
    bash "$case_dir/run_operator.sh" "$binary" "$scale_id" formal "$git_commit" "$dirty_state" "$results_root" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}; set -e
    printf '[stage07-b06-formal] operator=%s scale=%s rc=%s\n' "$operator" "$scale_id" "$rc"
    [[ "$rc" -eq 0 ]] || exit "$rc"
    count=$((count + 1))
  done
done
[[ "$count" -eq 15 ]]
printf '[stage07-b06-formal] operator=%s cases=15 status=completed\n' "$operator"
