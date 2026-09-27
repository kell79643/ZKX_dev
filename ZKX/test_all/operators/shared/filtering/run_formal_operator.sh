#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: run_formal_operator.sh OPERATOR BINARY GIT_COMMIT DIRTY_STATE RESULTS_ROOT CASE_DIR" >&2
  exit 2
fi

operator=$1
binary=$2
git_commit=$3
dirty_state=$4
results_root=$5
case_dir=$6

case "$operator" in
  hamming|firwin|firfilter) ;;
  *) echo "unsupported Batch04 operator: $operator" >&2; exit 2 ;;
esac
case "$dirty_state" in
  true|false) ;;
  *) echo "dirty state must be true or false" >&2; exit 2 ;;
esac

driver_root="$results_root/formal_driver_logs/$operator"
mkdir -p "$driver_root"
case_count=0

for dtype in fp32 fp16 int32 int16 int8; do
  for magnitude in 10e2 10e3 10e4; do
    scale_id="${operator}_${dtype}_${magnitude}"
    driver_log="$driver_root/${dtype}_${magnitude}.log"
    if [[ -e "$driver_log" ]]; then
      echo "refusing to overwrite driver log: $driver_log" >&2
      exit 2
    fi

    set +e
    set -o pipefail
    bash "$case_dir/run_operator.sh" \
      "$binary" "$scale_id" formal "$git_commit" "$dirty_state" "$results_root" \
      2>&1 | tee "$driver_log"
    rc=${PIPESTATUS[0]}
    set -e

    printf '[stage07-b04-formal] operator=%s scale=%s rc=%s\n' \
      "$operator" "$scale_id" "$rc"
    if [[ "$rc" -ne 0 ]]; then
      echo '[stage07-b04-formal] status=stopped_on_failure'
      exit "$rc"
    fi
    case_count=$((case_count + 1))
  done
done

test "$case_count" -eq 15
printf '[stage07-b04-formal] operator=%s cases=15 status=completed\n' "$operator"
