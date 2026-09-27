#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 9 ]]; then
  echo "usage: $0 BINARY DRIVER CONFIG THRESHOLDS SCALE_LIST RESULTS_ROOT GIT_SHA GIT_DIRTY RANDOM_BASE_HEX" >&2
  exit 2
fi
binary="$1"; driver="$2"; config="$3"; thresholds="$4"; scale_list="$5"
results_root="$6"; git_sha="$7"; git_dirty="$8"; random_base="$9"
if [[ ! "$random_base" =~ ^[0-9a-f]{8}$ ]]; then
  echo "RANDOM_BASE_HEX must be eight lowercase hex characters" >&2
  exit 2
fi
case_index=0
while IFS= read -r scale || [[ -n "$scale" ]]; do
  [[ -z "$scale" || "$scale" == \#* ]] && continue
  test_time="$(date -u +%Y%m%dT%H%M%SZ)"
  random="$(printf '%08x' $((16#$random_base + case_index)))"
  echo "[remaining_operator_case-b04-formal] case=$case_index scale=$scale status=starting"
  python3 "$driver" --binary "$binary" --config "$config" --thresholds "$thresholds" \
    --results-root "$results_root" --scale "$scale" --run-type formal \
    --git-commit "$git_sha" --git-dirty "$git_dirty" \
    --test-time-utc "$test_time" --random "$random"
  echo "[remaining_operator_case-b04-formal] case=$case_index scale=$scale status=completed"
  case_index=$((case_index + 1))
done < "$scale_list"
if [[ "$case_index" -ne 15 ]]; then
  echo "[remaining_operator_case-b04-formal] expected=15 actual=$case_index status=failed" >&2
  exit 3
fi
echo "[remaining_operator_case-b04-formal] cases=15 status=completed"
