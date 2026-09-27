#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 9 ]] || exit 2
binary="$1";driver="$2";config="$3";thresholds="$4";scales="$5";results="$6";git_commit="$7";git_dirty="$8";random_base="$9";index=0
while IFS= read -r scale || [[ -n "$scale" ]]; do
  [[ -z "$scale" || "$scale" == \#* ]] && continue
  test_time="$(date -u +%Y%m%dT%H%M%SZ)";random="$(printf '%08x' $((16#$random_base+index)))"
  echo "[remaining_operator_case-b05-formal] case=$index scale=$scale status=starting"
  python3 "$driver" --binary "$binary" --config "$config" --thresholds "$thresholds" --results-root "$results" --scale "$scale" --run-type formal --git-commit "$git_commit" --git-dirty "$git_dirty" --test-time-utc "$test_time" --random "$random"
  echo "[remaining_operator_case-b05-formal] case=$index scale=$scale status=completed";index=$((index+1))
done < "$scales"
[[ $index -eq 15 ]] || exit 3
echo "[remaining_operator_case-b05-formal] cases=15 status=completed"
