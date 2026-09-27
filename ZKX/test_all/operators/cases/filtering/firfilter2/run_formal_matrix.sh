#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 9 ]] || exit 2
b="$1";d="$2";c="$3";t="$4";list="$5";root="$6";sha="$7";dirty="$8";base="$9";i=0
while IFS= read -r scale || [[ -n "$scale" ]]; do
 [[ -z "$scale" || "$scale" == \#* ]] && continue
 time="$(date -u +%Y%m%dT%H%M%SZ)";random="$(printf '%08x' $((16#$base+i)))"
 echo "[remaining_operator_case-b04-formal] case=$i scale=$scale status=starting"
 python3 "$d" --binary "$b" --config "$c" --thresholds "$t" --results-root "$root" --scale "$scale" --run-type formal --git-commit "$sha" --git-dirty "$dirty" --test-time-utc "$time" --random "$random"
 echo "[remaining_operator_case-b04-formal] case=$i scale=$scale status=completed";i=$((i+1))
done < "$list"
[[ "$i" -eq 15 ]] || exit 3
echo "[remaining_operator_case-b04-formal] cases=15 status=completed"
