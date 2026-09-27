#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 9 ]] || exit 2
binary="$1";driver="$2";config="$3";thresholds="$4";scales="$5";results="$6";sha="$7";dirty="$8";base="$9";i=0
while IFS= read -r scale || [[ -n "$scale" ]];do [[ -z "$scale" || "$scale" == \#* ]]&&continue;t="$(date -u +%Y%m%dT%H%M%SZ)";r="$(printf '%08x' $((16#$base+i)))";echo "[remaining_operator_case-b04-formal] case=$i scale=$scale status=starting";python3 "$driver" --binary "$binary" --config "$config" --thresholds "$thresholds" --results-root "$results" --scale "$scale" --run-type formal --git-commit "$sha" --git-dirty "$dirty" --test-time-utc "$t" --random "$r";echo "[remaining_operator_case-b04-formal] case=$i scale=$scale status=completed";i=$((i+1));done < "$scales"
[[ $i -eq 15 ]]||exit 3;echo "[remaining_operator_case-b04-formal] cases=15 status=completed"
