#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 9 ]]||exit 2;b="$1";d="$2";c="$3";t="$4";s="$5";r="$6";g="$7";y="$8";x="$9";i=0
while IFS= read -r q||[[ -n "$q" ]];do [[ -z "$q"||"$q" == \#* ]]&&continue;u="$(date -u +%Y%m%dT%H%M%SZ)";v="$(printf '%08x' $((16#$x+i)))";echo "[remaining_operator_case-b06-formal] case=$i scale=$q status=starting";python3 "$d" --binary "$b" --config "$c" --thresholds "$t" --results-root "$r" --scale "$q" --run-type formal --git-commit "$g" --git-dirty "$y" --test-time-utc "$u" --random "$v";echo "[remaining_operator_case-b06-formal] case=$i scale=$q status=completed";i=$((i+1));done<"$s";[[ $i -eq 15 ]]||exit 3;echo "[remaining_operator_case-b06-formal] cases=15 status=completed"
