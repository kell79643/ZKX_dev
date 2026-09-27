#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 && $# -ne 9 ]]; then
    echo "usage: bash run_formal_matrix.sh <binary> <config> <thresholds> <scale-list> <results-root> <git-commit> <git-dirty> <random-base-hex> [batch-id]" >&2
    exit 2
fi

binary=$1
config=$2
thresholds=$3
scale_list=$4
results_root=$5
git_commit=$6
git_dirty=$7
random_base_hex=$8
batch_id=${9:-b01}

if [[ ! $random_base_hex =~ ^[0-9a-f]{8}$ ]]; then
    echo "random-base-hex must be eight lowercase hexadecimal characters" >&2
    exit 2
fi
if [[ ! $batch_id =~ ^b[0-9]{2}$ ]]; then
    echo "batch-id must match bNN" >&2
    exit 2
fi

case_index=0
while IFS= read -r scale_id; do
    [[ -n $scale_id ]] || continue
    for symmetric in true false; do
        test_time_utc=$(date -u +%Y%m%dT%H%M%SZ)
        random_value=$(printf '%08x' $((16#$random_base_hex + case_index)))
        "$binary" \
            --config "$config" \
            --thresholds "$thresholds" \
            --results-root "$results_root" \
            --scale "$scale_id" \
            --sym "$symmetric" \
            --run-type formal \
            --git-commit "$git_commit" \
            --git-dirty "$git_dirty" \
            --test-time-utc "$test_time_utc" \
            --random "$random_value"
        case_index=$((case_index + 1))
    done
done < "$scale_list"

echo "[remaining_operator_case-${batch_id}-formal] cases=$case_index status=completed"
