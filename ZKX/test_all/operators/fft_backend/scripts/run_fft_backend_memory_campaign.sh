#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: run_fft_backend_memory_campaign.sh FFT_THRUST_EXE DLFFT_EXE OUTPUT_DIR" >&2
    exit 64
fi

fft_thrust_exe=$1
dlfft_exe=$2
output_dir=$3
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
raw_dir="${output_dir}/raw"
analysis_dir="${output_dir}/analysis"
mkdir -p "${raw_dir}" "${analysis_dir}"

"${fft_thrust_exe}" --scenario verify --output "${raw_dir}/verify_fft_thrust.csv"
"${dlfft_exe}" --scenario verify --output "${raw_dir}/verify_dlfft.csv"
python3 "${script_dir}/../analysis/analyze_fft_backend_memory.py" \
    --raw-dir "${raw_dir}" --output-dir "${analysis_dir}" --verify-only

for run_id in 1 2 3; do
    "${fft_thrust_exe}" --scenario lifecycle --run-id "${run_id}" \
        --warmup 100 --iterations 2000 \
        --output "${raw_dir}/lifecycle_run_${run_id}_fft_thrust.csv"
    "${dlfft_exe}" --scenario lifecycle --run-id "${run_id}" \
        --warmup 100 --iterations 2000 \
        --output "${raw_dir}/lifecycle_run_${run_id}_dlfft.csv"
done

for run_id in 1 2 3; do
    "${fft_thrust_exe}" --scenario resident --run-id "${run_id}" \
        --warmup 100 --iterations 2000 \
        --output "${raw_dir}/resident_run_${run_id}_fft_thrust.csv"
    "${dlfft_exe}" --scenario resident --run-id "${run_id}" \
        --warmup 100 --iterations 2000 \
        --output "${raw_dir}/resident_run_${run_id}_dlfft.csv"
done

python3 "${script_dir}/../analysis/analyze_fft_backend_memory.py" \
    --raw-dir "${raw_dir}" --output-dir "${analysis_dir}"

echo "[FFT-BACKEND-MEMORY][CAMPAIGN] output=${output_dir} status=complete"
