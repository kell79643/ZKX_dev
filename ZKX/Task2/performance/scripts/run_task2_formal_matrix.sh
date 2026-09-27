#!/usr/bin/env bash

set -uo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_task2_formal_matrix.sh \
    [--target <task2_step1_benchmark|task2_step2_benchmark|task2_step3_benchmark|task2_step4_benchmark|task2_step5_benchmark|task2_step6_benchmark|task2_pipeline_benchmark>] \
    --backend <fft_thrust|dlfft> \
    --build-dir <build-dir> \
    --batch-id <unique-batch-id> \
    [--task-scales <json>] \
    [--accuracy-thresholds <json>]

Runs exactly one approved Task2 Step or pipeline target/backend over the frozen 110-case matrix.
Each case uses 20 warmup runs and 100 measured runs. The driver prints a
heartbeat every 60 seconds while a case is running and writes complete case
logs plus matrix_status.csv and matrix_summary.txt under results/formal/.
EOF
}

backend=""
build_dir=""
batch_id=""
target="task2_step1_benchmark"
task_scales="test_all/config/task_benchmarks/task_input_scales.json"
accuracy_thresholds="test_all/config/task_benchmarks/accuracy_thresholds.json"

while test "$#" -gt 0; do
  case "$1" in
    --target)
      test "$#" -ge 2 || { usage >&2; exit 2; }
      target="$2"
      shift 2
      ;;
    --backend)
      test "$#" -ge 2 || { usage >&2; exit 2; }
      backend="$2"
      shift 2
      ;;
    --build-dir)
      test "$#" -ge 2 || { usage >&2; exit 2; }
      build_dir="$2"
      shift 2
      ;;
    --batch-id)
      test "$#" -ge 2 || { usage >&2; exit 2; }
      batch_id="$2"
      shift 2
      ;;
    --task-scales)
      test "$#" -ge 2 || { usage >&2; exit 2; }
      task_scales="$2"
      shift 2
      ;;
    --accuracy-thresholds)
      test "$#" -ge 2 || { usage >&2; exit 2; }
      accuracy_thresholds="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

test "$backend" = "fft_thrust" || test "$backend" = "dlfft" || {
  printf 'invalid backend: %s\n' "$backend" >&2
  exit 2
}
test "$target" = "task2_step1_benchmark" || \
  test "$target" = "task2_step2_benchmark" || \
  test "$target" = "task2_step3_benchmark" || \
  test "$target" = "task2_step4_benchmark" || \
  test "$target" = "task2_step5_benchmark" || \
  test "$target" = "task2_step6_benchmark" || \
  test "$target" = "task2_pipeline_benchmark" || {
  printf 'invalid target: %s\n' "$target" >&2
  exit 2
}
test -n "$build_dir" || { printf 'missing --build-dir\n' >&2; exit 2; }
test -n "$batch_id" || { printf 'missing --batch-id\n' >&2; exit 2; }
test -f "$task_scales" || { printf 'missing task scales: %s\n' "$task_scales" >&2; exit 2; }
test -f "$accuracy_thresholds" || {
  printf 'missing accuracy thresholds: %s\n' "$accuracy_thresholds" >&2
  exit 2
}

exe="${build_dir}/bin/${target}"
root="results/formal/tasks/${backend}/${target}/${batch_id}"
manifest="${root}/matrix_status.csv"

test -x "$exe" || { printf 'missing executable: %s\n' "$exe" >&2; exit 20; }
test ! -e "$root" || { printf 'result root already exists: %s\n' "$root" >&2; exit 21; }

mapfile -t scale_ids < <(
  python3 -c \
    'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8")); print("\n".join(x["scale_id"] for x in d["tasks"]["Task2"]["scales"]))' \
    "$task_scales"
)
test "${#scale_ids[@]}" -eq 110 || {
  printf 'Task2 scale count must be 110, got %s\n' "${#scale_ids[@]}" >&2
  exit 22
}
test "$(printf '%s\n' "${scale_ids[@]}" | sort -u | wc -l)" -eq 110 || {
  printf 'Task2 scale IDs must be unique\n' >&2
  exit 22
}

mkdir -p "${root}/cases" "${root}/driver_logs"
printf 'target,backend,scale_id,run_id,result_path,exit_code,status\n' > "$manifest"

for scale_id in "${scale_ids[@]}"; do
  run_id="${batch_id}_${backend}_${target}_${scale_id}"
  result_path="${root}/cases/${scale_id}"
  log_path="${root}/driver_logs/${scale_id}.log"

  printf '[FORMAL] target=%s backend=%s scale=%s\n' \
    "$target" "$backend" "$scale_id"
  "$exe" \
    --task-scales "$task_scales" \
    --accuracy-thresholds "$accuracy_thresholds" \
    --scale-id "$scale_id" \
    --backend "$backend" \
    --run-id "$run_id" \
    --run-type formal \
    --warmup-runs 20 \
    --measured-runs 100 \
    --output-dir "$result_path" \
    > "$log_path" 2>&1 &
  case_pid=$!
  elapsed_seconds=0
  heartbeat_seconds=0
  while kill -0 "$case_pid" 2>/dev/null; do
    sleep 1
    elapsed_seconds=$((elapsed_seconds + 1))
    heartbeat_seconds=$((heartbeat_seconds + 1))
    if test "$heartbeat_seconds" -ge 60 && kill -0 "$case_pid" 2>/dev/null; then
      printf '[HEARTBEAT] target=%s backend=%s scale=%s pid=%s elapsed_seconds=%s still_running=true\n' \
        "$target" "$backend" "$scale_id" "$case_pid" "$elapsed_seconds"
      heartbeat_seconds=0
    fi
  done

  wait "$case_pid"
  rc=$?
  status=PASS
  test "$rc" -eq 0 || status=FAIL
  printf '%s,%s,%s,%s,%s,%s,%s\n' \
    "$target" "$backend" "$scale_id" "$run_id" "$result_path" "$rc" "$status" \
    >> "$manifest"

  if test "$rc" -ne 0; then
    printf 'status=FAIL\ntarget=%s\nbackend=%s\nfailed_scale=%s\nexit_code=%s\n' \
      "$target" "$backend" "$scale_id" "$rc" > "${root}/matrix_summary.txt"
    exit "$rc"
  fi
done

printf 'status=PASS\ntarget=%s\nbackend=%s\ncases=110\nwarmup_runs=20\nmeasured_runs=100\n' \
  "$target" "$backend" > "${root}/matrix_summary.txt"
printf '[MATRIX] target=%s backend=%s cases=110 status=PASS result_root=%s\n' \
  "$target" "$backend" "$root"
