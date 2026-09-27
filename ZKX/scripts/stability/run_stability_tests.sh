#!/usr/bin/env bash
set -u -o pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

params=""
task1_exe=""
task2_exe=""
run_id=""
backend=""
git_commit=""
git_dirty=""
output_root=""
start_index=""
end_index=""

while (($#)); do
  case "$1" in
    --params) params=$2; shift 2 ;;
    --task1-exe) task1_exe=$2; shift 2 ;;
    --task2-exe) task2_exe=$2; shift 2 ;;
    --run-id) run_id=$2; shift 2 ;;
    --backend) backend=$2; shift 2 ;;
    --git-commit) git_commit=$2; shift 2 ;;
    --git-dirty) git_dirty=$2; shift 2 ;;
    --output-root) output_root=$2; shift 2 ;;
    --case-index) start_index=$2; end_index=$2; shift 2 ;;
    --start-index) start_index=$2; shift 2 ;;
    --end-index) end_index=$2; shift 2 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

for value_name in params task1_exe task2_exe run_id backend git_commit git_dirty output_root start_index end_index; do
  test -n "${!value_name}" || { printf 'Missing required argument: %s\n' "$value_name" >&2; exit 2; }
done
test -f "$params" || { printf 'Parameter file missing: %s\n' "$params" >&2; exit 2; }
test -x "$task1_exe" || { printf 'Task1 executable missing: %s\n' "$task1_exe" >&2; exit 2; }
test -x "$task2_exe" || { printf 'Task2 executable missing: %s\n' "$task2_exe" >&2; exit 2; }
[[ "$start_index" =~ ^[0-9]+$ && "$end_index" =~ ^[0-9]+$ ]] || exit 2
((start_index >= 0 && end_index <= 999 && start_index <= end_index)) || exit 2
test ! -e "$output_root" || { printf 'Refusing to overwrite output root: %s\n' "$output_root" >&2; exit 2; }
mkdir -p "$output_root/cases" "$output_root/logs"
manifest="$output_root/stability_manifest.csv"
printf 'stability_case_index,case_id,task1_rc,task2_rc,status,task1_path,task2_path\n' > "$manifest"

overall_rc=0
for ((index=start_index; index<=end_index; ++index)); do
  case_id=$(python3 - "$params" "$index" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8"))
i=int(sys.argv[2])
case=d["cases"][i]
assert case["stability_case_index"] == i
print(case["case_id"])
PY
  ) || exit 2
  case_root="$output_root/cases/$(printf '%04d' "$index")_${case_id}"
  task1_path="$case_root/Task1"
  task2_path="$case_root/Task2"
  mkdir -p "$case_root"
  "$task1_exe" "$params" "$index" "$task1_path" "$run_id" "$backend" \
    "$git_commit" "$git_dirty" > "$output_root/logs/$(printf '%04d' "$index")_Task1.log" 2>&1
  task1_rc=$?
  "$task2_exe" "$params" "$index" "$task2_path" "$run_id" "$backend" \
    "$git_commit" "$git_dirty" > "$output_root/logs/$(printf '%04d' "$index")_Task2.log" 2>&1
  task2_rc=$?
  status=PASS
  if ((task1_rc != 0 || task2_rc != 0)); then status=FAIL; overall_rc=1; fi
  printf '%s,%s,%s,%s,%s,%s,%s\n' "$index" "$case_id" "$task1_rc" "$task2_rc" \
    "$status" "$task1_path" "$task2_path" >> "$manifest"
  printf '[STABILITY][CASE] index=%s case_id=%s task1_rc=%s task2_rc=%s status=%s\n' \
    "$index" "$case_id" "$task1_rc" "$task2_rc" "$status"
done

python3 - "$output_root" "$start_index" "$end_index" <<'PY'
import csv,json,sys
from pathlib import Path
root=Path(sys.argv[1]); start=int(sys.argv[2]); end=int(sys.argv[3])
rows=list(csv.DictReader((root/'stability_manifest.csv').open(encoding='utf-8')))
summary={
  'start_index':start,'end_index':end,'requested_cases':end-start+1,
  'completed_cases':len(rows),'passed_cases':sum(r['status']=='PASS' for r in rows),
  'failed_cases':sum(r['status']!='PASS' for r in rows),
  'status':'PASS' if len(rows)==end-start+1 and all(r['status']=='PASS' for r in rows) else 'FAIL'
}
(root/'stability_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
(root/'stability_summary.txt').write_text('\n'.join(f'{k}={v}' for k,v in summary.items())+'\n',encoding='utf-8')
PY
exit "$overall_rc"
