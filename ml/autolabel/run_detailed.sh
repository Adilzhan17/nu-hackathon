#!/usr/bin/env bash
# Bounded train-only analysis on the already authorized host. Re-running resumes records.
set -euo pipefail
cd /root/nuhack
run_name="${1:-detailed_next}"
[[ "$run_name" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid run name' >&2; exit 2; }
run_path="data/autolabel/$run_name"
report_path="data/autolabel/${run_name}_report"
remote_host='ubuntu@global.prd.ga.run.brev.nvidia.com'
remote_root='/home/ubuntu/nuhack-training'
ssh_args=(-i /root/.ssh/nuhack_brev -p 21453 -o BatchMode=yes -o ConnectTimeout=15)
scp_args=(-i /root/.ssh/nuhack_brev -P 21453 -o BatchMode=yes -o ConnectTimeout=15)
if [[ ! -f "$run_path/plan.json" ]]; then
  .venv/bin/python -m ml.autolabel.deep prepare --run "$run_path"
fi
if [[ ! -f "$run_path/incident_check_plan.json" ]]; then
  .venv/bin/python -m ml.autolabel.inspect_incidents prepare --run "$run_path"
fi
scp "${scp_args[@]}" ml/autolabel/*.py "$remote_host:$remote_root/ml/autolabel/"
if ! ssh "${ssh_args[@]}" "$remote_host" "test -f $remote_root/$run_path/plan.json"; then
  # A single archive avoids thousands of slow per-frame SFTP requests.
  tar -C data/autolabel -cf - "$run_name" | ssh "${ssh_args[@]}" "$remote_host" "tar -C $remote_root/data/autolabel -xf -"
fi
scp "${scp_args[@]}" "$run_path/incident_check_plan.json" "$remote_host:$remote_root/$run_path/"
ssh "${ssh_args[@]}" "$remote_host" "cd $remote_root && for batch in 1 2 3 4 5; do if test -f $run_path/inference.json; then break; fi; timeout 540 .venv-autolabel/bin/python -m ml.autolabel.deep infer --run $run_path --model-path data/autolabel/models/qwen2.5-vl-7b --max-windows 80 --max-seconds 450 || exit \$?; done; test -f $run_path/inference.json" | tee -a "$run_path/inference_batches.log"
ssh "${ssh_args[@]}" "$remote_host" "cd $remote_root && for batch in 1 2; do if .venv-autolabel/bin/python -m ml.autolabel.artifact_status $run_path/incident_check_summary.json; then break; fi; timeout 420 .venv-autolabel/bin/python -m ml.autolabel.inspect_incidents infer --run $run_path --model-path data/autolabel/models/qwen2.5-vl-7b --max-seconds 360 || exit \$?; done; .venv-autolabel/bin/python -m ml.autolabel.artifact_status $run_path/incident_check_summary.json" | tee -a "$run_path/incident_checks.log"
ssh "${ssh_args[@]}" "$remote_host" "tar -C $remote_root/$run_path -cf - inference.json inference_progress.json inference_code inference_records incident_check_records incident_check_summary.json incident_check_run.json proposals.jsonl event_details.jsonl window_analysis.jsonl raw_responses.jsonl rejected.jsonl" | tar -C "$run_path" -xf -
if [[ ! -f "$run_path/incident_check_code.py" ]]; then
  scp "${scp_args[@]}" "$remote_host:$remote_root/ml/autolabel/inspect_incidents.py" "$run_path/incident_check_code.py"
fi
if [[ ! -f "$report_path/summary.json" ]]; then
  .venv/bin/python -m ml.autolabel.deep_report --run "$run_path" --output "$report_path"
fi
printf '%s\n' "$report_path/index.html"
