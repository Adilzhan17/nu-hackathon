#!/usr/bin/env bash
# Bounded pilot on the existing authorized GPU. Never prints/copies the SSH key.
set -euo pipefail
cd /root/nuhack
run_name="${1:-pilot_$(date -u +%Y%m%dT%H%M%S)}"
if [[ $# -gt 0 ]]; then shift; fi
[[ "$run_name" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid run name' >&2; exit 2; }
if [[ $# -eq 0 ]]; then set -- videos/Х-21.mov videos/Х-22.mov videos/Х-5.mov; fi
run_path="data/autolabel/$run_name"
remote_host='ubuntu@global.prd.ga.run.brev.nvidia.com'
remote_root='/home/ubuntu/nuhack-training'
ssh_args=(-i /root/.ssh/nuhack_brev -p 21453 -o BatchMode=yes -o ConnectTimeout=15)
scp_args=(-i /root/.ssh/nuhack_brev -P 21453 -o BatchMode=yes -o ConnectTimeout=15)
.venv/bin/python -m ml.autolabel.pipeline prepare --run "$run_path" --videos "$@"
ssh "${ssh_args[@]}" "$remote_host" "test ! -e $remote_root/$run_path && test -x $remote_root/.venv-autolabel/bin/python && test -f $remote_root/data/autolabel/models/qwen2.5-vl-7b/autolabel_model_lock.json"
scp "${scp_args[@]}" ml/autolabel/*.py "$remote_host:$remote_root/ml/autolabel/"
scp -r "${scp_args[@]}" "$run_path" "$remote_host:$remote_root/data/autolabel/"
# infer acquires the shared flock itself, waits <=30 s, then releases on exit.
ssh "${ssh_args[@]}" "$remote_host" "cd $remote_root && timeout 600 .venv-autolabel/bin/python -m ml.autolabel.pipeline infer --run $run_path --model-path data/autolabel/models/qwen2.5-vl-7b" | tee "$run_path/inference.log"
scp "${scp_args[@]}" "$remote_host:$remote_root/$run_path/*.jsonl" "$run_path/"
scp "${scp_args[@]}" "$remote_host:$remote_root/$run_path/inference.json" "$run_path/"
scp -r "${scp_args[@]}" "$remote_host:$remote_root/$run_path/inference_code" "$run_path/"
ssh "${ssh_args[@]}" "$remote_host" "$remote_root/.venv-autolabel/bin/pip freeze" > "$run_path/gpu_environment.txt"
.venv/bin/python -m ml.autolabel.pipeline finalize --run "$run_path"
