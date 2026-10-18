#!/bin/bash
set -eu
cd /home/ubuntu/nuhack-training
mkdir -p runs/large_v1
.venv-training/bin/python - <<'PY'
import time
from pathlib import Path
from ml.train_x3d import atomic_json
p=Path('runs/large_v1/status.json')
if p.exists():raise SystemExit('Run already exists; use an explicitly new run')
atomic_json(p,dict(model_key='videomaev2_large',name='VideoMAE V2 Large',status='queued',stage='queued',epochs=16,total=79,history=[],updated_at=time.time(),message='Ожидаем общий GPU-lock после партии авторазметки',training_mode='full_finetune'))
PY
set +e
flock -w 1200 .gpu-work.lock timeout --signal=TERM --kill-after=30 960 .venv-training/bin/python -u ml/train_large.py --output runs/large_v1 --epochs 16 --max-seconds 900
result=$?
set -e
if [ "$result" -ne 0 ]; then
  RUN_EXIT_CODE="$result" .venv-training/bin/python - <<'PY'
import json,time,os
from pathlib import Path
from ml.train_x3d import atomic_json
p=Path('runs/large_v1/status.json');s=json.loads(p.read_text())
if s['status'] not in ('failed','stopped','completed'):
    s.update(status='failed',stage='failed',updated_at=time.time(),message='Процесс завершился: код '+os.environ['RUN_EXIT_CODE'])
    atomic_json(p,s)
PY
fi
exit "$result"
