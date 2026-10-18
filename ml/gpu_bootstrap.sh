#!/usr/bin/env bash
set -euo pipefail
run_dir="$1"
finish() {
  code=$?
  if [ "$code" -ne 0 ]; then
    python3 - "$run_dir" "$code" <<'PY'
import json, os, sys, time
from pathlib import Path
p=Path(sys.argv[1])/'status.json'
state=json.loads(p.read_text()) if p.exists() else {}
state.update(status='failed',stage='failed',updated_at=time.time(),message='Ошибка подготовки/обучения. Подробности в журнале.',exit_code=int(sys.argv[2]))
t=p.with_suffix('.tmp'); t.write_text(json.dumps(state)); os.replace(t,p)
PY
  fi
}
trap finish EXIT
echo 'Подготовка отдельного Python-окружения для X3D-S'
python3 -m venv .venv-training
.venv-training/bin/python -m pip install --disable-pip-version-check torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.venv-training/bin/python -m pip install --disable-pip-version-check pytorchvideo==0.1.5 av==16.0.1 numpy==2.2.6
.venv-training/bin/python -m pip freeze > "$run_dir/environment.txt"
if [ -f "$run_dir/STOP" ]; then
  echo 'Остановка запрошена до запуска обучения'
  python3 - "$run_dir" <<'PY'
import json,sys,time
from pathlib import Path
(Path(sys.argv[1])/'status.json').write_text(json.dumps(dict(status='stopped',stage='stopped',updated_at=time.time(),message='Остановлено до обучения. Аренда VM продолжается.')))
PY
  exit 0
fi
.venv-training/bin/python -u ml/train_x3d.py --run-dir "$run_dir" --epochs 16 --max-seconds 7200
