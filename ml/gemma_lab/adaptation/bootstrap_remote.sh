#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/nuhack-training
mkdir -p runs/gemma_lab/adaptation_v1 artifacts/gemma_adaptation
python3 -m venv .venv-gemma-adapt
.venv-gemma-adapt/bin/python -m pip install --disable-pip-version-check --quiet --upgrade 'pip==25.2'
.venv-gemma-adapt/bin/python -m pip install --disable-pip-version-check --quiet 'torch==2.8.0' 'torchvision==0.23.0' 'transformers==5.10.1' 'peft==0.19.0' 'bitsandbytes==0.48.2' 'accelerate==1.10.1' 'Pillow==11.2.1' 'av==13.1.0'
HF_HOME=/home/ubuntu/nuhack-training/artifacts/gemma_adaptation/hf_cache .venv-gemma-adapt/bin/python -u - <<'PY'
from huggingface_hub import snapshot_download
import json,time
from pathlib import Path
repo='google/gemma-4-12B-it-qat-q4_0-unquantized';revision='b6ed86275a6a5735884e208bfed95b445a684ca2'
root=Path('artifacts/gemma_adaptation');out=root/'base'
print('Downloading pinned trainable Gemma weights; no private data uploaded',flush=True)
snapshot_download(repo,revision=revision,local_dir=out,allow_patterns=['*.json','*.safetensors','*.model','*.jinja','*.txt','README.md','LICENSE'],max_workers=3)
(root/'source.json').write_text(json.dumps(dict(repo=repo,revision=revision,path=str(out),license='Apache-2.0',downloaded_at=time.time()),indent=2))
print('Trainable base ready',flush=True)
PY
