"""Explicit network-only preparation step; never reads videos or annotations."""
import argparse
import json
from pathlib import Path


def main():
    from huggingface_hub import snapshot_download
    p = argparse.ArgumentParser()
    p.add_argument('--lock', type=Path, default=Path('data/autolabel/model_lock.json'))
    p.add_argument('--destination', type=Path, required=True)
    args = p.parse_args()
    lock = json.loads(args.lock.read_text())
    snapshot_download(lock['model_id'], revision=lock['model_revision'], local_dir=str(args.destination),
                      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'README.md', 'LICENSE*'],
                      max_workers=4)
    (args.destination / 'autolabel_model_lock.json').write_text(json.dumps(lock, indent=2))
    print('Downloaded pinned model:', lock['model_id'], lock['model_revision'])


if __name__ == '__main__':
    main()
