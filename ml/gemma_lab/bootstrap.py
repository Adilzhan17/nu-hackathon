"""Download pinned local Gemma/llama.cpp assets. No video or inference requests."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time
import urllib.request

REVISION = '29d097773436b69ff9feafd636ab4cf873786537'
REPO = 'google/gemma-4-12B-it-qat-q4_0-gguf'
FILES = {
    'gemma-4-12b-it-qat-q4_0.gguf': '93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b',
    'mmproj-gemma-4-12b-it-qat-q4_0.gguf': 'cb018338a7538a9814d994bfe54644c71eb7ed54e31eae2f721e45fd3c260da7',
}
RUNTIME = {
    'llama-b11716-bin-ubuntu-cuda-12.8-x64.tar.gz': 'f467d630fd94d0bcf2a51269f0ecb6dd84125be620daca6497340a93373b2e69',
    'cudart-llama-b11716-bin-ubuntu-cuda-12.8-x64.tar.gz': '93138d385a12272383bb51bbc7bbe03b6f718136737a500c0c8b4db54cb18d24',
}


def digest(path):
    sha=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):sha.update(chunk)
    return sha.hexdigest()


def download(url, path, expected=None):
    if path.exists() and (not expected or digest(path) == expected):
        return
    partial = path.with_suffix(path.suffix + '.partial')
    subprocess.run(['curl', '-fL', '--retry', '2', '--connect-timeout', '15',
                    '--max-time', '650', '-sS', '-o', str(partial), url], check=True, timeout=700)
    if expected and digest(partial) != expected:
        raise ValueError('Checksum mismatch: ' + path.name)
    partial.replace(path)
    print(json.dumps({'downloaded': path.name, 'bytes': path.stat().st_size}), flush=True)


def main(root):
    assets = root / 'artifacts/gemma_lab'; assets.mkdir(parents=True, exist_ok=True)
    runtime = assets / 'runtime'; runtime.mkdir(exist_ok=True)
    for name, sha in RUNTIME.items():
        archive = assets / name
        download('https://github.com/ggml-org/llama.cpp/releases/download/b11716/' + name, archive, sha)
        with tarfile.open(archive) as t:
            t.extractall(runtime, filter='data')
    for name, sha in FILES.items():
        download(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}', assets / name, sha)
    download(f'https://huggingface.co/{REPO}/resolve/{REVISION}/README.md', assets / 'GEMMA4_MODEL_CARD.md')
    download('https://www.apache.org/licenses/LICENSE-2.0.txt', assets / 'LICENSE.txt')
    source = dict(repo=REPO, revision=REVISION, base_model='google/gemma-4-12B',
                  quantization='QAT Q4_0', files=FILES, runtime_release='b11716',
                  runtime_archives=RUNTIME, license='Apache-2.0',
                  license_url='https://www.apache.org/licenses/LICENSE-2.0', downloaded_at=time.time())
    (assets / 'source.json').write_text(json.dumps(source, indent=2)+'\n')
    print(json.dumps({'ready': True, 'assets': str(assets)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root.resolve())
