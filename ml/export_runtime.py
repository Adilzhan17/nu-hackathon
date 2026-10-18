"""Build a private API runtime package without training/test videos or secrets."""
import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

FILES = [
    'ml/serve_jury.py', 'ml/jury_api.py', 'ml/video_jobs.py', 'ml/infer_video.py',
    'ml/model_suite.py', 'ml/train_x3d.py', 'ml/requirements-runtime.txt',
    'ml/licenses/pytorchvideo-Apache-2.0.txt', 'ml/RUNNING.txt',
    'artifacts/models/x3d/model.pt',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('dist/forte-api-runtime.tar.gz'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    files = {}
    for name in FILES:
        path = root/name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing regular runtime file: '+name)
        files[name] = path
    manifest = {'files': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()},
                'model': 'x3d', 'contains_videos': False, 'contains_api_key': False,
                'includes_python_environment': False, 'quality_claim': None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, 'w:gz') as archive:
        for name, path in files.items():
            archive.add(path, arcname=name, recursive=False)
        data = (json.dumps(manifest, indent=2)+'\n').encode()
        info = tarfile.TarInfo('MANIFEST.json');info.size = len(data);info.mode = 0o600
        archive.addfile(info, io.BytesIO(data))
    args.output.chmod(0o600)
    print(json.dumps({'archive': str(args.output), 'files': len(files), 'bytes': args.output.stat().st_size}))


if __name__ == '__main__':
    main()
