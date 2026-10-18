"""Download only official, pinned pretrained assets; never send video data."""
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from model_suite import REVISION


def main():
    root=Path('artifacts/assets');root.mkdir(parents=True,exist_ok=True)
    entries=[]
    for name in ['config.json','preprocessor_config.json','modeling_config.py','modeling_videomaev2.py','README.md','model.safetensors']:
        entries.append((root/'videomaev2'/name,'https://huggingface.co/OpenGVLab/VideoMAEv2-Base/resolve/'+REVISION+'/'+name))
    for name in ['yolov8n-cls.pt','yolo11n-cls.pt','yolov8n.pt','yolo11n.pt']:
        entries.append((root/name,'https://github.com/ultralytics/assets/releases/download/v8.3.0/'+name))
    provenance=[]
    for path,url in entries:
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            print('Download',path,flush=True)
            with urllib.request.urlopen(url,timeout=60) as source,path.with_suffix(path.suffix+'.part').open('wb') as out:
                while chunk:=source.read(1024*1024):out.write(chunk)
            os.replace(path.with_suffix(path.suffix+'.part'),path)
        provenance.append(dict(path=str(path),url=url,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2))


if __name__=='__main__':main()
