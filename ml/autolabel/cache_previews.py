"""Cache video-only review previews independently of expensive model inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from .dataset import digest,snapshot
from .pipeline import inside,save
from .training_data import export_clip


def prepare(root,run):
    root=Path(root).resolve();run=inside(run,root/'data/autolabel');plan=json.loads((run/'plan.json').read_text())
    if snapshot(root)!=plan['protected_inputs']:raise ValueError('Source labels changed')
    dest=run/'preview_cache';dest.mkdir(exist_ok=True)
    def one(scan):
        group=scan['group_id'];source=plan['registry'][scan['video']];path=dest/(group+'.mp4');meta_path=dest/(group+'.json')
        provenance=dict(group_id=group,source_sha256=source['sha256'],duration_sec=scan['duration_sec'],fps=12,max_side=768)
        if meta_path.exists():
            old=json.loads(meta_path.read_text())
            if old['provenance']!=provenance or digest(path)!=old['clip']['clip_sha256']:raise ValueError('Preview cache changed')
            return old
        if source['split']!='train' or digest(root/'data/raw/open'/scan['video'])!=source['sha256']:raise ValueError('Source mismatch')
        meta=export_clip(root/'data/raw/open'/scan['video'],path,0,scan['duration_sec'],fps=12,max_side=768)
        result=dict(provenance=provenance,clip=meta);save(meta_path,result);return result
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i,result in enumerate(pool.map(one,plan['scans']),1):print(json.dumps(dict(previews=i,total=len(plan['scans']))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--run',type=Path,default=Path('data/autolabel/detailed_train'));a=p.parse_args();prepare(a.root,a.run)
