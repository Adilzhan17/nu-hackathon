"""Run the organizer's unchanged pipeline with the required T4 FP16 dtype."""
import hashlib,json,runpy,sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path.cwd()))
source=Path('ml/narration_training/organizer/analyze.py')
original=source.read_text()
old='dtype=torch.bfloat16,attn_implementation='
if original.count(old)!=1:
    raise RuntimeError('Organizer source changed: inspect dtype before resuming')
if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
patched=original.replace(old,'dtype=torch.float16,attn_implementation=',1)
record=dict(source=str(source),source_sha256=hashlib.sha256(original.encode()).hexdigest(),executed_sha256=hashlib.sha256(patched.encode()).hexdigest(),dtype='float16',gpu=torch.cuda.get_device_name(),torch=torch.__version__,change='Only model loading dtype: bfloat16 -> float16')
if '--run' in sys.argv:
    dest=Path(sys.argv[sys.argv.index('--run')+1]);dest.mkdir(parents=True,exist_ok=True)
    (dest/'t4_migration_runtime.json').write_text(json.dumps(record,indent=2))
exec(compile(patched,str(source),'exec'),{'__name__':'__main__','__file__':str(source),'__package__':'ml.narration_training.organizer'})
