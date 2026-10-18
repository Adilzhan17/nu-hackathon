import json,hashlib
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
root=Path.cwd();p=root/'artifacts/assets/scenario_qwen3';source=json.loads((p/'source.json').read_text())
for name,sha in source['files'].items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==sha,name
processor=AutoProcessor.from_pretrained(p,local_files_only=True,min_pixels=64*32*32,max_pixels=128*32*32)
model=Qwen3VLForConditionalGeneration.from_pretrained(p,local_files_only=True,dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
messages=[{'role':'user','content':[{'type':'image','image':Image.new('RGB',(384,384),'blue')},{'type':'text','text':'Name the main color in one word.'}]}]
batch=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda')
with torch.inference_mode():output=model.generate(**batch,max_new_tokens=8,do_sample=False)
answer=processor.batch_decode(output[:,batch['input_ids'].shape[1]:],skip_special_tokens=True)[0]
report=dict(status='passed',gpu=torch.cuda.get_device_name(),dtype=str(next(model.parameters()).dtype),answer=answer,max_vram_bytes=torch.cuda.max_memory_allocated(),weights_verified=True,model=source['repo'],revision=source['revision'])
(root/'runs/migration').mkdir(parents=True,exist_ok=True)
(root/'runs/migration/t4_smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
