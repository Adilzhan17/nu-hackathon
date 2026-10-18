"""Targeted local visual-state checks around organizer anchors; never legal ground truth."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from .dataset import read_jsonl,digest
from .pipeline import save,jsonl,inside,gpu_lock

PROMPT='''Inspect this numbered sequence for visible states, not legal compliance.
Do not identify people, read personal data, use audio, or infer an absent object from an unclear frame.
Find whether the WORKER visibly leaves the workstation (not a customer merely walking away).
If visible, departure_frames contains two or more ordered frame indices showing that departure; otherwise [].
Independently list only visibly supported states from:
safe_door_open, cash_on_work_surface, key_in_safe_lock.
A cabinet must not be called a safe without clear visual evidence. Paper is not automatically money.
Small unidentified items must not be called keys. Require at least two distinct frames for each state.
An empty states list is acceptable. Do not conclude that a violation occurred.
Use supplied integer frame indices exactly. confidence is an uncalibrated self-rating, not a probability.
Return ONLY JSON:
{"departure_frames":[],"departure_confidence":0.0,"states":[
{"state":"safe_door_open","evidence_frames":[0,1],"confidence":0.5,"uncertainty_reason":"specific limitation or empty"}],
"uncertainty_reason":"specific visibility limitation or empty"}
The example indices are placeholders. Do not fabricate unseen states.
'''
STATES={'safe_door_open','cash_on_work_surface','key_in_safe_lock'}


def prepare(run):
    plan=json.loads((run/'plan.json').read_text());checks=[];normal_controls=0
    for scan in plan['scans']:
        group=scan['group_id'];duration=scan['duration_sec'];reviews=plan['reviews'][group]
        positives=sorted({float(r['anchor_sec']) for r in reviews if r['source_verdict']=='нарушение' and r.get('anchor_sec')})
        if positives:anchors=positives;role='organizer_anchor'
        elif normal_controls<3:anchors=[duration*.65];normal_controls+=1;role='normal_video_control'
        else:continue
        all_frames={f['timestamp']:f for c in plan['candidates'] if c['group_id']==group for f in c['frames']}
        for anchor in anchors:
            start=max(0,min(duration-8,anchor-4));end=min(duration,start+8)
            available=sorted(t for t in all_frames if start<=t<=end)
            # Use at most 33 distinct actual frames; no timestamp fabrication.
            if len(available)>33:available=[available[round(i*(len(available)-1)/32)] for i in range(33)]
            if len(available)<2:continue
            checks.append(dict(check_id=f'check_{len(checks):03d}',video=scan['video'],group_id=group,split='train',
                               start_sec=start,end_sec=end,anchor_sec=anchor,sampling_origin=role,
                               organizer_video_verdict=reviews[0]['source_verdict'],frames=[all_frames[t] for t in available]))
    save(run/'incident_check_plan.json',dict(source_plan_sha256=digest(run/'plan.json'),checks=checks))
    print(json.dumps(dict(visual_state_checks=len(checks))))


def decode(raw,check):
    text=raw.strip()
    if text.startswith('```json') and text.endswith('```'):text=text[7:-3].strip()
    elif text.startswith('```') and text.endswith('```'):text=text[3:-3].strip()
    def bad(value):raise ValueError('Nonfinite value')
    d=json.loads(text,parse_constant=bad)
    if not isinstance(d,dict) or set(d)!={'departure_frames','departure_confidence','states','uncertainty_reason'}:raise ValueError('Bad visual state schema')
    from .schema import number
    n=len(check['frames'])
    def indices(xs,empty=False):
        return isinstance(xs,list) and ((empty and not xs) or len(xs)>=2) and all(type(i) is int and 0<=i<n for i in xs) and xs==sorted(set(xs))
    if not indices(d['departure_frames'],True) or not number(d['departure_confidence']) or not 0<=d['departure_confidence']<=1 or not isinstance(d['uncertainty_reason'],str):raise ValueError('Invalid departure evidence')
    if not isinstance(d['states'],list):raise ValueError('Invalid states')
    for s in d['states']:
        if not isinstance(s,dict) or set(s)!={'state','evidence_frames','confidence','uncertainty_reason'} or s['state'] not in STATES or not indices(s['evidence_frames']) or not number(s['confidence']) or not 0<=s['confidence']<=1 or not isinstance(s['uncertainty_reason'],str):raise ValueError('Invalid state evidence')
    return d


def infer(args):
    from .vlm import LocalVLM,offline
    offline()
    from PIL import Image
    import torch
    if not 1<=args.max_seconds<=600:raise ValueError('State-check budget must be 1..600 seconds')
    run=inside(args.run,args.root/'data/autolabel');plan=json.loads((run/'plan.json').read_text());checkplan=json.loads((run/'incident_check_plan.json').read_text())
    if checkplan['source_plan_sha256']!=digest(run/'plan.json'):raise ValueError('Plan changed')
    audit=dict(check_plan_sha256=digest(run/'incident_check_plan.json'),source_code_sha256=digest(Path(__file__)),vlm_code_sha256=digest(Path(__file__).with_name('vlm.py')),prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),model_id=plan['model_lock']['model_id'],model_revision=plan['model_lock']['model_revision'],max_pixels_per_frame=512*28*28,max_new_tokens=700)
    audit_path=run/'incident_check_run.json'
    if audit_path.exists() and json.loads(audit_path.read_text())!=audit:raise ValueError('State check code/plan changed; use a new run')
    save(audit_path,audit)
    output=run/'incident_check_records';output.mkdir(exist_ok=True)
    checks=checkplan['checks'];start=time.monotonic()
    with gpu_lock(args.gpu_lock):
        model=LocalVLM(args.model_path,plan['model_lock'])
        from transformers import AutoProcessor
        model.processor=AutoProcessor.from_pretrained(str(args.model_path),local_files_only=True,trust_remote_code=False,use_fast=False,min_pixels=128*28*28,max_pixels=512*28*28)
        for check in checks:
            target=output/(check['check_id']+'.json')
            if target.exists():continue
            if time.monotonic()-start>args.max_seconds:break
            source=plan['registry'][check['video']]
            if source['split']!='train' or source['group_id']!=check['group_id']:raise ValueError('Non-train check')
            images=[];content=[dict(type='text',text=PROMPT)]
            for i,f in enumerate(check['frames']):
                path=inside(run/f['path'],run/'frames')
                if digest(path)!=f['sha256']:raise ValueError('Changed frame')
                images.append(Image.open(path).convert('RGB'));content.extend([dict(type='text',text=f'Frame index {i}:'),dict(type='image')])
            prompt=model.processor.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
            inputs=model.processor(text=[prompt],images=images,padding=True,return_tensors='pt').to('cuda');t0=time.monotonic()
            with torch.inference_mode():out=model.model.generate(**inputs,max_new_tokens=700,do_sample=False)
            raw=model.processor.batch_decode(out[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
            try:parsed=decode(raw,check);error=''
            except (ValueError,TypeError,KeyError) as exc:parsed=None;error=str(exc)
            record=dict(check_id=check['check_id'],raw_response=raw,parsed=parsed,error=error,elapsed_sec=time.monotonic()-t0,model_revision=audit['model_revision'],prompt_sha256=audit['prompt_sha256'],check_plan_sha256=audit['check_plan_sha256'])
            temp=target.with_suffix('.tmp');save(temp,record);temp.replace(target)
            print(json.dumps(dict(check=check['check_id'],valid=parsed is not None,states=len(parsed['states']) if parsed else 0,seconds=round(record['elapsed_sec'],2))),flush=True)
    rows=[json.loads(p.read_text()) for p in sorted(output.glob('*.json'))]
    save(run/'incident_check_summary.json',dict(completed=len(rows)==len(checks),checks=len(rows),total_checks=len(checks),
         valid=sum(r['parsed'] is not None for r in rows),inference_seconds=sum(r['elapsed_sec'] for r in rows),
         source_code_sha256=digest(Path(__file__)),model_id=plan['model_lock']['model_id'],model_revision=plan['model_lock']['model_revision']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('command',choices=['prepare','infer']);p.add_argument('--run',type=Path,required=True);p.add_argument('--model-path',type=Path);p.add_argument('--max-seconds',type=int,default=240);p.add_argument('--gpu-lock',type=Path,default=Path('/home/ubuntu/nuhack-training/.gpu-work.lock'))
    args=p.parse_args()
    if args.command=='prepare':prepare(inside(args.run,args.root/'data/autolabel'))
    else:infer(args)


if __name__=='__main__':main()
