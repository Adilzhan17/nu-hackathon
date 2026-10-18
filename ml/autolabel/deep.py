"""Detailed train-only scan with full temporal coverage and bounded resumable local inference."""
import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

from .dataset import registry, snapshot, read_jsonl, digest, csv_rows
from .pipeline import inside, save, jsonl, gpu_lock
from .schema import validate_proposal, ACTIONS, OBJECTS

DEEP_PROMPT = '''Examine this chronological sequence of numbered video frames carefully.
Describe the visible sequence in Russian, then annotate distinct observable actions.
Attend to hands, held objects, transfers, bundles, counting machines, cabinet/safe doors,
keys, documents, stamps, and entering/leaving the workstation. Include intermediate
steps such as unpacking, loading a counter and bundling, when actually visible.
Do not identify people, read personal details, analyze audio or infer intentions/legal violations.
Paper is not necessarily money, a cabinet is not necessarily a safe, a small item is not necessarily a key.
Use object_handling with unidentified_item for visible manipulation whose object is unclear.
Do not infer absence from one frame. Describe limitations instead of inventing visibility.
Use action codes: ACTIONS_PLACEHOLDER
Use object codes: OBJECTS_PLACEHOLDER
cash_received/transferred means receiving from/giving to another person, from worker perspective.
cash_counting requires a counting sequence, not just holding or moving money.
Use exact supplied INTEGER FRAME INDICES, not timestamps. start_frame/end_frame enclose
the observed event; evidence_frames must contain at least two different indices showing change.
Do not use the whole window unless the action really spans it. Include up to 6 distinct events.
For each event give a short visible before/during/after description and a specific uncertainty.
confidence is an uncalibrated model self-rating, not a probability. An empty events list is allowed.
Return only JSON:
{"sequence_summary":"brief chronological description, no identities or personal data",
"events":[{"observed_action":"object_handling","objects":["unidentified_item"],
"start_frame":0,"end_frame":3,"evidence_frames":[0,2,3],"confidence":0.5,
"uncertainty_reason":"specific uncertainty or empty string","evidence_description":"visible changes"}]}
The example indices are placeholders; use the actual numbered frames.
'''


def coverage_windows(duration, window=6., stride=4.):
    if not 0 < stride < window or duration <= 0:
        raise ValueError('Invalid coverage schedule')
    starts = [0.]
    while starts[-1]+window < duration-1e-6:
        next_start = min(starts[-1]+stride, max(0., duration-window))
        if next_start <= starts[-1]+1e-6:
            break
        starts.append(next_start)
    return [dict(start_sec=round(a, 6), end_sec=min(duration,round(a+window, 6)),
                 reason='full_temporal_coverage', dense_fps=4.) for a in starts]


def prepare(args):
    from .sampling import frames, coarse_scan, candidates, metadata
    run = inside(args.run, args.root/'data/autolabel')
    if run.exists():
        raise FileExistsError('Use a new detailed run directory')
    before = snapshot(args.root)
    reg, reviews, observations = registry(args.root)
    originals = csv_rows(args.root/'data/training/manifest.csv')
    wanted = args.videos or sorted({r['video'] for r in originals if r['split']=='train'})
    selected, seen = [], set()
    for v in wanted:
        if v not in reg or reg[v]['split'] != 'train':
            raise ValueError('Only frozen train videos are allowed')
        if reg[v]['group_id'] not in seen:
            selected.append(v); seen.add(reg[v]['group_id'])
    run.mkdir(parents=True); (run/'frames').mkdir()
    scans, all_candidates, durations = [], [], {}
    for video in selected:
        info=reg[video]; path=args.root/'data/raw/open'/video
        meta=metadata(path); duration=meta['duration_sec']; durations[info['group_id']]=duration
        scan=coarse_scan(path,duration,2.)
        save(run/f"scan_{info['group_id']}.json",scan)
        intervals=coverage_windows(duration)
        # Independent detailed windows at the two strongest separated visual changes.
        for peak in candidates(scan,duration,limit=2,window=3.):
            intervals.append(dict(peak,reason='change_refinement',dense_fps=6.))
        intervals.sort(key=lambda x:(x['start_sec'],x['end_sec']))
        scans.append(dict(video=video,group_id=info['group_id'],**meta,coarse_samples=len(scan),
                          candidate_intervals=len(intervals),scheduled_coverage_fraction=1.))
        for c in intervals:
            c['end_sec']=min(duration,c['end_sec'])
            cid=f'candidate_{len(all_candidates):05d}'; frame_rows=[]
            for i,(ts,im) in enumerate(frames(path,c['start_sec'],c['end_sec'],c['dense_fps'],max_side=960)):
                rel=f'frames/{cid}_{i:03d}.jpg';im.save(run/rel,quality=93)
                frame_rows.append(dict(timestamp=ts,path=rel,sha256=digest(run/rel),width=im.width,height=im.height))
            if len(frame_rows)<2:
                raise ValueError('Insufficient frames in scheduled interval')
            all_candidates.append(dict(c,candidate_id=cid,video=video,group_id=info['group_id'],split='train',frames=frame_rows))
        print(json.dumps(dict(prepared=video,windows=len(intervals),duration=round(duration,2))),flush=True)
    if snapshot(args.root)!=before:
        raise RuntimeError('Source annotation files changed during preparation')
    plan=dict(version=2,mode='detailed',protected_inputs=before,registry=reg,
              reviews={g:reviews[g] for g in seen},observations={g:observations[g] for g in seen},
              selected_videos=selected,durations=durations,scans=scans,candidates=all_candidates,
              model_lock=json.loads((args.root/'data/autolabel/model_lock.json').read_text()),
              sampling=dict(coarse_fps=2,coverage_window=6,coverage_stride=4,dense_fps=4,
                            refinement_window=3,refinement_fps=6,refinement_per_video=2))
    save(run/'plan.json',plan)
    print(json.dumps(dict(videos=len(scans),windows=len(all_candidates),source_seconds=sum(durations.values()))),flush=True)


def parse_deep(raw):
    t=raw.strip()
    if t.startswith('```json') and t.endswith('```'): t=t[7:-3].strip()
    elif t.startswith('```') and t.endswith('```'): t=t[3:-3].strip()
    def bad_constant(value): raise ValueError('Nonfinite JSON constant')
    d=json.loads(t,parse_constant=bad_constant)
    if not isinstance(d,dict) or set(d)!={'sequence_summary','events'} or not isinstance(d['sequence_summary'],str):
        raise ValueError('Expected sequence_summary and events')
    if not isinstance(d['events'],list) or len(d['events'])>6:
        raise ValueError('At most six events per window')
    required={'observed_action','objects','start_frame','end_frame','evidence_frames','confidence','uncertainty_reason','evidence_description'}
    if any(not isinstance(e,dict) or set(e)!=required for e in d['events']):
        raise ValueError('Unexpected event structure')
    return d


def convert_event(e,c,plan,pid):
    n=len(c['frames']); indices=e['evidence_frames']
    if (type(e['start_frame']) is not int or type(e['end_frame']) is not int
        or not 0<=e['start_frame']<e['end_frame']<n or not isinstance(indices,list)
        or len(indices)<2 or any(type(i) is not int or not e['start_frame']<=i<=e['end_frame'] for i in indices)
        or indices!=sorted(set(indices)) or not isinstance(e['evidence_description'],str)):
        raise ValueError('Invalid frame-index sequence')
    if not isinstance(e['uncertainty_reason'],str): raise ValueError('Invalid uncertainty')
    ts=[f['timestamp'] for f in c['frames']]
    uncertainty=e['uncertainty_reason']
    if e['start_frame']==0 or e['end_frame']==n-1:
        uncertainty += ('; ' if uncertainty else '')+'Граница действия может выходить за окно; проверить соседний контекст.'
    p=dict(video=c['video'],group_id=c['group_id'],split='train',candidate_id=c['candidate_id'],proposal_id=pid,
           start_sec=ts[e['start_frame']],end_sec=ts[e['end_frame']],observed_action=e['observed_action'],
           objects=e['objects'],evidence_timestamps=[ts[i] for i in indices],confidence=e['confidence'],
           uncertainty_reason=uncertainty,model_id=plan['model_lock']['model_id'],model_revision=plan['model_lock']['model_revision'],
           review_status='proposed',confidence_kind='uncalibrated_model_self_rating')
    validate_proposal(p,plan['registry'],plan['durations'][c['group_id']],ts)
    return p


def infer(args):
    from .vlm import LocalVLM,offline
    offline()
    from PIL import Image
    import torch
    run=inside(args.run,args.root/'data/autolabel');plan=json.loads((run/'plan.json').read_text())
    if (run/'inference.json').exists(): raise FileExistsError('Detailed inference already complete')
    plan_hash=digest(run/'plan.json')
    state_path=run/'inference_progress.json'
    if state_path.exists() and json.loads(state_path.read_text())['plan_sha256']!=plan_hash:
        raise ValueError('Cannot resume a changed plan')
    # Per-candidate atomic records are the only resume authority; derived JSONL is rebuilt.
    records=run/'inference_records'; records.mkdir(exist_ok=True)
    code_dir=run/'inference_code'
    if not code_dir.exists():
        code_dir.mkdir()
        for name in ('deep.py','vlm.py','schema.py','pipeline.py'):
            shutil.copy2(Path(__file__).with_name(name),code_dir/name)
    else:
        for name in ('deep.py','vlm.py','schema.py','pipeline.py'):
            if digest(code_dir/name)!=digest(Path(__file__).with_name(name)):
                raise ValueError('Resume requires the same inference code')
    groups={}
    for r in plan['registry'].values(): groups.setdefault(r['group_id'],set()).add(r['split'])
    if any(len(s)!=1 for s in groups.values()): raise ValueError('Group leakage')
    for c in plan['candidates']:
        source=plan['registry'][c['video']]
        if source['split']!='train' or source['group_id']!=c['group_id'] or not 0<=c['start_sec']<c['end_sec']<=plan['durations'][c['group_id']]:
            raise ValueError('Invalid candidate source/time')
    pending=[c for c in plan['candidates'] if not (records/(c['candidate_id']+'.json')).exists()]
    pending=pending[:args.max_windows]
    start=time.monotonic(); completed=0
    prompt_text=DEEP_PROMPT.replace('ACTIONS_PLACEHOLDER',', '.join(ACTIONS)).replace('OBJECTS_PLACEHOLDER',', '.join(OBJECTS))
    if pending:
      with gpu_lock(args.gpu_lock):
        model=LocalVLM(args.model_path,plan['model_lock'])
        # Keep a bounded visual token budget. Broad view, not a fictitious object crop.
        for c in pending:
            if completed and time.monotonic()-start>=args.max_seconds: break
            images=[]
            for f in c['frames']:
                path=inside(run/f['path'],run/'frames')
                if digest(path)!=f['sha256']: raise ValueError('Frame provenance mismatch')
                images.append(Image.open(path).convert('RGB'))
            content=[{'type':'text','text':prompt_text}]
            for i,f in enumerate(c['frames']):
                content.extend([{'type':'text','text':f'Frame index {i}, time {f["timestamp"]:.6f} sec:'},{'type':'image'}])
            text=model.processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
            inputs=model.processor(text=[text],images=images,padding=True,return_tensors='pt').to('cuda')
            t0=time.monotonic()
            with torch.inference_mode():
                out=model.model.generate(**inputs,max_new_tokens=1100,do_sample=False)
            raw=model.processor.batch_decode(out[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
            props=[];details=[];errors=[];summary='';status='ok'
            try:
                parsed=parse_deep(raw); summary=parsed['sequence_summary']
                for i,e in enumerate(parsed['events']):
                    try:
                        p=convert_event(e,c,plan,f"{c['candidate_id']}_event_{i:02d}");props.append(p)
                        details.append(dict(proposal_id=p['proposal_id'],evidence_description=e['evidence_description'],
                                            boundary_basis='local_model_frame_indices_not_human_verified'))
                    except (ValueError,TypeError,KeyError) as ex: errors.append(str(ex))
                status=('partially_invalid' if props else 'invalid') if errors else ('ok' if props else 'empty')
            except (ValueError,TypeError,KeyError) as ex: errors.append(str(ex));status='invalid'
            row=dict(candidate_id=c['candidate_id'],video=c['video'],group_id=c['group_id'],start_sec=c['start_sec'],end_sec=c['end_sec'],
                     raw_response=raw,sequence_summary=summary,proposals=props,event_details=details,errors=errors,
                     inference_status=status,elapsed_sec=round(time.monotonic()-t0,3))
            target=records/(c['candidate_id']+'.json');temp=target.with_suffix('.tmp');save(temp,row);temp.replace(target)
            completed+=1
            print(json.dumps(dict(candidate=c['candidate_id'],status=status,events=len(props),seconds=row['elapsed_sec'])),flush=True)
    rows=[json.loads(p.read_text()) for p in sorted(records.glob('*.json'))]
    jsonl(run/'proposals.jsonl',[p for r in rows for p in r['proposals']])
    jsonl(run/'event_details.jsonl',[d for r in rows for d in r['event_details']])
    jsonl(run/'window_analysis.jsonl',[{k:v for k,v in r.items() if k not in {'proposals','event_details','raw_response'}} for r in rows])
    jsonl(run/'raw_responses.jsonl',[{k:v for k,v in r.items() if k not in {'proposals','event_details','sequence_summary'}} for r in rows])
    jsonl(run/'rejected.jsonl',[dict(candidate_id=r['candidate_id'],errors=r['errors']) for r in rows if r['errors']])
    meta=dict(plan_sha256=plan_hash,completed_windows=len(rows),total_windows=len(plan['candidates']),
              completed= len(rows)==len(plan['candidates']),inference_seconds=sum(r['elapsed_sec'] for r in rows),
              last_batch_wall_seconds=round(time.monotonic()-start,3),model_id=plan['model_lock']['model_id'],model_revision=plan['model_lock']['model_revision'],
              code_sha256={p.name:digest(p) for p in code_dir.glob('*.py')},prompt_sha256=hashlib.sha256(prompt_text.encode()).hexdigest(),
              max_pixels_per_frame=256*28*28,max_new_tokens=1100,do_sample=False)
    save(state_path,meta)
    if meta['completed']:save(run/'inference.json',meta)
    print(json.dumps(meta),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd())
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('prepare');s.add_argument('--run',type=Path,required=True);s.add_argument('--videos',nargs='+')
    s=sub.add_parser('infer');s.add_argument('--run',type=Path,required=True);s.add_argument('--model-path',type=Path,required=True)
    s.add_argument('--max-windows',type=int,default=50);s.add_argument('--max-seconds',type=float,default=450)
    s.add_argument('--gpu-lock',type=Path,default=Path('/home/ubuntu/nuhack-training/.gpu-work.lock'))
    args=p.parse_args()
    if args.command=='infer' and (not 1<=args.max_windows<=400 or not 1<=args.max_seconds<=1800):raise ValueError('Bounded batch required')
    globals()[args.command](args)


if __name__=='__main__':main()
