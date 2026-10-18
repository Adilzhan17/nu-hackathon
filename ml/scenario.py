"""Local context → 4-second observations → temporal memory → visual recheck.

Every output remains a model proposal. Organizer targets and narration are never
sent to the VLM. Missing observations never establish a missing procedure.
"""
import argparse,hashlib,json,math,time
from pathlib import Path
from train_x3d import atomic_json

ACTIONS={'cash_received','cash_transferred','cash_put_on_surface','cash_removed_from_surface','cash_visible_on_surface',
 'document_received','document_transferred','document_checked','key_visible_on_surface','key_used',
 'safe_opened','safe_closed','safe_visibly_open','safe_visibly_closed','employee_leaves','employee_returns','employee_absent',
 'cash_counted_by_hand','cash_counter_used','banknote_detector_used','phone_used','drink_used','occluded','unclear'}

def valid_events(answer,focus,frames):
    """Reject unsupported frame references and timestamps; do not repair guesses."""
    accepted=[];errors=[];lookup={f['id']:f for f in frames}
    for raw in answer.get('events',[]) if isinstance(answer.get('events'),list) else []:
        try:
            a,b=raw['start_sec'],raw['end_sec'];refs=raw['evidence_frames']
            if type(a) not in (int,float) or type(b) not in (int,float) or not math.isfinite(a+b):raise ValueError('time type')
            if not focus[0]<=a<b<=focus[1]+.01:raise ValueError('outside focus')
            if raw['action'] not in ACTIONS or raw.get('visibility') not in ('clear','partial','unclear'):raise ValueError('category')
            if not isinstance(refs,list) or not refs or any(ref not in lookup for ref in refs):raise ValueError('unknown evidence')
            if not any(a-.5<=lookup[ref]['time_sec']<=b+.5 for ref in refs):raise ValueError('evidence time mismatch')
            accepted.append(dict(start_sec=a,end_sec=b,action=raw['action'],actor=str(raw.get('actor','unknown'))[:40],
                object=str(raw.get('object','unknown'))[:40],recipient=str(raw.get('recipient','unknown'))[:40],
                visibility=raw['visibility'],description=str(raw.get('description',''))[:500],evidence_frames=refs,
                status='model_proposed',ground_truth=False))
        except (KeyError,TypeError,ValueError) as exc:errors.append(str(exc))
    return accepted,errors

def hypotheses(events):
    """Require positive visual observations on both sides; never infer omissions."""
    clear=[e for e in events if e['visibility']=='clear'];result=[]
    for leave in clear:
        if leave['action']!='employee_leaves':continue
        after=[e for e in clear if leave['end_sec']<=e['start_sec'] and e['end_sec']<=leave['end_sec']+12]
        if any(e['action']=='employee_returns' for e in after):continue
        absent=next((e for e in after if e['action']=='employee_absent'),None)
        if absent is None:continue
        for action,name in [('safe_visibly_open','possible_unattended_open_safe'),('cash_visible_on_surface','possible_unattended_cash'),('key_visible_on_surface','possible_unattended_keys')]:
            visible=next((e for e in after if e['action']==action),None)
            if visible:
                result.append(dict(kind=name,start_sec=leave['start_sec'],end_sec=max(absent['end_sec'],visible['end_sec']),
                    supporting_event_ids=[leave['id'],absent['id'],visible['id']],status='requires_visual_recheck',legal_violation=None))
    return result

def ledger(events):
    """Persistent last-seen records, explicitly not an assertion of current state."""
    last={};transfers=[]
    for event in events:
        record={k:event[k] for k in ['id','start_sec','end_sec','action','actor','object','recipient','visibility','description','status']}
        obj=event['object']
        if obj!='unknown':last[obj]=record
        if event['action'].startswith('employee_'):last['employee_location']=record
        if event['action'] in ('cash_received','cash_transferred','document_received','document_transferred'):
            transfers.append(record)
    return dict(meaning='Last observed proposals only. No assumption that an object remains there. Different cash bundles may be indistinguishable.',
                last_observed=last,transfer_history=transfers[-16:])

def sample(path,stamps,folder,run):
    import av
    frames=[];images=[];i=0
    with av.open(str(path)) as video:
        stream=video.streams.video[0];stream.thread_type='AUTO'
        video.seek(max(0,int(stamps[0]/float(stream.time_base))),stream=stream,backward=True)
        for f in video.decode(stream):
            if f.time is None:continue
            if i>=len(stamps):break
            if f.time<stamps[i]:continue
            image=f.to_image();image.thumbnail((768,768))
            name=f'f{len(frames):03d}';dest=folder/(name+'.jpg');dest.parent.mkdir(parents=True,exist_ok=True);image.save(dest,quality=90)
            frames.append(dict(id=name,time_sec=round(f.time,4),file=str(dest.relative_to(run)),sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
            images.append(image)
            while i<len(stamps) and f.time>=stamps[i]:i+=1
    return frames,images

def parse(text):
    text=text.strip();start=text.find('{');end=text.rfind('}')
    if start<0 or end<start:raise ValueError('No JSON object')
    value=json.loads(text[start:end+1])
    if not isinstance(value,dict):raise ValueError('Object required')
    return value

def main():
    import numpy as np,torch
    from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=Path('runs/scenario_v1'));args=parser.parse_args();run=args.run
    plan=json.loads((run/'plan.json').read_text());assert hashlib.sha256(Path('data/training/manifest.csv').read_bytes()).hexdigest()==plan['manifest_sha256']
    start=time.time();state=dict(status='running',stage='loading',videos_total=len(plan['videos']),videos_done=0,windows_done=0,events=0,updated_at=time.time())
    def emit(**kw):state.update(kw,updated_at=time.time(),elapsed_sec=time.time()-start);atomic_json(run/'status.json',state)
    def budget():
        if time.time()-start>plan['max_seconds']:raise TimeoutError('Scenario pilot time budget reached')
    results=[];emit()
    try:
        torch.set_num_threads(8);folder=Path('artifacts/assets/scenario_qwen3')
        model=Qwen3VLForConditionalGeneration.from_pretrained(folder,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
        params=sum(p.numel() for p in model.parameters());assert params<=3_000_000_000
        processor=AutoProcessor.from_pretrained(folder,local_files_only=True,min_pixels=128*32*32,max_pixels=384*32*32)
        emit(parameters=params,model='Qwen3-VL-2B-Instruct',precision='BF16',stage='survey')
        def ask(prompt,frames,images,max_tokens=640):
            budget();content=[dict(type='text',text=prompt)]
            for f,im in zip(frames,images):
                content.extend([dict(type='text',text=f"SOURCE FRAME {f['id']} at {f['time_sec']:.3f} seconds."),dict(type='image',image=im)])
            text=processor.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
            inputs=processor(text=[text],images=images,return_tensors='pt').to('cuda')
            began=time.time()
            with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=max_tokens,do_sample=False)
            raw=processor.batch_decode(output[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
            try:return parse(raw),dict(seconds=time.time()-began,input_tokens=inputs.input_ids.shape[1],output_tokens=output.shape[1]-inputs.input_ids.shape[1])
            except (ValueError,json.JSONDecodeError):return dict(parse_error=True,raw=raw),dict(seconds=time.time()-began)
        for vi,video in enumerate(plan['videos']):
            budget();path=Path('data/raw/open')/video['video'];assert hashlib.sha256(path.read_bytes()).hexdigest()==video['sha256']
            vid='video_'+video['sha256'][:16];duration=video['duration'];out=run/vid;out.mkdir(exist_ok=True)
            result=dict(video=video['video'],duration=duration,events=[],windows=[],hypotheses=[],ground_truth_created=0,complete=False);results.append(result)
            emit(stage='survey',current_video=vid)
            stamps=np.linspace(0,max(0,duration-.1),min(32,max(2,math.ceil(duration)))).tolist();frames,images=sample(path,stamps,out/'survey',run)
            survey,telemetry=ask('Analyze these ordered CCTV frames as context for later close inspection. Do not decide whether there is a violation. Identify observable roles and locations; do not identify people or read personal information. Distinguish cash, paper, identity card and unknown object. Return JSON only: {"scene_summary":"short factual overview", "roles":[{"id":"person_1","role":"employee|customer|unknown","location":"visible location"}], "zones":[{"id":"zone_1","type":"work_surface|safe|door|unknown","description":"location"}], "uncertainties":["..."]}. Never infer an unseen action.',frames,images)
            result['survey']=survey;atomic_json(out/'survey.json',dict(answer=survey,frames=frames,telemetry=telemetry));memory=[]
            for wi,a in enumerate(np.arange(0,duration,4)):
                b=min(duration,float(a)+4);a=float(a);budget();emit(stage='focused_observation',focus_start=a,focus_end=b,current_window=wi+1)
                stamps=np.arange(a,max(a+.01,b-.05),.25).tolist();frames,images=sample(path,stamps,out/f'window_{wi:03d}',run)
                prompt=('Inspect only the current 4-second focus interval. The images are ordered and have source timestamps. Describe visible actions and states, not legal conclusions. '
                    'A missing observation is UNKNOWN, never proof that a procedure did not happen. Prior context and memory are fallible model proposals: independently verify current frames; do not repeat a prior guess. '
                    'Track visible roles and objects across time: who hands what to whom, where cash is placed, safe door state, and departure/return of the employee. '
                    'If cash vs paper or keys are not visually distinguishable, use unclear, not a specific object. '
                    f'FOCUS [{a:.3f}, {b:.3f}] seconds. Context: '+json.dumps(survey,ensure_ascii=False)[:2500]+'. Persistent last-seen ledger (not current facts): '+json.dumps(ledger(result['events']),ensure_ascii=False)[:7000]+'. Recent proposed observations: '+json.dumps(memory[-6:],ensure_ascii=False)[:3500]+
                    '. Allowed actions: '+','.join(sorted(ACTIONS))+
                    '. Return JSON only, at most 4 events: {"events":[{"start_sec":0.0,"end_sec":1.0,"action":"allowed action","actor":"person_1 or unknown","object":"cash|document|key|safe|unknown","recipient":"person_2 or surface or unknown","visibility":"clear|partial|unclear","description":"brief visible evidence","evidence_frames":["f000"]}],"uncertainty":"brief"}. Event times must lie inside the focus interval and evidence IDs must refer to supplied current images.')
                answer,telemetry=ask(prompt,frames,images);events,errors=valid_events(answer,(a,b),frames)
                for ei,event in enumerate(events):
                    event['id']=f'{vid}_w{wi:03d}_e{ei}';event['evidence']=[f for f in frames if f['id'] in event['evidence_frames']]
                    memory.append({k:event[k] for k in ['id','start_sec','end_sec','action','actor','object','recipient','visibility','description','status']})
                result['events'].extend(events);result['windows'].append(dict(start_sec=a,end_sec=b,events=len(events),rejected=len(errors),parse_error=bool(answer.get('parse_error'))))
                atomic_json(out/f'window_{wi:03d}.json',dict(answer=answer,accepted=events,rejections=errors,frames=frames,telemetry=telemetry,focus=[a,b]))
                atomic_json(out/'timeline.json',result)
                emit(windows_done=state['windows_done']+1,events=state['events']+len(events))
            candidates=hypotheses(result['events']);emit(stage='visual_recheck',candidates=len(candidates))
            for ci,candidate in enumerate(candidates[:4]):
                a=max(0,candidate['start_sec']-4);b=min(duration,candidate['end_sec']+4)
                stamps=np.linspace(a,max(a,b-.1),min(32,max(4,math.ceil((b-a)*2)))).tolist();frames,images=sample(path,stamps,out/f'recheck_{ci:02d}',run)
                answer,telemetry=ask('Independently inspect the complete ordered sequence. A previous stage proposed '+candidate['kind']+'. Treat that proposal as untrusted. Look for evidence that contradicts it, unclear object identity, occlusion, another employee, or a safe actually closed. Do not infer missing procedures. Return JSON only: {"assessment":"supported|contradicted|insufficient_data","visible_description":"what is actually visible","reason":"why the hypothesis is supported or not","evidence_frames":["f000"]}. This is a visual review proposal, never a legal finding.',frames,images,512)
                refs=answer.get('evidence_frames',[])
                supported=answer.get('assessment')=='supported' and isinstance(refs,list) and bool(refs) and all(r in {f['id'] for f in frames} for r in refs)
                candidate.update(status='model_supported' if supported else 'unconfirmed',verification=answer,evidence=[f for f in frames if f['id'] in refs],ground_truth=False)
                atomic_json(out/f'recheck_{ci:02d}.json',dict(hypothesis=candidate,frames=frames,telemetry=telemetry))
            result['hypotheses']=candidates;result['memory']=ledger(result['events']);result['complete']=True;atomic_json(out/'timeline.json',result);emit(videos_done=vi+1)
        atomic_json(run/'report.json',dict(model='Qwen3-VL-2B-Instruct',parameters=params,manifest_sha256=plan['manifest_sha256'],videos=results,ground_truth_created=0,
            evaluation_scope='3 train videos only; no accuracy or F1 claim',limitations=['All observations and rechecks remain unverified model proposals.','Fixed camera-role identities can be wrong.','No inference of absent identity/currency checks from missing observations.'],elapsed_sec=time.time()-start))
        emit(status='completed',stage='completed')
    except Exception as exc:
        atomic_json(run/'partial_report.json',dict(videos=results,ground_truth_created=0,error=str(exc)))
        emit(status='failed',stage='failed',error=str(exc));raise

if __name__=='__main__':main()
