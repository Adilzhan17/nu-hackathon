"""Bounded local Qwen: whole-video survey -> contextual transitions -> matched no-history ablation."""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
from ml.autolabel.dataset import digest
from ml.autolabel.sampling import frames
from ml.narration_training.prepare import write_json,write_jsonl
from .context import ACTIONS,PROPERTIES,VERDICTS,RULES,general_rules_text,parse_answer,validate_observations,memory,rule_assessments

SAFETY='''Analyze ordered CCTV frames, not audio. Do not identify people or read personal information.
Only describe observable actions and states. No legal conclusions. Do not infer an action is absent
because it is missing from selected frames. A dark screen does not establish power off.
Do not judge banknote authenticity from appearance: distinguish counting from visibly using a checking device.
A small object is not automatically a key, a card is not automatically an ID, paper is not automatically cash.
Use unknown for uncertain object identity. Room door and safe door are different. Closed is not locked.
Use local person_1/person_2 identifiers and employee/customer/unknown roles, never names or biometric identity.
Cash track labels are tentative visual links, not legal ownership. When occluded, current state is unknown.
All outputs are unconfirmed proposals. Preserve every uncertainty; memory is fallible and may not be repeated as new evidence.
Only supplied image IDs can be evidence; do not cite memory as a current-frame observation.
Return concise JSON, no Markdown. Each event requires at least TWO supplied frame IDs.
State observations can use one supplied frame. Use short English codes and short factual descriptions.
'''
SCHEMA='''JSON structure:
{"sequence_summary":"brief visible sequence","events":[{"action":"allowed action","actor":"person_1|person_2|unknown","role":"employee|customer|unknown","object":"cash|document|key|safe|door|phone|stamp|bag|clothing|computer|unknown","track_id":"cash_1|safe_1|door_1|unknown","from_actor":"person_2|unknown","to_actor":"person_1|unknown","recipient":"person_1|person_2|desk|safe|unknown","visibility":"clear|partial|unclear","description":"visible change","uncertainty":"specific limitation or empty","evidence_frames":["f000","f001"]}],
"states":[{"property":"allowed property","value":"open|closed|desk|work_surface|safe_top|in_lock|in_safe|held|present|absent|unknown","actor":"person_1|unknown","role":"employee|customer|unknown","object":"cash|safe|door|key|document|stamp|phone|unknown","track_id":"cash_1|safe_1|door_1|unknown","visibility":"clear|partial|unclear","description":"visible state","uncertainty":"limitation or empty","evidence_frames":["f001"]}],
"unknowns":["remaining uncertainty"]}
For scene survey give at most 7 events and 5 states; for a focus window at most 4 events and 4 states.
Do not output rule verdicts in this visual-observation pass. The rules guide what to inspect, not the answer.
'''
ABLATION_VIDEOS={'videos/X-2.mov','videos/Х-10.mov','videos/Х-22.mov','videos/Х-38.mov'}

@contextlib.contextmanager
def gpu_lease(path):
    with open(path,'a') as f:
        deadline=time.monotonic()+30
        while True:
            try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()>deadline:raise TimeoutError('Shared GPU lock busy; no process stopped')
                time.sleep(.5)
        try:
            busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            others=[pid.strip() for pid in busy.splitlines() if pid.strip() and pid.strip()!=str(os.getpid())]
            if others:raise RuntimeError('Existing other GPU compute process detected; defer rather than run concurrently')
            yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def save_frames(sampled,folder,run):
    folder.mkdir(parents=True,exist_ok=True);result=[]
    for i,(t,im) in enumerate(sampled):
        path=folder/f'f{i:03d}.jpg';im.save(path,quality=92)
        result.append(dict(id=f'f{i:03d}',timestamp=t,path=str(path.relative_to(run)),sha256=digest(path)))
    return result

def transitions(observations,coarse,duration,maximum):
    candidates=[]
    preferred={'document_transfer','document_checked','cash_received','person_enters','employee_leaves','employee_returns','phone_put_away','banknote_detector_used','cash_stored','safe_closing','drink_used'}
    for o in observations:
        if o['kind']=='event':
            priority=3 if o['action'] in preferred else 2
            candidates.extend([(o['start_sec'],priority,'observed_transition'),(o['end_sec'],priority,'observed_transition')])
    changes=[]
    for (ta,ia),(tb,ib) in zip(coarse,coarse[1:]):
        da=np.asarray(ia.resize((64,48)),dtype=float);db=np.asarray(ib.resize((64,48)),dtype=float)
        changes.append((float(abs(da-db).mean()),tb))
    candidates += [(t,1+score/255,'visual_change') for score,t in sorted(changes,reverse=True)[:4]]
    candidates += [(0.,.5,'opening_context'),(duration,.5,'closing_context')]
    result=[]
    for center,priority,reason in sorted(candidates,key=lambda r:(-r[1],r[0])):
        a=max(0,min(center-2,max(0,duration-4)));b=min(duration,a+4)
        if any(abs((a+b)/2-(x['start_sec']+x['end_sec'])/2)<2.5 for x in result):continue
        result.append(dict(start_sec=a,end_sec=b,context_start_sec=max(0,a-2),context_end_sec=min(duration,b+2),reason=reason))
        if len(result)>=maximum:break
    return sorted(result,key=lambda r:r['start_sec'])

def run(args):
    import torch
    from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
    from transformers import __version__ as transformers_version
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    root=args.root.resolve();run=args.run.resolve();plan=json.loads((run/'plan.json').read_text())
    if not run.is_relative_to(root/'runs/narration_training'):raise ValueError('Invalid output root')
    allowed={'video','group_id','sha256','aliases','split','duration_sec'}
    if any(set(v)!=allowed for v in plan['videos']):raise ValueError('Visual plan contains non-blind fields')
    if digest(root/'data/training/manifest.csv')!=plan['manifest_sha256']:raise ValueError('Frozen split changed')
    source=json.loads((args.model/'source.json').read_text())
    if (source['repo'],source['revision'])!=(plan['model_id'],plan['model_revision']):raise ValueError('Model revision mismatch')
    if 'apache-2.0' not in (args.model/'README.md').read_text().lower():raise ValueError('License missing')
    for name,sha in source['files'].items():
        if digest(args.model/name)!=sha:raise ValueError('Model file mismatch')
    write_json(run/'model_provenance.json',dict(**source,license='Apache-2.0',transformers=transformers_version,
       torch=torch.__version__,claims_received_by_model=False,general_rules=RULES,prompt_version=3,
       model_card='https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct'))
    write_json(run/'general_rules.json',RULES)
    torch.set_num_threads(4);began=time.monotonic();completed=0
    for video in plan['videos']:
        if args.videos and video['video'] not in args.videos:continue
        folder=run/video['group_id'];folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():continue
        if time.monotonic()-began>args.max_seconds:break
        paths=[root/'data/raw/open'/name for name in [video['video'],*video['aliases']] if (root/'data/raw/open'/name).is_file()]
        if not paths or digest(paths[0])!=video['sha256']:raise ValueError('Video missing/changed')
        path=paths[0];duration=video['duration_sec'];video_started=time.monotonic()
        coarse=list(frames(path,0,duration-1e-5,plan['coarse_fps'],max_side=768))
        tail=list(frames(path,max(0,duration-.12),duration,30,max_side=768))
        if tail and tail[-1][0]>coarse[-1][0]:coarse.append(tail[-1])
        survey_frames=save_frames(coarse,folder/'survey_frames',run)
        write_json(folder/'sampling.json',dict(coarse_fps=plan['coarse_fps'],duration_sec=duration,
         sampled_frames=len(coarse),first_timestamp=coarse[0][0],last_timestamp=coarse[-1][0],
         full_video_temporally_surveyed=True,all_raw_frames_semantically_inspected=False,
         audio_decoded=False,actual_decoded_video=str(path.relative_to(root))))
        with gpu_lease(args.lock):
            model=Qwen3VLForConditionalGeneration.from_pretrained(args.model,local_files_only=True,trust_remote_code=False,
                     dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
            parameters=sum(p.numel() for p in model.parameters())
            if parameters>3_000_000_000:raise ValueError('Parameter cap')
            processor=AutoProcessor.from_pretrained(args.model,local_files_only=True,trust_remote_code=False,
                min_pixels=64*32*32,max_pixels=128*32*32)
            def ask(prompt,sampled,refs,output,limit):
                if output.exists():return json.loads(output.read_text())
                if time.monotonic()-began>args.max_seconds:raise TimeoutError('Bounded run budget reached; resumable files saved')
                content=[dict(type='text',text=prompt)]
                for (t,im),ref in zip(sampled,refs):
                    content.extend([dict(type='text',text=f"Frame {ref['id']} at {t:.6f} seconds"),dict(type='image',image=im)])
                rendered=processor.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
                inputs=processor(text=[rendered],images=[im for _,im in sampled],return_tensors='pt').to('cuda')
                start=time.monotonic()
                with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=limit,do_sample=False,repetition_penalty=1.15)
                raw=processor.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
                try:answer=parse_answer(raw);error=None
                except (ValueError,json.JSONDecodeError) as e:answer={};error=str(e)
                record=dict(prompt=prompt,answer=answer,raw_response=raw,parse_error=error,frames=refs,
                  input_tokens=inputs.input_ids.shape[1],output_tokens=generated.shape[1]-inputs.input_ids.shape[1],
                  seconds=time.monotonic()-start,model_id=source['repo'],model_revision=source['revision'])
                write_json(output,record);return record
            common=SAFETY+'\nGeneral scenario review rules (never a video-specific answer):\n'+general_rules_text()+\
                '\nAllowed actions: '+','.join(ACTIONS)+'\nAllowed state properties: '+','.join(PROPERTIES)+'\n'+SCHEMA
            survey=ask(common+'\nFIRST PASS: Survey the ENTIRE supplied video sequence. Describe chronological phases, people entering/leaving and changes in cash location. Do not invent observations between sampled frames.',coarse,survey_frames,folder/'survey.json',2200)
            observations,errors=validate_observations(survey['answer'],survey_frames,video['group_id'],'survey',(0,duration))
            focus=transitions(observations,coarse,duration,args.max_windows or plan['max_refinement_windows'])
            write_json(folder/'focus_plan.json',focus)
            contextual=[];independent=[];memories=[];model_findings=[]
            for wi,window in enumerate(focus):
                a,b=window['context_start_sec'],window['context_end_sec']
                sampled=list(frames(path,a,b-1e-5,plan['dense_fps'],max_side=768))
                refs=save_frames(sampled,folder/f'focus_{wi:02d}_frames',run)
                mem=memory(observations+contextual,a);memories.append(dict(window=wi,**window,memory_before=mem))
                compact='\nFINAL OUTPUT LIMIT: ONLY the primary transition. At most TWO events and TWO states. At most three evidence frame IDs per item. Keep descriptions under 15 words. Use compact one-line JSON, no indentation. Do not repeat any event or list historical events again. Finish the JSON after these items.'
                prompt=common+f"\nSECOND PASS: Focus {window['start_sec']:.3f}–{window['end_sec']:.3f}s; supplied frames include BEFORE/AFTER context {a:.3f}–{b:.3f}s. Relevant prior history (unconfirmed, current state unknown until reobserved): "+json.dumps(mem,ensure_ascii=False)+compact
                answer=ask(prompt,sampled,refs,folder/f'focus_{wi:02d}.json',1500)
                obs,rejected=validate_observations(answer['answer'],refs,video['group_id'],f'focus_{wi:02d}',(a,b));contextual+=obs;errors+=rejected
                model_findings.append(dict(window=wi,source='local_model_raw_rule_proposals',findings=answer['answer'].get('rule_findings',[])))
                if video['video'] in ABLATION_VIDEOS:
                    plain=common+f"\nSECOND PASS: Focus {window['start_sec']:.3f}–{window['end_sec']:.3f}s; supplied frames include BEFORE/AFTER context {a:.3f}–{b:.3f}s. No prior history is available."+compact
                    baseline=ask(plain,sampled,refs,folder/f'independent_{wi:02d}.json',1500)
                    baseobs,rejected=validate_observations(baseline['answer'],refs,video['group_id'],f'independent_{wi:02d}',(a,b));independent+=baseobs;errors+=rejected
                write_jsonl(folder/'visual_observations.jsonl',observations+contextual)
                write_jsonl(folder/'memory_trace.jsonl',memories)
                print(json.dumps(dict(video=video['video'],stage='refinement',window=wi+1,total_windows=len(focus),elapsed=round(time.monotonic()-began,1))),flush=True)
            # Separate rule reasoning AFTER observations; no organizer answer or human decision is available.
            available={f['timestamp']:f for o in observations+contextual for f in o['evidence']}
            evidence_sorted=sorted(available.items())
            selected=[evidence_sorted[i] for i in sorted(set(np.linspace(0,len(evidence_sorted)-1,min(12,len(evidence_sorted)),dtype=int).tolist()))] if evidence_sorted else []
            if selected:
                rule_refs=[dict(f,id=f'r{i:03d}') for i,(_,f) in enumerate(selected)]
                rule_images=[(t,Image.open(run/f['path']).convert('RGB')) for t,f in selected]
                compact_observations=[{k:o[k] for k in ['observation_id','kind','action','property','value','actor','role','object','track_id','start_sec','end_sec','visibility','uncertainty']} for o in observations+contextual]
                rule_prompt=SAFETY+'\nGeneral rules:\n'+general_rules_text()+\
                    '\nIndependent observations, all unconfirmed: '+json.dumps(compact_observations,ensure_ascii=False)+\
                    '\nAssess at most THREE relevant rules. If an absence-based claim lacks complete verified coverage, verdict MUST be insufficient_data. Evidence may link observation IDs and supplied frame IDs; never invent either. Do not infer banknote authenticity. Output compact JSON only: '+\
                    '{"rule_findings":[{"rule_id":"listed rule","verdict":"violation_supported|no_violation_observed|insufficient_data|not_applicable","triggering_event":"observation ID or unknown","evidence_frames":["r000"],"missing_evidence":["missing fact"],"explanation":"under 30 words"}]}'
                rule_answer=ask(rule_prompt,rule_images,rule_refs,folder/'rule_recheck.json',1200)
                model_findings.append(dict(window=None,phase='whole_scenario_rule_recheck',source='local_model_raw_rule_proposals',findings=rule_answer['answer'].get('rule_findings',[])))
            write_jsonl(folder/'visual_observations.jsonl',observations+contextual)
            write_jsonl(folder/'independent_observations.jsonl',independent)
            write_jsonl(folder/'memory_trace.jsonl',memories)
            write_jsonl(folder/'qwen_rule_proposals.jsonl',model_findings)
            write_jsonl(folder/'rule_assessments.jsonl',rule_assessments(observations+contextual))
            write_json(folder/'final_memory.json',memory(observations+contextual,duration+1e-6))
            write_json(folder/'validation_rejections.json',errors)
            write_json(folder/'complete.json',dict(video=video['video'],group_id=video['group_id'],split=video['split'],
                parameters=parameters,full_survey_frames=len(coarse),focus_windows=len(focus),
                observations=len(observations+contextual),independent_observations=len(independent),
                ablation=video['video'] in ABLATION_VIDEOS,seconds=time.monotonic()-video_started,
                parse_errors=sum(json.loads(p.read_text()).get('parse_error') is not None for p in [folder/'survey.json',*folder.glob('focus_??.json'),*folder.glob('rule_recheck.json')]),
                human_confirmed=False,whole_video_verdict=None))
            del model;del processor;torch.cuda.empty_cache()
        completed+=1
        print(json.dumps(dict(video=video['video'],stage='complete',completed_this_run=completed,elapsed=round(time.monotonic()-began,1))),flush=True)
        time.sleep(.25)
    write_json(run/'execution_status.json',dict(completed_total=len(list(run.glob('video_*/complete.json'))),
       total_videos=len(plan['videos']),elapsed_seconds=time.monotonic()-began,complete=len(list(run.glob('video_*/complete.json')))==len(plan['videos'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--run',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--videos',nargs='*');p.add_argument('--max-seconds',type=int,default=1200);p.add_argument('--max-windows',type=int,default=0)
    run(p.parse_args())
