"""Bounded T4 Qwen worker for one uploaded video. No labels or transcripts read."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

from ml.autolabel.sampling import frames
from ml.narration_training.organizer.analyze import SAFETY,SCHEMA,save_frames,transitions,gpu_lease
from ml.narration_training.organizer.context import ACTIONS,PROPERTIES,general_rules_text,parse_answer,validate_observations,memory
from .rules import guarded_assessments,whole_video_verdict
from ml.narration_training.prepare import write_json

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(job,max_seconds):
    import torch
    from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
    import transformers.integrations.sdpa_attention as sdpa
    from torch.nn.attention import sdpa_kernel,SDPBackend
    root=Path.cwd();job=job.resolve()
    if not job.is_relative_to(root/'runs/context_pipeline'):raise ValueError('Invalid private job root')
    plan=json.loads((job/'input.json').read_text())
    if set(plan)!={'duration_sec','sha256','candidate_windows','detector_observations'}:raise ValueError('Non-blind input fields')
    if sha(job/'source')!=plan['sha256']:raise ValueError('Changed source')
    duration=plan['duration_sec']
    if not 0<duration<=60:raise ValueError('Pilot accepts at most 60 seconds')
    model_path=root/'artifacts/assets/scenario_qwen3'
    provenance=json.loads((model_path/'source.json').read_text())
    if provenance['revision']!='89644892e4d85e24eaac8bacfd4f463576704203':raise ValueError('Wrong Qwen revision')
    for name,digest in provenance['files'].items():
        if sha(model_path/name)!=digest:raise ValueError('Changed Qwen weights')
    started=time.monotonic();torch.set_num_threads(4)
    observations=[];rejected=[];memories=[];focus=[];stages=[]
    def emit(stage):
        write_json(job/'progress.json',dict(stage=stage,updated_at=time.time()))
        print(json.dumps(dict(stage=stage,elapsed=round(time.monotonic()-started,2))),flush=True)
    with gpu_lease(root/'.gpu-work.lock'):
        emit('loading_qwen')
        original=sdpa.use_gqa_in_sdpa;sdpa.use_gqa_in_sdpa=lambda attention_mask,key:False
        try:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                model=Qwen3VLForConditionalGeneration.from_pretrained(model_path,local_files_only=True,dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
                assert sum(p.numel() for p in model.parameters())<=3_000_000_000
                processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,min_pixels=64*32*32,max_pixels=128*32*32)
                common=SAFETY+'\nGeneral rules:\n'+general_rules_text()+'\nAllowed actions: '+','.join(ACTIONS)+'\nAllowed state properties: '+','.join(PROPERTIES)+'\n'+SCHEMA
                def ask(stage,sampled,prompt,limit):
                    if time.monotonic()-started>max_seconds-15:raise TimeoutError('Qwen budget exhausted')
                    emit(stage);refs=save_frames(sampled,job/(stage+'_frames'),job)
                    content=[dict(type='text',text=prompt)]
                    for (_,im),ref in zip(sampled,refs):content.extend([dict(type='text',text=f"Frame {ref['id']} at {ref['timestamp']:.5f} seconds"),dict(type='image',image=im)])
                    rendered=processor.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
                    inputs=processor(text=[rendered],images=[im for _,im in sampled],return_tensors='pt').to('cuda')
                    with torch.inference_mode():
                        output=model.generate(**inputs,max_new_tokens=limit,do_sample=False,repetition_penalty=1.15,max_time=min(55,max_seconds-(time.monotonic()-started)-5))
                    raw=processor.batch_decode(output[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
                    error=None
                    try:answer=parse_answer(raw)
                    except (ValueError,TypeError) as exc:answer={};error=str(exc)
                    write_json(job/(stage+'.json'),dict(prompt=prompt,frames=refs,answer=answer,raw_response=raw,parse_error=error,model_id=provenance['repo'],model_revision=provenance['revision']))
                    stages.append(dict(stage=stage,parse_error=error,frames=len(refs)))
                    return answer,refs
                coarse=list(frames(job/'source',0,duration-1e-5,1,max_side=768))
                if len(coarse)<2:raise ValueError('Insufficient video frames')
                compact='\nReturn at most TWO events and TWO states, compact JSON; each description under 15 words. Include evidence_frames. Do not repeat history as current evidence.'
                answer,refs=ask('survey',coarse,common+'\nSurvey the whole ordered sequence; inspect arrivals, departures and object transfers.'+compact,1100)
                obs,errors=validate_observations(answer,refs,'upload_'+plan['sha256'][:16],'survey',(0,duration));observations+=obs;rejected+=errors
                focus=transitions(observations,coarse,duration,2)
                # VideoMAE adds a candidate only; it supplies no action/violation label to Qwen.
                candidates=sorted(plan['candidate_windows'],key=lambda r:r['score'],reverse=True)
                if candidates:
                    center=(candidates[0]['start_sec']+candidates[0]['end_sec'])/2
                    a=max(0,min(center-2,duration-4));b=min(duration,a+4)
                    if not any(abs((w['start_sec']+w['end_sec'])/2-center)<2 for w in focus):
                        focus=focus[:1]+[dict(start_sec=a,end_sec=b,context_start_sec=max(0,a-2),context_end_sec=min(duration,b+2),reason='videomae_candidate_unverified')]
                focus.sort(key=lambda w:w['start_sec'])
                for i,w in enumerate(focus):
                    a,b=w['context_start_sec'],w['context_end_sec'];mem=memory(observations,a)
                    memories.append(dict(window=w,memory_before=mem))
                    detections=[f for f in plan['detector_observations'] if a<=f['time_sec']<=b][:12]
                    prompt=common+f'\nFocus {w["start_sec"]:.3f}–{w["end_sec"]:.3f}s with before/after context. Prior fallible memory: '+json.dumps(mem)+'. Unverified detector proposals (may be wrong; confirm in images; not proof of ownership): '+json.dumps(detections)+compact
                    sampled=list(frames(job/'source',a,b-1e-5,3,max_side=768))
                    answer,refs=ask(f'focus_{i:02d}',sampled,prompt,1000)
                    obs,errors=validate_observations(answer,refs,'upload_'+plan['sha256'][:16],f'focus_{i:02d}',(a,b));observations+=obs;rejected+=errors
                assessed_rules=guarded_assessments(observations)
                result=dict(schema_version=1,status='completed' if not any(s['parse_error'] for s in stages) else 'partial',
                    observations=observations,memory=memories,final_memory=memory(observations,duration+1e-6),
                    rules=assessed_rules,rejections=rejected,stages=stages,focus_windows=focus,
                    model_id=provenance['repo'],model_revision=provenance['revision'],dtype='float16',
                    review_status='proposed',ground_truth=False,whole_video_verdict=whole_video_verdict(assessed_rules),
                    sparse_sampling_is_not_absence_proof=True,organizer_answers_in_prompt=False,seconds=time.monotonic()-started)
                write_json(job/'result.json',result);emit('completed')
                del model,processor;gc.collect();torch.cuda.empty_cache()
        finally:sdpa.use_gqa_in_sdpa=original

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--max-seconds',type=int,default=190);a=p.parse_args()
    if not 20<=a.max_seconds<=220:raise ValueError('Bounded pilot only')
    run(a.job,a.max_seconds)
