"""Audit and reconcile entire causal chains, never vote or splice individual answers.
CPU only. Source runs stay immutable. All alternatives and their SHA256s are retained.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np
from ml.autolabel.dataset import digest,read_jsonl
from ml.autolabel.sampling import frames
from ml.narration_training.prepare import write_json,write_jsonl
from .analyze import SAFETY,SCHEMA,ABLATION_VIDEOS,transitions
from .context import ACTIONS,PROPERTIES,general_rules_text,parse_answer,validate_observations,memory,rule_assessments

CODE_HASH='e0f62b52e6a9b927ad52d7e51382517f754c6fb44bb89b2f27807b089b87702f'
CONTEXT_HASH='00cbff909e6074ceed3825022d16b085cd0179a0a6e172348d6720e4988c1367'
COMPACT='\nFINAL OUTPUT LIMIT: ONLY the primary transition. At most TWO events and TWO states. At most three evidence frame IDs per item. Keep descriptions under 15 words. Use compact one-line JSON, no indentation. Do not repeat any event or list historical events again. Finish the JSON after these items.'

def common_prompt():
    return SAFETY+'\nGeneral scenario review rules (never a video-specific answer):\n'+general_rules_text()+\
        '\nAllowed actions: '+','.join(ACTIONS)+'\nAllowed state properties: '+','.join(PROPERTIES)+'\n'+SCHEMA

def survey_prompt():
    return common_prompt()+'\nFIRST PASS: Survey the ENTIRE supplied video sequence. Describe chronological phases, people entering/leaving and changes in cash location. Do not invent observations between sampled frames.'

def focus_prompt(w,mem=None):
    prefix=common_prompt()+f"\nSECOND PASS: Focus {w['start_sec']:.3f}–{w['end_sec']:.3f}s; supplied frames include BEFORE/AFTER context {w['context_start_sec']:.3f}–{w['context_end_sec']:.3f}s. "
    return prefix+('No prior history is available.' if mem is None else 'Relevant prior history (unconfirmed, current state unknown until reobserved): '+json.dumps(mem,ensure_ascii=False))+COMPACT

def rule_input(observations):
    available={f['timestamp']:f for o in observations for f in o['evidence']}
    ordered=sorted(available.items())
    selected=[ordered[i] for i in sorted(set(np.linspace(0,len(ordered)-1,min(12,len(ordered)),dtype=int).tolist()))] if ordered else []
    refs=[dict(f,id=f'r{i:03d}') for i,(_,f) in enumerate(selected)]
    compact=[{k:o[k] for k in ['observation_id','kind','action','property','value','actor','role','object','track_id','start_sec','end_sec','visibility','uncertainty']} for o in observations]
    prompt=SAFETY+'\nGeneral rules:\n'+general_rules_text()+\
        '\nIndependent observations, all unconfirmed: '+json.dumps(compact,ensure_ascii=False)+\
        '\nAssess at most THREE relevant rules. If an absence-based claim lacks complete verified coverage, verdict MUST be insufficient_data. Evidence may link observation IDs and supplied frame IDs; never invent either. Do not infer banknote authenticity. Output compact JSON only: '+\
        '{"rule_findings":[{"rule_id":"listed rule","verdict":"violation_supported|no_violation_observed|insufficient_data|not_applicable","triggering_event":"observation ID or unknown","evidence_frames":["r000"],"missing_evidence":["missing fact"],"explanation":"under 30 words"}]}'
    return prompt,refs

def make_refs(sampled,group,stage):
    result=[]
    for i,(t,im) in enumerate(sampled):
        b=io.BytesIO();im.save(b,format='JPEG',quality=92)
        result.append(dict(id=f'f{i:03d}',timestamp=t,path=f'{group}/{stage}_frames/f{i:03d}.jpg',sha256=hashlib.sha256(b.getvalue()).hexdigest()))
    return result

def check_record(record,expected_prompt,expected_refs,blind,video,plan):
    issues=[]
    if record.get('prompt')!=expected_prompt:issues.append('prompt_or_prior_memory_mismatch')
    if (record.get('model_id'),record.get('model_revision'))!=(plan['model_id'],plan['model_revision']):issues.append('model_revision_mismatch')
    refs=record.get('frames',[])
    if refs!=expected_refs:issues.append('decoded_frames_or_timestamps_mismatch')
    for ref in refs:
        path=(blind/ref['path']).resolve()
        if not path.is_relative_to(blind.resolve()) or not path.is_file() or digest(path)!=ref['sha256']:
            issues.append('frame_bytes_mismatch');break
        if not 0<=ref['timestamp']<=video['duration_sec']:issues.append('timestamp_outside_video')
    if record.get('parse_error'):issues.append('response_parse_error')
    try:
        answer=parse_answer(record['raw_response'])
        if answer!=record['answer']:issues.append('raw_parsed_answer_mismatch')
    except (ValueError,KeyError,TypeError):issues.append('invalid_raw_response')
    return sorted(set(issues))

def audit_chain(root,blind,video,plan):
    root=Path(root);blind=Path(blind);folder=blind/video['group_id'];duration=video['duration_sec']
    result=dict(video=video['video'],group_id=video['group_id'],source=str(blind),issues=[],stages=[],
                reusable_stages=[],complete_marker=(folder/'complete.json').exists(),complete_valid=False)
    video_path=root/'data/raw/open'/video['video']
    if not video_path.exists() or digest(video_path)!=video['sha256']:
        result['issues'].append('source_video_identity_mismatch');return result
    if not (folder/'survey.json').exists():return result
    coarse=list(frames(video_path,0,duration-1e-5,plan['coarse_fps'],max_side=768))
    tail=list(frames(video_path,max(0,duration-.12),duration,30,max_side=768))
    if tail and tail[-1][0]>coarse[-1][0]:coarse.append(tail[-1])
    allrefs={};records={};chain_ok=True
    def stage(name,prompt,refs):
        nonlocal chain_ok
        path=folder/(name+'.json')
        if not path.exists():
            result['stages'].append(dict(stage=name,status='missing'));chain_ok=False;return None
        try:r=json.loads(path.read_text());issues=check_record(r,prompt,refs,blind,video,plan)
        except (ValueError,KeyError,TypeError):issues=['malformed_stage_file'];r=None
        if not chain_ok:issues.append('earlier_stage_unavailable_or_invalid')
        fingerprint=hashlib.sha256(json.dumps(dict(prompt=prompt,frames=refs,model_id=plan['model_id'],model_revision=plan['model_revision']),ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        result['stages'].append(dict(stage=name,status='invalid' if issues else 'valid',issues=issues,
            sha256=digest(path),input_sha256=fingerprint,answer_sha256=hashlib.sha256(json.dumps(r.get('answer'),sort_keys=True).encode()).hexdigest() if r else None))
        if issues:
            result['issues'] += [name+':'+e for e in issues];chain_ok=False;return None
        result['reusable_stages'].append(name);records[name]=r;allrefs[name]=refs;return r
    refs=make_refs(coarse,video['group_id'],'survey')
    survey=stage('survey',survey_prompt(),refs)
    if survey is None:return result
    observations,rejections=validate_observations(survey['answer'],refs,video['group_id'],'survey',(0,duration))
    focus=transitions(observations,coarse,duration,plan['max_refinement_windows'])
    result['expected_focus_windows']=len(focus)
    if (folder/'focus_plan.json').exists() and json.loads((folder/'focus_plan.json').read_text())!=focus:
        result['issues'].append('focus_plan_mismatch');return result
    contextual=[];independent=[];memories=[];findings=[]
    for wi,w in enumerate(focus):
        a,b=w['context_start_sec'],w['context_end_sec']
        refs=make_refs(list(frames(video_path,a,b-1e-5,plan['dense_fps'],max_side=768)),video['group_id'],f'focus_{wi:02d}')
        mem=memory(observations+contextual,a);memories.append(dict(window=wi,**w,memory_before=mem))
        r=stage(f'focus_{wi:02d}',focus_prompt(w,mem),refs)
        if r is None:return result
        obs,err=validate_observations(r['answer'],refs,video['group_id'],f'focus_{wi:02d}',(a,b));contextual+=obs;rejections+=err
        findings.append(dict(window=wi,source='local_model_raw_rule_proposals',findings=r['answer'].get('rule_findings',[])))
        if video['video'] in ABLATION_VIDEOS:
            r=stage(f'independent_{wi:02d}',focus_prompt(w),refs)
            if r is None:return result
            obs,err=validate_observations(r['answer'],refs,video['group_id'],f'independent_{wi:02d}',(a,b));independent+=obs;rejections+=err
    prompt,refs=rule_input(observations+contextual)
    if refs:
        r=stage('rule_recheck',prompt,refs)
        if r is None:return result
        findings.append(dict(window=None,phase='whole_scenario_rule_recheck',source='local_model_raw_rule_proposals',findings=r['answer'].get('rule_findings',[])))
    expected={'visual_observations.jsonl':observations+contextual,'independent_observations.jsonl':independent,
              'memory_trace.jsonl':memories,'qwen_rule_proposals.jsonl':findings,
              'rule_assessments.jsonl':rule_assessments(observations+contextual)}
    for name,rows in expected.items():
        if not (folder/name).exists() or read_jsonl(folder/name)!=rows:result['issues'].append('derived_layer_mismatch:'+name)
    for name,value in [('final_memory.json',memory(observations+contextual,duration+1e-6)),('validation_rejections.json',rejections)]:
        if not (folder/name).exists() or json.loads((folder/name).read_text())!=value:result['issues'].append('derived_layer_mismatch:'+name)
    if result['complete_marker']:
        c=json.loads((folder/'complete.json').read_text())
        for k,v in dict(video=video['video'],group_id=video['group_id'],split=video['split'],focus_windows=len(focus),observations=len(observations+contextual),independent_observations=len(independent),full_survey_frames=len(coarse),parse_errors=0).items():
            if c.get(k)!=v:result['issues'].append('completion_metadata_mismatch:'+k)
        if c.get('human_confirmed') is not False or c.get('whole_video_verdict') is not None:result['issues'].append('invalid_target_promotion')
    result['validation_rejections']=len(rejections)
    result['accepted_observations']=len(observations+contextual)
    result['complete_valid']=result['complete_marker'] and not result['issues']
    return result

def validate_plan(root,plan,allow_missing_aliases=False):
    if digest(Path(root)/'data/training/manifest.csv')!=plan['manifest_sha256']:raise ValueError('Frozen manifest changed')
    groups={};files={}
    for v in plan['videos']:
        if set(v)!={'video','group_id','sha256','aliases','split','duration_sec'}:raise ValueError('Non-blind plan')
        if v['group_id'] in groups:raise ValueError('Repeated source group')
        groups[v['group_id']]=v['split']
        for f in [v['video'],*v['aliases']]:
            if f in files and files[f]!=v['group_id']:raise ValueError('Alias crosses source groups')
            files[f]=v['group_id']
    import csv
    with (Path(root)/'data/training/manifest.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    byfile={r['video']:r for r in rows}
    splits={}
    for r in rows:splits.setdefault(r['group_id'],set()).add(r['split'])
    if any(len(v)>1 for v in splits.values()):raise ValueError('Source leakage in frozen manifest')
    for v in plan['videos']:
        for f in [v['video'],*v['aliases']]:
            r=byfile.get(f)
            if r:
                if (r['group_id'],r['sha256'],r['split'])!=(v['group_id'],v['sha256'],v['split']):raise ValueError('Plan / manifest mismatch')
            else:
                # Exact duplicate files are deliberately deduplicated in the frozen manifest.
                alias=Path(root)/'data/raw/open'/f
                if alias.is_file():
                    if digest(alias)!=v['sha256']:raise ValueError('Unverified duplicate alias')
                elif not allow_missing_aliases:raise ValueError('Unverified duplicate alias')
                if not any((r['group_id'],r['sha256'],r['split'])==(v['group_id'],v['sha256'],v['split']) for r in rows):raise ValueError('Alias source absent from manifest')


def verify_source(blind):
    for base in [blind,*blind.parents]:
        manifest=base/'snapshot_manifest.json'
        if manifest.exists():
            data=json.loads(manifest.read_text())
            if data.get('changed_during_copy'):raise ValueError('Inconsistent live snapshot')
            bad=[name for name,sha in data['files'].items() if not (base/name).is_file() or digest(base/name)!=sha]
            if bad:raise ValueError('Snapshot checksum mismatch: '+str(bad[:3]))
            code=[p for p in base.rglob('analyze.py') if digest(p)==CODE_HASH]
            context=[p for p in base.rglob('context.py') if digest(p)==CONTEXT_HASH]
            if not code or not context:raise ValueError('Snapshot lacks matching analyzer/context source')
            return dict(manifest=str(manifest),manifest_sha256=digest(manifest),verified_files=len(data['files']),
                        source_code_sha256=CODE_HASH,context_code_sha256=CONTEXT_HASH)
    reconciled=blind.parent/'reconciliation_summary.json'
    if reconciled.exists():
        data=json.loads(reconciled.read_text())
        if data['source_code_sha256']!=CODE_HASH or data['context_code_sha256']!=CONTEXT_HASH:raise ValueError('Reconciled code mismatch')
        return dict(local_reconciled_run=True,metadata_sha256=digest(reconciled),records_revalidated_individually=True)
    raise ValueError('Input must be a checksum-verified snapshot or an existing reconciled run')


def reconcile(root,source,inputs,output,audit_only=False):
    root=Path(root).resolve();source=Path(source).resolve();out=Path(output).resolve()
    if out.exists() or not out.is_relative_to(root/'runs/narration_training'):raise ValueError('New narration output required')
    if digest(root/'ml/narration_training/organizer/analyze.py')!=CODE_HASH or digest(root/'ml/narration_training/organizer/context.py')!=CONTEXT_HASH:raise ValueError('Audit implementation must match pinned analyzer/context')
    plan=json.loads((source/'blind/plan.json').read_text());validate_plan(root,plan)
    out.mkdir(parents=True);audits=[];tagged=[];integrity=[]
    for item in inputs:
        tag,path=item.split('=',1);blind=Path(path).resolve()
        if not tag.replace('_','').isalnum():raise ValueError('Invalid source tag')
        integrity.append(dict(source_tag=tag,**verify_source(blind)))
        if json.loads((blind/'plan.json').read_text())!=plan:raise ValueError('Plans differ; no automatic merge')
        tagged.append((tag,blind))
        for video in plan['videos']:
            if not (blind/video['group_id']).exists():continue
            a=audit_chain(root,blind,video,plan);a['source_tag']=tag;audits.append(a)
            print(json.dumps(dict(source=tag,video=video['video'],complete_valid=a['complete_valid'],reusable=len(a['reusable_stages']),issues=a['issues'])),flush=True)
    write_jsonl(out/'chain_audit.jsonl',audits);write_json(out/'source_integrity.json',integrity)
    comparison=[];selected=[]
    for video in plan['videos']:
        candidates=[a for a in audits if a['group_id']==video['group_id']]
        # Stable source order resolves ties; never use agreement with organizer/manual labels.
        ranked=sorted(candidates,key=lambda a:(a['complete_valid'],len(a['reusable_stages'])) ,reverse=True)
        chosen=ranked[0] if ranked else None
        selected.append(dict(video=video['video'],group_id=video['group_id'],source_tag=chosen['source_tag'] if chosen else None,
          complete_valid=bool(chosen and chosen['complete_valid']),reusable_stages=chosen['reusable_stages'] if chosen else [],
          reason='Prefer validated complete causal chain, then longest exact-input prefix, then stable input order; no label-based selection'))
        for i,a in enumerate(candidates):
            for b in candidates[i+1:]:
                common=set(x['stage'] for x in a['stages']) & set(x['stage'] for x in b['stages'])
                for name in sorted(common):
                    x=next(s for s in a['stages'] if s['stage']==name);y=next(s for s in b['stages'] if s['stage']==name)
                    if not x.get('sha256') or not y.get('sha256'):continue
                    if x.get('sha256')==y.get('sha256'):continue
                    comparison.append(dict(video=video['video'],group_id=video['group_id'],stage=name,sources=[a['source_tag'],b['source_tag']],
                      source_record_sha256=[x.get('sha256'),y.get('sha256')],same_inputs=x.get('input_sha256')==y.get('input_sha256'),
                      same_parsed_answer=x.get('answer_sha256')==y.get('answer_sha256'),review_status='preserved_alternative_not_ground_truth'))
    write_jsonl(out/'selection.jsonl',selected);write_jsonl(out/'response_variants.jsonl',comparison)
    summary=dict(total_videos=len(plan['videos']),fully_validated=sum(r['complete_valid'] for r in selected),
                 remaining=sum(not r['complete_valid'] for r in selected),response_variants=len(comparison),
                 invalid_chains=sum(bool(a['issues']) for a in audits),selection_uses_labels=False,
                 source_code_sha256=CODE_HASH,context_code_sha256=CONTEXT_HASH,source_inputs=[dict(tag=t,path=str(p)) for t,p in tagged])
    write_json(out/'reconciliation_summary.json',summary)
    if audit_only:return summary
    write_json(out/'source_identity_verification.json',dict(
      plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),
      manifest_sha256=plan['manifest_sha256'],all_aliases_verified_on_main=True,
      aliases=[dict(video=f,sha256=v['sha256'],group_id=v['group_id'],split=v['split']) for v in plan['videos'] for f in v['aliases']]))
    code=out/'code';code.mkdir()
    for p in (root/'ml/narration_training/organizer').glob('*.py'):
        shutil.copy2(p,code/p.name)
    write_json(out/'code_manifest.json',{p.name:digest(p) for p in code.iterdir() if p.is_file()})
    for p in source.iterdir():
        if p.name=='blind':continue
        if p.is_dir():shutil.copytree(p,out/p.name)
        else:shutil.copy2(p,out/p.name)
    (out/'blind').mkdir();write_json(out/'blind/plan.json',plan)
    for tag,blind in tagged:
        shutil.copytree(blind,out/'alternatives'/tag/'blind')
    for video,row in zip(plan['videos'],selected):
        if not row['source_tag']:continue
        src=dict(tagged)[row['source_tag']]/row['group_id'];dest=out/'blind'/row['group_id']
        shutil.copytree(src,dest)
        if not row['complete_valid']:
            # Keep only an exact coherent response prefix. Everything else survives under alternatives/.
            keep=set(row['reusable_stages'])
            for p in dest.iterdir():
                if p.is_file() and p.suffix in ['.json','.jsonl'] and p.stem not in keep:p.unlink()
        write_json(dest/'reconciled_origin.json',dict(row,source_directory=str(src),
          original_files={str(p.relative_to(src)):digest(p) for p in src.rglob('*') if p.is_file()},
          dtype_policy='Original response runtime preserved; FP16 resumption is recorded per new response. No numerical-equivalence claim.'))
    ledger=[]
    for tag,blind in tagged:
        runtime=json.loads((blind/'t4_migration_runtime.json').read_text()) if (blind/'t4_migration_runtime.json').exists() else None
        for path in sorted(blind.glob('video_*/*.json')):
            record=json.loads(path.read_text())
            if not isinstance(record,dict) or 'raw_response' not in record:continue
            sha=digest(path)
            matches=[dict(source_tag=t,path=str(other)) for t,b in tagged if t!=tag and (other:=b/path.relative_to(blind)).exists() and digest(other)==sha]
            ledger.append(dict(source_tag=tag,group_id=path.parent.name,stage=path.stem,source_path=str(path),sha256=sha,
              identical_artifacts_in=matches,runtime_at_snapshot=runtime,
              runtime_ref=record.get('runtime_ref'),
              historical_dtype='not_recorded_per_response' if not record.get('runtime_ref') else 'see_runtime_ref',
              interpretation='Run-level T4 metadata does not retroactively assign FP16 to cached pre-migration responses.'))
    write_jsonl(out/'response_provenance.jsonl',ledger)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--source',type=Path,required=True)
    p.add_argument('--input',action='append',required=True,help='tag=/path/to/blind');p.add_argument('--output',type=Path,required=True);p.add_argument('--audit-only',action='store_true')
    a=p.parse_args();print(json.dumps(reconcile(a.root,a.source,a.input,a.output,a.audit_only),ensure_ascii=False))
