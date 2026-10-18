"""Build detailed local timelines, deduplicated moments and prioritized human review."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import html
import json
import os
import shutil
from pathlib import Path

from .dataset import snapshot,registry,read_jsonl,digest
from .pipeline import save,jsonl,inside
from .review import reconcile
from .schema import validate_proposal
from .training_data import export_clip


def merged_moments(proposals):
    buckets=defaultdict(list)
    for p in proposals:buckets[(p['group_id'],p['observed_action'],tuple(sorted(p['objects'])))].append(p)
    moments=[]
    for key,rows in sorted(buckets.items()):
        clusters=[]
        for p in sorted(rows,key=lambda x:x['start_sec']):
            # Merge only overlapping observations of the same action/object, never a gap.
            if clusters and p['start_sec']<max(q['end_sec'] for q in clusters[-1]):clusters[-1].append(p)
            else:clusters.append([p])
        for cluster in clusters:
            moments.append(dict(moment_id=f'moment_{len(moments):04d}',video=cluster[0]['video'],group_id=key[0],split='train',
                 start_sec=min(p['start_sec'] for p in cluster),end_sec=max(p['end_sec'] for p in cluster),
                 observed_action=key[1],objects=list(key[2]),proposal_ids=[p['proposal_id'] for p in cluster],
                 evidence_timestamps=sorted({t for p in cluster for t in p['evidence_timestamps']}),
                 confidence_range=[min(p['confidence'] for p in cluster),max(p['confidence'] for p in cluster)],
                 uncertainty_reasons=sorted({p['uncertainty_reason'] for p in cluster if p['uncertainty_reason']}),
                 conflict_ids=sorted({cid for p in cluster for cid in p.get('conflict_ids',[])}),
                 review_status='proposed',eligible_for_action_training=False))
    return sorted(moments,key=lambda m:(m['video'],m['start_sec'],m['observed_action']))


def review_tasks(queue,proposals,conflicts=(),max_seconds=12.):
    by_id={p['proposal_id']:p for p in proposals};out=[]
    conflict_map={c['conflict_id']:c for c in conflicts}
    for q in queue:
        a=q['start_sec']
        while a<q['end_sec']-1e-6:
            b=min(q['end_sec'],a+max_seconds)
            ids=[pid for pid in q['proposal_ids'] if by_id[pid]['start_sec']<b and by_id[pid]['end_sec']>a]
            refs=[o for o in q['observations'] if o.get('start_sec') is not None and a-2<=float(o['start_sec'])<=b+2]
            conflict_ids=[cid for cid in q['conflict_ids'] if cid in conflict_map and conflict_map[cid]['start_sec']<b and conflict_map[cid]['end_sec']>a]
            reasons=[]
            if q.get('source_anchor'):reasons.append('organizer_incident_type_unresolved')
            if q.get('source_reference'):reasons.append('user_reference_not_recovered_by_model')
            if any(by_id[pid]['confidence']<.8 for pid in ids):reasons.append('low_model_self_rating')
            if any(by_id[pid]['uncertainty_reason'] for pid in ids):reasons.append('model_visibility_or_timing_uncertainty')
            reasons.extend(sorted({conflict_map[cid]['kind'] for cid in conflict_ids}))
            if not ids:reasons.append('unresolved_visual_interval')
            priority=100 if conflict_ids else (90 if q.get('source_anchor') else 85 if q.get('source_reference') else 40 if ids else 10)
            actions={by_id[p]['observed_action'] for p in ids}
            if actions & {'safe_opening','safe_closing','safe_locking','leave_workstation','cash_received','cash_transferred','stamp_handling','document_stamping'}:priority+=20
            out.append(dict(q,start_sec=a,end_sec=b,context_start_sec=max(0,a-2),
                            context_end_sec=min(q['context_end_sec'],b+2),proposal_ids=ids,
                            observations=refs,conflict_ids=conflict_ids,reasons=reasons,review_id=f'task_{len(out):04d}',priority=priority))
            a=b
    return sorted(out,key=lambda q:(-q['priority'],q['video'],q['start_sec']))


def select_review(tasks,limit=64):
    # First provide one high-priority item per video; then fill by priority.
    chosen=[];seen=set()
    for t in tasks:
        if t['group_id'] not in seen:
            chosen.append(t);seen.add(t['group_id'])
    ids={t['review_id'] for t in chosen}
    for t in tasks:
        if len(chosen)>=limit:break
        if t['review_id'] not in ids:chosen.append(t);ids.add(t['review_id'])
    return sorted(chosen,key=lambda t:(-t['priority'],t['video'],t['start_sec']))


def build(args):
    root=args.root.resolve();run=inside(args.run,root/'data/autolabel');out=inside(args.output,root/'data/autolabel')
    if out.exists():raise FileExistsError('Use a new report directory')
    plan=json.loads((run/'plan.json').read_text());meta=json.loads((run/'inference.json').read_text())
    if not meta['completed'] or meta['completed_windows']!=len(plan['candidates']):raise ValueError('Detailed inference incomplete')
    if snapshot(root)!=plan['protected_inputs'] or meta['plan_sha256']!=digest(run/'plan.json'):raise ValueError('Inputs changed')
    reg,_,_=registry(root)
    if reg!=plan['registry']:raise ValueError('Source registry changed')
    from .normalize_deep import normalize_record
    normalized=[]
    for c in plan['candidates']:
        record=json.loads((run/'inference_records'/(c['candidate_id']+'.json')).read_text())
        normalized.append(normalize_record(record,c,plan))
    props=[p for r in normalized for p in r['proposals']]
    by_c={c['candidate_id']:c for c in plan['candidates']}
    for p in props:
        c=by_c[p['candidate_id']]
        validate_proposal(p,reg,plan['durations'][p['group_id']],[f['timestamp'] for f in c['frames']])
    audit={r['candidate_id']:{k:v for k,v in r.items() if k not in {'proposals','event_details'}} for r in normalized}
    cs=[dict(c,inference_status=audit[c['candidate_id']]['inference_status']) for c in plan['candidates']]
    conflicts,queue=reconcile(props,cs,plan['observations'],plan['reviews'],plan['durations'],reg)
    moments=merged_moments(props)
    incidents=[]
    for group,source_rows in plan['reviews'].items():
        duration=plan['durations'][group]
        for source in source_rows:
            if source['source_verdict']!='нарушение' or not source.get('anchor_sec'):continue
            anchor=float(source['anchor_sec']);a=max(0.,anchor-4);b=min(duration,anchor+4)
            if any(i['group_id']==group and i['anchor_sec']==anchor for i in incidents):continue
            video=next(s['video'] for s in plan['scans'] if s['group_id']==group)
            ids=[p['proposal_id'] for p in props if p['group_id']==group and p['start_sec']<b and p['end_sec']>a]
            incidents.append(dict(incident_id=f'incident_{len(incidents):03d}',video=video,group_id=group,split='train',
                                  start_sec=a,end_sec=b,anchor_sec=anchor,video_target=1,label_origin='organizer',
                                  interval_target=None,interval_review_status='unknown',label_scope='whole_source_video',
                                  incident_type='unresolved',supporting_action_proposal_ids=ids,
                                  note='Original binary incident label; action proposals do not establish a specific legal violation.'))
            queue.append(dict(video=video,group_id=group,start_sec=a,end_sec=b,context_start_sec=max(0,a-2),
                              context_end_sec=min(duration,b+2),proposal_ids=ids,conflict_ids=[],reasons=['organizer_incident_type_unresolved'],
                              observations=plan['observations'][group],organizer_labels=source_rows,aliases=reg[video]['aliases'],source_anchor=True))
    from .reference_gaps import collect as collect_references
    references,reference_queue=collect_references(plan,props);queue.extend(reference_queue)
    from .queue_compact import compact
    tasks=compact(review_tasks(queue,props,conflicts));selected=select_review(tasks)
    from .incident_report import collect,attach_assets
    state_proposals,patterns,incident_meta=collect(run,plan)
    if incident_meta:attach_assets(state_proposals,patterns,run,out)
    state_normalizations=incident_meta.pop('normalized_records') if incident_meta else []
    from .rules import assessments,RULES
    checks=json.loads((run/'incident_check_plan.json').read_text())['checks'] if incident_meta else []
    rule_assessments=assessments(checks,patterns)
    out.mkdir(parents=True);jsonl(out/'proposals.jsonl',props);jsonl(out/'moments.jsonl',moments)
    jsonl(out/'state_check_normalizations.jsonl',state_normalizations)
    jsonl(out/'rule_assessments.jsonl',rule_assessments);save(out/'rule_definitions.json',RULES)
    jsonl(out/'visual_state_proposals.jsonl',state_proposals);jsonl(out/'risk_pattern_candidates.jsonl',patterns)
    jsonl(out/'window_analysis.jsonl',list(audit.values()))
    jsonl(out/'normalizations.jsonl',[dict(candidate_id=r['candidate_id'],normalizations=r['normalizations']) for r in normalized if r['normalizations']])
    jsonl(out/'event_details.jsonl',[d for r in normalized for d in r['event_details']])
    jsonl(out/'incident_candidates.jsonl',incidents);jsonl(out/'user_reference_alignment.jsonl',references)
    jsonl(out/'conflicts.jsonl',conflicts);jsonl(out/'review_queue_all.jsonl',tasks);jsonl(out/'review_queue.jsonl',selected)
    details={d['proposal_id']:d for r in normalized for d in r['event_details']}
    for p in props:
        sampled={f['timestamp']:f for f in by_c[p['candidate_id']]['frames']}
        details[p['proposal_id']]['evidence_frames']=[dict(timestamp=t,path=os.path.relpath(run/sampled[t]['path'],out),sha256=sampled[t]['sha256']) for t in p['evidence_timestamps']]
        details[p['proposal_id']]['object_visibility_status']='model_claim_pending_human_review'
        details[p['proposal_id']]['object_visibility']=[dict(object=o,visibility='unverified',source='local_vlm_object_claim',evidence_timestamps=p['evidence_timestamps'],uncertainty_reason=p['uncertainty_reason']) for o in p['objects']]
    jsonl(out/'event_details.jsonl',list(details.values()))
    videos=[]
    for s in plan['scans']:
        g=s['group_id'];rows=[c for c in cs if c['group_id']==g]
        successful=sorted((c['start_sec'],c['end_sec']) for c in rows if c['inference_status'] in {'ok','empty','partially_invalid'})
        intervals=[]
        for a,b in successful:
            if intervals and a<=intervals[-1][1]:intervals[-1][1]=max(intervals[-1][1],b)
            else:intervals.append([a,b])
        duration=plan['durations'][g]
        videos.append(dict(video=s['video'],group_id=g,duration_sec=duration,
             organizer_labels=plan['reviews'][g],user_observations=plan['observations'][g],
             user_references=[r for r in references if r['group_id']==g],
             visual_state_proposals=[s for s in state_proposals if s['group_id']==g],
             risk_pattern_candidates=[p for p in patterns if p['group_id']==g],
             rule_assessments=[r for r in rule_assessments if r['group_id']==g],
             moments=[m for m in moments if m['group_id']==g],
             windows=[audit[c['candidate_id']] for c in rows],
             parsed_window_coverage_fraction=min(1,sum(b-a for a,b in intervals)/duration),
             preview=f'previews/{g}.mp4'))
    def preview(v):
        cache=run/'preview_cache'/(v['group_id']+'.json')
        if cache.exists():
            cached=json.loads(cache.read_text());source=reg[v['video']]
            expected=dict(group_id=v['group_id'],source_sha256=source['sha256'],duration_sec=v['duration_sec'],fps=12,max_side=768)
            cached_video=cache.with_suffix('.mp4')
            if cached['provenance']!=expected or digest(cached_video)!=cached['clip']['clip_sha256']:raise ValueError('Preview cache mismatch')
            (out/v['preview']).parent.mkdir(exist_ok=True);shutil.copy2(cached_video,out/v['preview']);meta=cached['clip']
        else:meta=export_clip(root/'data/raw/open'/v['video'],out/v['preview'],0,v['duration_sec'],fps=12,max_side=768)
        return v['group_id'],meta
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i,(g,m) in enumerate(pool.map(preview,videos),1):
            next(v for v in videos if v['group_id']==g)['preview_metadata']=m
            print(json.dumps(dict(previews=i,total=len(videos))),flush=True)
    jsonl(out/'video_timelines.jsonl',videos)
    # Separate candidate manifest cannot be accidentally consumed as confirmed training labels.
    fields=['moment_id','video','group_id','split','start_sec','end_sec','observed_action','objects','review_status','eligible_for_action_training']
    with (out/'action_candidates.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for m in moments:w.writerow({k:json.dumps(m[k],ensure_ascii=False) if k=='objects' else m[k] for k in fields})
    summary=dict(videos=len(videos),independent_train_groups=len(videos),source_seconds=sum(plan['durations'].values()),
                 analyzed_windows=len(cs),proposals=len(props),deduplicated_moments=len(moments),
                 organizer_incident_windows=len(incidents),visual_state_proposals=len(state_proposals),risk_pattern_candidates=len(patterns),
                 incident_visual_checks=incident_meta,rule_assessment_counts=dict(Counter(r['assessment'] for r in rule_assessments)),
                 action_counts=dict(Counter(m['observed_action'] for m in moments)),user_reference_status_counts=dict(Counter(r['model_support_status'] for r in references)),
                 conflicts=len(conflicts),conflict_kinds=dict(Counter(c['kind'] for c in conflicts)),
                 review_tasks=len(tasks),priority_review_tasks=len(selected),
                 normalized_windows=sum(bool(r['normalizations']) for r in normalized),
                 empty_windows=sum(c['inference_status']=='empty' for c in cs),invalid_windows=sum(c['inference_status']=='invalid' for c in cs),
                 evidence_frames=sum(len(c['frames']) for c in cs),inference_seconds=meta['inference_seconds'],
                 model_id=meta['model_id'],model_revision=meta['model_revision'],
                 source_run=str(run.relative_to(root)),protected_inputs=plan['protected_inputs'],
                 ground_truth_created=0,validation_untouched=True,
                 note='Temporal coverage is not verified action recall. All VLM actions remain proposed; organizer labels are a separate training target.')
    save(out/'summary.json',summary)
    write_dashboard(out/'index.html',videos,selected,props,details,summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def write_dashboard(path,videos,queue,proposals,details,summary):
    data=json.dumps(dict(videos=videos,queue=queue,proposals=proposals,details=details,summary=summary),ensure_ascii=False).replace('<','\\u003c')
    template=Path(__file__).with_name('timeline.html').read_text()
    Path(path).write_text(template.replace('/*__DATA__*/',data),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    build(p.parse_args())


if __name__=='__main__':main()
