"""Explicit human-decision import into a NEW action-training dataset. Never edits source labels."""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from .dataset import read_jsonl,registry,snapshot
from .pipeline import inside,save,jsonl
from .schema import ACTIONS,OBJECTS,number
from .training_data import export_clip


def validate_decision(d,moment,duration):
    required={'moment_id','group_id','video','decision','reviewer','start_sec','end_sec','observed_action','objects','note','reviewed_at'}
    if not isinstance(d,dict) or set(d)!=required:raise ValueError('Unexpected decision fields')
    if any(d[k]!=moment[k] for k in ('moment_id','group_id','video')):raise ValueError('Review provenance mismatch')
    if d['decision'] not in {'accept','reject','needs_context'} or not isinstance(d['reviewer'],str) or not d['reviewer'].strip():raise ValueError('Explicit named human decision required')
    if not all(number(d[k]) for k in ('start_sec','end_sec')) or not 0<=d['start_sec']<d['end_sec']<=duration:raise ValueError('Invalid reviewed interval')
    if d['observed_action'] not in ACTIONS or not isinstance(d['objects'],list) or any(o not in OBJECTS for o in d['objects']):raise ValueError('Invalid reviewed taxonomy')
    if not isinstance(d['note'],str) or not isinstance(d['reviewed_at'],str) or not d['reviewed_at']:raise ValueError('Review audit fields required')
    required_object='cash' if d['observed_action'].startswith('cash_') else 'safe' if d['observed_action'].startswith('safe_') else 'document' if d['observed_action'].startswith('document_') else 'key' if d['observed_action']=='key_handling' else 'stamp' if d['observed_action']=='stamp_handling' else None
    if d['decision']=='accept' and d['observed_action']=='unclear_action':
        raise ValueError('An unclear action must remain needs_context, not a strong training label')
    if d['decision']=='accept' and required_object and required_object not in d['objects']:
        raise ValueError('Reviewed action/object mismatch')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--report',type=Path,required=True);p.add_argument('--decisions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve();report=inside(args.report,root/'data/autolabel');dest=inside(args.output,root/'data/autolabel')
    if dest.exists():raise FileExistsError('New destination required')
    summary=json.loads((report/'summary.json').read_text())
    if snapshot(root)!=summary['protected_inputs']:raise ValueError('Source annotations changed since review')
    reg,_,_=registry(root);moments={m['moment_id']:m for m in read_jsonl(report/'moments.jsonl')}
    timelines={v['group_id']:v for v in read_jsonl(report/'video_timelines.jsonl')}
    decisions=read_jsonl(args.decisions);seen=set()
    windows={w['candidate_id']:w for w in read_jsonl(report/'window_analysis.jsonl')}
    references={r['reference_id']:r for r in read_jsonl(report/'user_reference_alignment.jsonl')}
    decision_references={}
    for d in decisions:
        reference_match=re.fullmatch(r'reference_(observation_\d{4})_\d{10,16}',str(d.get('moment_id','')))
        if reference_match:
            r=references.get(reference_match.group(1))
            if r is None:raise ValueError('Unknown user source reference')
            moments[d['moment_id']]=dict(moment_id=d['moment_id'],video=r['video'],group_id=r['group_id'])
            decision_references[d['moment_id']]=r['reference_id']
        match=re.fullmatch(r'manual_(candidate_\d{5})_\d{10,16}',str(d.get('moment_id','')))
        if match:
            w=windows.get(match.group(1))
            if w is None:raise ValueError('Unknown source interval for manual action')
            moments[d['moment_id']]=dict(moment_id=d['moment_id'],video=w['video'],group_id=w['group_id'])
    for d in decisions:
        if d.get('moment_id') not in moments or d['moment_id'] in seen:raise ValueError('Unknown/duplicate moment decision')
        m=moments[d['moment_id']];validate_decision(d,m,timelines[m['group_id']]['duration_sec']);seen.add(d['moment_id'])
        if reg[d['video']]['split']!='train':raise ValueError('No action training labels for val')
    dest.mkdir(parents=True);jsonl(dest/'human_decisions.jsonl',decisions)
    approved=[]
    for d in decisions:
        if d['decision']!='accept':continue
        # Keep long reviewed events as one annotation; sampler may crop inside it at training time.
        clip=f"clips/{d['moment_id']}.mp4"
        meta=export_clip(root/'data/raw/open'/d['video'],dest/clip,d['start_sec'],d['end_sec'])
        approved.append(dict(d,split='train',review_status='human_reviewed',eligible_for_action_training=True,
                             clip=clip,**meta,label_origin='explicit_human_decision',source_reference_id=decision_references.get(d['moment_id']),
                             organizer_video_verdict=timelines[d['group_id']]['organizer_labels'][0]['source_verdict'],
                             event_violation_target='not_annotated'))
    jsonl(dest/'reviewed_actions.jsonl',approved)
    fields=list(approved[0]) if approved else ['moment_id','video','group_id','split','observed_action','review_status','eligible_for_action_training']
    with (dest/'reviewed_actions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in approved:w.writerow(dict(row,objects=json.dumps(row['objects'],ensure_ascii=False)))
    save(dest/'summary.json',dict(review_decisions=len(decisions),approved_action_clips=len(approved),
                                 action_counts=dict(Counter(r['observed_action'] for r in approved)),
                                 source_report=str(report),protected_inputs=snapshot(root),validation_untouched=True))
    print(json.dumps(dict(approved_action_clips=len(approved),output=str(dest))))


if __name__=='__main__':main()
