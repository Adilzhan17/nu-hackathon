"""Audit real local autolabel artifacts without promoting predictions into labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import av
from PIL import Image
from .dataset import registry,snapshot,digest,read_jsonl,csv_rows
from .pipeline import inside,save
from .schema import validate_proposal
from .normalize_deep import normalize_record
from .video_bags import VideoBagDataset
from .inspect_incidents import decode


def check(condition,message):
    if not condition:raise ValueError(message)


def video_check(path,expected_hash=None):
    if expected_hash:check(digest(path)==expected_hash,'Exported video changed')
    with av.open(str(path)) as c:
        check(len(c.streams.video)==1 and len(c.streams.audio)==0,'Export must be video-only')
        previous=-1.;count=0
        for f in c.decode(video=0):
            check(f.pts is not None and f.time>previous,'Nonmonotonic exported PTS')
            previous=f.time;count+=1
    check(count>=2,'Empty/short export')
    return count


def audit(root,run,report,objects,bags,allow_pending_state_checks=False):
    root=Path(root).resolve();run=inside(run,root/'data/autolabel');report=inside(report,root/'data/autolabel');objects=inside(objects,root/'data/autolabel');bags=inside(bags,root/'data/autolabel')
    reg,_,_=registry(root);protected=snapshot(root);plan=json.loads((run/'plan.json').read_text());meta=json.loads((run/'inference.json').read_text())
    check(plan['protected_inputs']==protected and plan['registry']==reg,'Frozen inputs/registry changed')
    check(meta['completed'] and meta['completed_windows']==len(plan['candidates']),'Incomplete inference')
    check(meta['plan_sha256']==digest(run/'plan.json'),'Plan changed after inference')
    for name,sha in meta['code_sha256'].items():check(digest(run/'inference_code'/name)==sha,'Inference code snapshot changed')
    candidates={c['candidate_id']:c for c in plan['candidates']};checked_frames=set();normalized=[]
    for c in plan['candidates']:
        check(reg[c['video']]['split']=='train' and reg[c['video']]['group_id']==c['group_id'],'Candidate leakage')
        check(0<=c['start_sec']<c['end_sec']<=plan['durations'][c['group_id']],'Invalid candidate bounds')
        timestamps=[f['timestamp'] for f in c['frames']]
        check(len(timestamps)>=2 and timestamps==sorted(set(timestamps)),'Invalid evidence sequence')
        for f in c['frames']:
            check(c['start_sec']-1e-6<=f['timestamp']<=c['end_sec']+1e-6,'Frame outside window')
            path=inside(run/f['path'],run/'frames')
            if path not in checked_frames:
                check(digest(path)==f['sha256'],'Frame changed');checked_frames.add(path)
        record=json.loads((run/'inference_records'/(c['candidate_id']+'.json')).read_text())
        check(record['candidate_id']==c['candidate_id'] and record['group_id']==c['group_id'],'Record provenance mismatch')
        normalized.append(normalize_record(record,c,plan))
    ps=read_jsonl(report/'proposals.jsonl');by_id={p['proposal_id']:p for p in ps}
    check(len(ps)==len(by_id),'Repeated proposal IDs')
    expected={p['proposal_id']:p for r in normalized for p in r['proposals']}
    check(set(expected)==set(by_id),'Report dropped/invented normalized proposals')
    for p in ps:
        check(all(p[k]==v for k,v in expected[p['proposal_id']].items()),'Proposal differs from raw local model output')
        c=candidates[p['candidate_id']];validate_proposal(p,reg,plan['durations'][p['group_id']],[f['timestamp'] for f in c['frames']])
    summary=json.loads((report/'summary.json').read_text());check(summary['protected_inputs']==protected,'Report input hash mismatch')
    check(summary['proposals']==len(ps),'Wrong proposal count')
    moments=read_jsonl(report/'moments.jsonl')
    for m in moments:
        check(m['review_status']=='proposed' and m['eligible_for_action_training'] is False,'Unapproved pseudo-label promoted')
        check(all(pid in by_id and by_id[pid]['group_id']==m['group_id'] for pid in m['proposal_ids']),'Moment provenance mismatch')
    queues=read_jsonl(report/'review_queue.jsonl')
    for q in queues:check(0<=q['start_sec']<q['end_sec']<=plan['durations'][q['group_id']] and q['end_sec']-q['start_sec']<=12.000001,'Invalid review bounds')
    check(len(queues)<=64,'Review budget exceeded')
    for r in read_jsonl(report/'incident_candidates.jsonl'):check(r['interval_target'] is None and r['label_scope']=='whole_source_video','Positive video label leaked to interval')
    for r in read_jsonl(report/'rule_assessments.jsonl'):check(r['legal_violation_target'] is None and r['eligible_for_violation_training'] is False,'Rule became legal ground truth')
    checks=json.loads((run/'incident_check_plan.json').read_text())['checks']
    state_path=run/'incident_check_summary.json'
    if not state_path.exists() and allow_pending_state_checks:
        state_summary=dict(completed=False,checks=0,inference_seconds=0.)
    else:state_summary=json.loads(state_path.read_text())
    if not allow_pending_state_checks:check(state_summary['completed'] and state_summary['checks']==len(checks),'State checks incomplete')
    if state_summary['completed']:
        state_audit=json.loads((run/'incident_check_run.json').read_text())
        check(state_audit['check_plan_sha256']==digest(run/'incident_check_plan.json'),'State plan changed')
        check(state_audit['source_code_sha256']==digest(run/'incident_check_code.py'),'State inference code snapshot changed')
        from .incident_report import collect,attach_assets
        states,patterns,_=collect(run,plan)
        attach_assets(states,patterns,run,report)
        check(states==read_jsonl(report/'visual_state_proposals.jsonl'),'State proposals differ from validated raw output')
        check(patterns==read_jsonl(report/'risk_pattern_candidates.jsonl'),'Risk patterns differ from evidence')
    for c in checks:
        check(reg[c['video']]['split']=='train','State check split leakage')
        record_path=run/'incident_check_records'/(c['check_id']+'.json')
        if not record_path.exists() and allow_pending_state_checks:continue
        record=json.loads(record_path.read_text())
        if record['parsed'] is not None:check(decode(record['raw_response'],c)==record['parsed'],'State record differs from raw inference')
    ds=VideoBagDataset(root,bags,frames=4,size=64)
    check(json.loads((bags/'summary.json').read_text())['protected_inputs']==protected,'Bag source hash mismatch')
    for r in ds.rows:check(digest(root/'data/raw/open'/r['video'])==r['source_sha256'],'Bag video changed')
    clip_rows=csv_rows(bags/'source_windows.csv')
    for r in clip_rows:
        check(r['split']=='train' and reg[r['video']]['group_id']==r['group_id'] and reg[r['video']]['split']=='train','Clip leakage')
        if int(r['organizer_video_target'])==1:check(r['interval_target']=='unknown' and r['eligible_for_supervised_interval_training']=='False','Positive anchor was promoted')
        video_check(inside(root/r['clip'],root/'data/autolabel'),r['clip_sha256'])
    object_rows=read_jsonl(objects/'object_review_manifest.jsonl')
    check(json.loads((objects/'summary.json').read_text())['protected_inputs']==protected,'Object input hash mismatch')
    check(len({(r['group_id'],r['timestamp']) for r in object_rows})==len(object_rows),'Repeated object frames')
    for r in object_rows:
        check(reg[r['video']]['split']=='train' and reg[r['video']]['group_id']==r['group_id'],'Object train/val leakage')
        check(r['review_status']=='unreviewed' and r['objects'] is None and r['eligible_for_detector_training'] is False,'Unreviewed frames became detection truth')
        check(0<=r['timestamp']<=plan['durations'][r['group_id']],'Invalid object frame time')
        path=inside(objects/r['image'],objects/'images');check(digest(path)==r['image_sha256'],'Object image changed')
        with Image.open(path) as im:check(im.size==(r['width'],r['height']),'Object image dimensions changed');im.verify()
    timelines=read_jsonl(report/'video_timelines.jsonl')
    for v in timelines:video_check(report/v['preview'],v['preview_metadata']['clip_sha256'])
    pairs=read_jsonl(root/'data/autolabel/user_pairs_v1/user_video_text_pairs.jsonl')
    for r in pairs:check(reg[r['video']]['split']=='train' and reg[r['video']]['group_id']==r['group_id'],'User pair leakage')
    check(snapshot(root)==protected,'Protected files changed during validation')
    output=dict(passed=True,source_files=len(reg),train_groups=len({r['group_id'] for r in reg.values() if r['split']=='train'}),
                val_groups=len({r['group_id'] for r in reg.values() if r['split']=='val'}),analyzed_windows=len(candidates),
                evidence_jpegs_checked=len(checked_frames),proposals_checked=len(ps),moments_checked=len(moments),
                local_state_checks=state_summary['checks'],pending_state_checks=len(checks)-state_summary['checks'],review_tasks=len(queues),train_video_bags=len(ds),object_frames_checked=len(object_rows),
                user_pairs=len(pairs),decoded_clips=len(clip_rows),decoded_previews=len(timelines),
                positive_interval_ground_truth_created=0,human_object_boxes_created=0,protected_inputs=protected,
                inference_seconds=meta['inference_seconds']+state_summary['inference_seconds'],
                note='Validation establishes provenance, format and split isolation, not semantic model accuracy.')
    save(root/'data/autolabel/validation_report.json',output)
    print(json.dumps(output,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--run',type=Path,default=Path('data/autolabel/detailed_train'));p.add_argument('--report',type=Path,default=Path('data/autolabel/detailed_train_report'));p.add_argument('--objects',type=Path,default=Path('data/autolabel/objects_v1'));p.add_argument('--bags',type=Path,default=Path('data/autolabel/training_v2'));p.add_argument('--allow-pending-state-checks',action='store_true');a=p.parse_args();audit(a.root,a.run,a.report,a.objects,a.bags,a.allow_pending_state_checks)

if __name__=='__main__':main()
