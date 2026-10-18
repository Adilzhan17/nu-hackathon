"""CPU verification of report boundaries, five-layer separation and training eligibility."""
import argparse
from collections import Counter
import json
from pathlib import Path

from ml.autolabel.dataset import digest,read_jsonl
from .context import VERDICTS
from .dataset import scenario_loader
from .reconcile import validate_plan
from ml.narration_training.prepare import write_json


def verify(root,run,report):
    root=Path(root).resolve();run=Path(run).resolve();report=Path(report).resolve()
    plan=json.loads((run/'blind/plan.json').read_text());validate_plan(root,plan)
    mapping=read_jsonl(run/'number_mapping.jsonl');groups={r['group_id']:r for r in mapping}
    claims=read_jsonl(run/'source/organizer_claims_v1.jsonl');by_claim={c['claim_id']:c for c in claims}
    original=run/'source/original_message.txt';lines=original.read_text().splitlines()
    for c in claims:
        assert c['raw_source_line']==lines[c['source_line']-1]
        assert c['source_sha256']==digest(original)
        assert c['human_confirmed'] is False and c['binary_window_target'] is None
    scenarios=read_jsonl(report/'scenarios.jsonl');observations=0;completed=0;ablation_pairs=0
    for s in scenarios:
        group=groups[s['group_id']];assert s['split']==group['split'] and s['video']==group['video']
        assert s['whole_video_binary_target'] is None and s['organizer_claim_overrides_human'] is False
        obs=s['visual_evidence']['observations'];observations+=len(obs)
        completed+=s['visual_evidence']['full_video_survey_completed']
        for o in obs:
            assert 0<=o['start_sec']<=o['end_sec']<=group['duration_sec']
            assert o['human_confirmed'] is False and o['authority']=='visual_evidence'
            assert o['legal_cash_owner']=='unknown'
            ts=[]
            for f in o['evidence']:
                assert 0<=f['timestamp']<=group['duration_sec']
                assert digest(run/'blind'/f['path'])==f['sha256'];ts.append(f['timestamp'])
            assert min(ts)==o['start_sec'] and max(ts)==o['end_sec']
            if o['kind']=='event':assert len(set(ts))>=2
        ids={o['observation_id'] for o in obs}
        for r in s['rule_assessments']:
            assert r['verdict'] in VERDICTS and r['review_status']=='proposed'
            assert r['human_confirmed'] is False and r['whole_video_normal'] is False and r['legal_violation_target'] is None
            assert all(e['observation_id'] in ids for e in r['evidence'])
            if r['verdict']=='insufficient_data':assert r['missing_evidence']
        folder=run/'blind'/s['group_id']
        if (folder/'complete.json').exists():
            for independent in folder.glob('independent_??.json'):
                a=json.loads(independent.read_text());b=json.loads((folder/independent.name.replace('independent','focus')).read_text())
                assert a['frames']==b['frames'] and a['model_revision']==b['model_revision']
                ablation_pairs+=1
    train=read_jsonl(report/'train_export.jsonl')
    for r in train:
        c=by_claim[r['source_claim_id']];assert r['original_text']==c['original_text']
        assert r['split']==c['split']=='train' and r['group_id']==c['group_id']
        assert r['supervision']=='weak_whole_video_scenario' and r['event_targets']==[]
        assert r['binary_window_target'] is None and r['eligible_for_strong_event_training'] is False
    train_human=read_jsonl(report/'train_human_actions.jsonl')
    for h in train_human:
        assert h['decision']=='accept' and h['action_confirmed'] is True and h['eligible_for_action_training'] is True
        assert h['split']==groups[h['group_id']]['split']=='train'
    loaded=sum(len(batch) for batch in scenario_loader(root,report/'train_export.jsonl',batch_size=4))
    assert loaded==len(train)
    for q in read_jsonl(report/'review_queue.jsonl'):
        assert q['status']=='requires_human_review'
        for f in q['frames']:
            assert digest(report/f['path'])==f['sha256']
            assert 0<=f['timestamp']<=groups[q['group_id']]['duration_sec']
    changed=[];count=0
    for name in ['protected_inputs.json','protected_autolabel.json']:
        for path,sha in json.loads((run/'source'/name).read_text()).items():
            count+=1
            if not (root/path).exists() or digest(root/path)!=sha:changed.append(path)
    result=dict(ok=not changed,completed_chains=completed,total_scenarios=len(scenarios),
      visual_proposals=observations,matched_ablation_window_pairs=ablation_pairs,
      weak_train_scenarios_loaded=loaded,confirmed_train_actions=len(train_human),
      protected_files_checked=count,changed_protected_files=changed,
      split_counts=dict(Counter(r['split'] for r in mapping)),
      independent_test=False,automatic_strong_targets=0)
    write_json(report/'verification.json',result)
    if changed:raise ValueError('Protected input changed; see verification.json')
    print(json.dumps(result,ensure_ascii=False));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--run',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();verify(a.root,a.run,a.report)
