"""Build review output, then require actor/operation continuity for positive rule findings.
Keeps the original rule layer and all raw Qwen responses alongside the stricter layer.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil

from ml.autolabel.dataset import read_jsonl,digest
from ml.narration_training.prepare import write_json,write_jsonl
from .build import build,render
from .context import ACTIONS,rule_assessments,validate_observations
from .evaluation import compare,partition


def guarded_assessments(observations):
    rows=rule_assessments(observations);lookup={o['observation_id']:o for o in observations}
    for r in rows:
        r['rule_engine_version']='continuity_guard_v1';r['base_rule_verdict']=r['verdict']
        if r['verdict'] not in ['no_violation_observed','violation_supported']:continue
        obs=[lookup[e['observation_id']] for e in r['evidence']];missing=[]
        if any(o['uncertainty'].strip() or o['visibility']!='clear' for o in obs):missing.append('неустранённая неопределённость в опорном наблюдении')
        actors={o['actor'] for o in obs if o['kind']=='event'}
        if 'unknown' in actors or len(actors)!=1:missing.append('единый наблюдаемый участник последовательности не установлен')
        events=[o for o in obs if o['kind']=='event']
        if len(events)>=2:
            start=min(o['start_sec'] for o in events);end=max(o['end_sec'] for o in events)
            interruptions=[o for o in observations if o['kind']=='event' and o['action'] in ['person_enters','person_leaves','employee_leaves','employee_returns','occluded'] and start<o['start_sec']<end and o not in obs]
            if interruptions:missing.append('между этапами есть смена участников или разрыв наблюдения')
        if r['rule_id']=='authenticity_after_counting':
            tracks={o['track_id'] for o in events}
            if 'unknown' in tracks or len(tracks)!=1 or any(o['object']!='cash' for o in events):missing.append('не установлено, что пересчёт и проверка относятся к тем же деньгам')
        if r['rule_id']=='identity_before_cash':
            # Employee identity alone cannot link a check to the next customer's transaction.
            customers={o.get('from_actor','unknown') for o in events}
            if 'unknown' in customers or len(customers)!=1:missing.append('не установлено, что документ и деньги относятся к одному клиенту')
        if missing:
            r['verdict']='insufficient_data';r['missing_evidence']+=missing
            r['explanation']='Порядок предложенных действий виден в таймкодах, но связь участников/операции или уверенность в наблюдениях недостаточны для положительного вывода.'
    return rows


def coverage(folder,duration):
    path=folder/'focus_plan.json';windows=json.loads(path.read_text()) if path.exists() else []
    spans=sorted((w['context_start_sec'],w['context_end_sec']) for w in windows);merged=[]
    for a,b in spans:
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    gaps=[];cursor=0.
    for a,b in merged:
        if a>cursor:gaps.append([cursor,a])
        cursor=max(cursor,b)
    if cursor<duration:gaps.append([cursor,duration])
    return dict(dense_context_intervals=merged,dense_context_seconds=sum(b-a for a,b in merged),
      dense_context_fraction=sum(b-a for a,b in merged)/duration,coarse_only_intervals=gaps,
      missing_action_absence_proven=False,all_raw_frames_semantically_inspected=False)


def finalize(root,run,output):
    root=Path(root).resolve();run=Path(run).resolve();out=Path(output).resolve()
    build(root,run,out)
    code=out/'code';code.mkdir()
    for path in (root/'ml/narration_training/organizer').glob('*.py'):shutil.copy2(path,code/path.name)
    write_json(out/'report_code_manifest.json',{p.name:digest(p) for p in code.glob('*.py')})
    scenarios=read_jsonl(out/'scenarios.jsonl');base=deepcopy(scenarios)
    write_jsonl(out/'scenarios_before_continuity_guard.jsonl',base)
    changes=[];examples=[];rejected=[]
    for s in scenarios:
        observations=s['visual_evidence']['observations']
        folder=run/'blind'/s['group_id']
        for path in [folder/'survey.json',*sorted(folder.glob('focus_??.json')),*sorted(folder.glob('independent_??.json'))]:
            if not path.exists():continue
            raw=json.loads(path.read_text());refs=raw.get('frames',[])
            if not refs:continue
            _,errors=validate_observations(raw['answer'],refs,s['group_id'],path.stem,(min(f['timestamp'] for f in refs),max(f['timestamp'] for f in refs)))
            for error in errors:
                proposal=raw['answer']['events' if error['kind']=='event' else 'states'][error['index']]
                rejected.append(dict(video=s['video'],group_id=s['group_id'],split=s['split'],stage=path.stem,
                  raw_source=str(path.relative_to(run)),source_sha256=digest(path),reason=error['error'],
                  unvalidated_model_proposal=proposal,review_status='requires_human_review',
                  supplied_context_start_sec=min(f['timestamp'] for f in refs),supplied_context_end_sec=max(f['timestamp'] for f in refs),
                  question='Укажите реальные кадры и допустимое действие/состояние для этого предложения либо отклоните его. Таймкод окна не является доказательством утверждения.',
                  eligible_for_training=False,used_in_memory=False))
        s['rule_assessments']=guarded_assessments(observations) if s['visual_evidence']['full_video_survey_completed'] else []
        s['coverage']=coverage(run/'blind'/s['group_id'],s['organizer_claim']['duration_sec'])
        s['visual_evidence']['not_ground_truth']=True
        for r in s['rule_assessments']:
            if r['verdict']!=r['base_rule_verdict']:
                changes.append(dict(video=s['video'],group_id=s['group_id'],rule_id=r['rule_id'],
                  old_verdict=r['base_rule_verdict'],verdict=r['verdict'],missing_evidence=r['missing_evidence'],
                  evidence=r['evidence'],review_status='requires_human_review'))
        if s['organizer_number'] in [2,10,15,22,34,38,43,44]:
            examples.append(dict(video=s['video'],group_id=s['group_id'],split=s['split'],
              visual_proposals=[dict(action=o['action'],property=o['property'],value=o['value'],start_sec=o['start_sec'],end_sec=o['end_sec'],visibility=o['visibility'],uncertainty=o['uncertainty'],evidence=o['evidence']) for o in observations],
              memory_path=str(run/'blind'/s['group_id']/'memory_trace.jsonl'),rule_assessments=s['rule_assessments'],
              organizer_claim=s['organizer_claim'],human_confirmed=s['human_confirmed'],
              conflicts=s['conflict_ids'],coverage=s['coverage'],human_review_required=True))
    write_jsonl(out/'scenarios.jsonl',scenarios);write_jsonl(out/'continuity_guard_reviews.jsonl',changes)
    write_jsonl(out/'worked_scenarios.jsonl',examples)
    write_jsonl(out/'rejected_visual_proposals.jsonl',rejected)
    human=read_jsonl(out/'human_confirmed.jsonl')
    eligible=[r for r in human if r['split']=='train' and r['action_confirmed'] and r['eligible_for_action_training']]
    strong,states,other=partition(eligible)
    write_jsonl(out/'train_human_actions.jsonl',strong)
    write_jsonl(out/'train_human_states.jsonl',states);write_jsonl(out/'train_human_unmapped.jsonl',other)
    summary=json.loads((out/'summary.json').read_text());before=json.loads((out/'context_ablation.json').read_text())
    write_json(out/'context_ablation_before_type_filter.json',before)
    comparison=compare(run/'blind',read_jsonl(run/'number_mapping.jsonl'),human)
    write_json(out/'context_ablation.json',comparison);summary['ablation_totals']=comparison['totals']
    summary['rule_verdicts']=dict(Counter(r['verdict'] for s in scenarios for r in s['rule_assessments']))
    all_events=[o for s in scenarios for o in s['visual_evidence']['observations'] if o['kind']=='event']
    summary['model_quality_diagnostics']=dict(event_role_counts=dict(Counter(o['role'] for o in all_events)),
      event_action_counts=dict(Counter(o['action'] for o in all_events)),
      role_labels_are_unconfirmed=True,customer_change_not_established_by_role_frequency=True,
      observation_count_is_not_unique_event_count=True)
    summary.update(continuity_guard_reviews=len(changes),train_confirmed_action_records=len(strong),
      train_confirmed_state_records=len(states),train_confirmed_unmapped_records=len(other),
      rejected_visual_proposals=len(rejected),rejection_reasons=dict(Counter(r['reason'] for r in rejected)),
      automatically_confirmed_action_records=0,automatically_confirmed_violation_records=0,
      remaining_visual_surveys=22-summary['completed_visual_surveys'],
      split_policy='frozen original manifest; active split changes are not adopted by this organizer source',
      manifest_sha256=digest(root/'data/training/manifest.csv'))
    queue=read_jsonl(out/'review_queue.jsonl');write_json(out/'summary.json',summary)
    write_json(out/'training_eligibility.json',dict(
      weak_organizer_scenarios=len(read_jsonl(out/'train_export.jsonl')),
      confirmed_native_action_records=len(strong),confirmed_state_records=len(states),
      confirmed_unspecified_records_requiring_classification=len(other),
      automatically_promoted_visual_proposals=0,organizer_video_claims_used_as_window_targets=0,
      legal_violation_targets=0,validation_explanations_in_training=0,
      automatic_training_started=False,production_model_replaced=False,
      source_contract='Read-only snapshot of live journal schema_version=1, explicit accept/action_confirmed/eligible flags, source fingerprint and frozen split checked.',
      classifier_integration='Separate exports; no change to existing training entrypoints or production. Native human actions outside Qwen vocabulary remain available.',
      human_records_outside_organizer_groups='Retained in human_journal_snapshot.jsonl; excluded from this scenario-scoped export, not deleted or treated as negative.'))
    render(root,run,out,scenarios,queue,summary,comparison)
    page=out/'index.html';text=page.read_text();text=text.replace('<h2>Файлы</h2>',
      '<h2>Дополнительные проверки</h2><p><a href="worked_scenarios.jsonl">Разобранные сценарии с таймкодами</a> · <a href="continuity_guard_reviews.jsonl">Проверка связи участников и операции</a> · <a href="rejected_visual_proposals.jsonl">Отклонённые предложения без корректных доказательств</a> · <a href="train_human_actions.jsonl">Только подтверждённые train-действия</a></p><h2>Файлы</h2>')
    page.write_text(text)
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();finalize(a.root,a.run,a.output)
