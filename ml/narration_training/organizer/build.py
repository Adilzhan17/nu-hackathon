"""Join five separate layers AFTER independent inference; snapshot live reviews read-only."""
import argparse
from collections import Counter
import datetime as dt
import hashlib
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

from ml.autolabel.dataset import read_jsonl,digest
from ml.autolabel.sampling import frames
from ml.narration_training.prepare import write_json,write_jsonl,new_output
from .context import RULES,rule_assessments

ALIASES={'cash_counting':'cash_counter_used','cash_placed_on_desk':'cash_placed',
         'document_presented':'document_transfer','document_returned':'document_transfer','document_inspection':'document_checked',
         'safe_door_closing':'safe_closing','cash_counted_by_hand':'cash_counter_used','leave_workstation':'employee_leaves',
         'visitor_enters':'person_enters','visitor_leaves':'person_leaves'}

def canonical(action):return ALIASES.get(action,action)

def overlap(a,b):return max(0,min(a['end_sec'],b['end_sec'])-max(a['start_sec'],b['start_sec']))

def human_snapshot(root,out,mapping):
    source=root/'data/human_review/decisions.jsonl';raw=source.read_bytes() if source.exists() else b''
    # A writer may be appending the last record. Snapshot only complete JSONL records; never lock or write its store.
    prefix=raw[:raw.rfind(b'\n')+1] if raw else b''
    (out/'human_journal_snapshot.jsonl').write_bytes(prefix)
    rows=[json.loads(line) for line in prefix.splitlines() if line.strip()];latest={}
    for row in rows:
        if row['task_id'] not in latest or row['version']>latest[row['task_id']]['version']:latest[row['task_id']]=row
    sources=[root/'data/training/manifest.csv']+[root/'data/autolabel/detailed_train_report'/n for n in
             ['video_timelines.jsonl','user_reference_alignment.jsonl','moments.jsonl','review_queue.jsonl']]
    hashes={str(p.relative_to(root)):digest(p) if p.exists() else hashlib.sha256(b'').hexdigest() for p in sources}
    fingerprint=hashlib.sha256(json.dumps(hashes,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    groups={r['group_id']:r for r in mapping};accepted=[];excluded=[]
    for row in latest.values():
        reasons=[];group=groups.get(row['group_id'])
        if not group:reasons.append('outside_current_scenario_groups')
        else:
            if row.get('video_sha256')!=group['video_sha256'] or row['video'] not in group['aliases']:reasons.append('identity_mismatch')
            if row.get('split')!=group['split']:reasons.append('split_mismatch')
            if not 0<=row['start_sec']<row['end_sec']<=group['duration_sec']:reasons.append('bounds')
        if row.get('source_fingerprint')!=fingerprint:reasons.append('stale_source_fingerprint')
        if row.get('decision')!='accept' or row.get('action_confirmed') is not True or row.get('eligible_for_action_training') is not True:reasons.append('not_confirmed_action')
        if row.get('visibility') not in ['clear','partial'] or not row.get('reviewer'):reasons.append('missing_review_provenance')
        evidence=row.get('evidence_time_sec')
        if not isinstance(evidence,(int,float)) or not row['start_sec']<=evidence<=row['end_sec']:reasons.append('missing_evidence_timestamp')
        if reasons:excluded.append(dict(task_id=row['task_id'],reasons=reasons));continue
        accepted.append(dict(row,authority='human_confirmed',source_ref='human_journal_snapshot.jsonl',
          source_snapshot_sha256=hashlib.sha256(prefix).hexdigest(),legal_violation_target=None))
    write_jsonl(out/'human_confirmed.jsonl',accepted);write_jsonl(out/'human_excluded.jsonl',excluded)
    write_json(out/'human_snapshot_metadata.json',dict(captured_at=dt.datetime.now(dt.timezone.utc).isoformat(),
       source=str(source.relative_to(root)),snapshot_bytes=len(prefix),snapshot_sha256=hashlib.sha256(prefix).hexdigest(),
       ignored_incomplete_tail_bytes=len(raw)-len(prefix),journal_records=len(rows),current_decisions=len(latest),
       confirmed_in_selected_groups=len(accepted),live_store_read_only=True,source_fingerprint=fingerprint))
    return accepted

def ablation(blind,mapping,human):
    per_video=[];matches=[]
    for row in mapping:
        folder=blind/row['group_id'];done=folder/'complete.json'
        if not done.exists() or not json.loads(done.read_text())['ablation']:continue
        context=[r for r in read_jsonl(folder/'visual_observations.jsonl') if r['kind']=='event' and r['stage'].startswith('focus')]
        independent=[r for r in read_jsonl(folder/'independent_observations.jsonl') if r['kind']=='event']
        windows=json.loads((folder/'focus_plan.json').read_text())
        relevant=[h for h in human if h['group_id']==row['group_id'] and h['observed_action'] not in ['other_observed_action','unclear_action']
                  and any(h['start_sec']<=w['context_end_sec'] and w['context_start_sec']<=h['end_sec'] for w in windows)]
        counts={}
        for mode,events in [('without_history',independent),('with_history',context)]:
            found=0
            for h in relevant:
                evidence=[o['observation_id'] for o in events if canonical(o['action'])==canonical(h['observed_action'])
                          and overlap(h,o)>0 and overlap(h,o)/max(.001,min(h['end_sec']-h['start_sec'],o['end_sec']-o['start_sec']))>=.2]
                found+=bool(evidence)
                matches.append(dict(video=row['video'],group_id=row['group_id'],human_task_id=h['task_id'],
                  action=h['observed_action'],start_sec=h['start_sec'],end_sec=h['end_sec'],condition=mode,
                  recovered=bool(evidence),supporting_proposal_ids=evidence))
            counts[mode]=dict(proposed_actions=len(events),confirmed_records_recovered=found,
                             confirmed_records_in_supplied_windows=len(relevant),
                             reference_recovery_fraction=found/len(relevant) if relevant else None)
        per_video.append(dict(video=row['video'],group_id=row['group_id'],**counts))
    totals={m:dict(recovered=sum(v[m]['confirmed_records_recovered'] for v in per_video),
                  references=sum(v[m]['confirmed_records_in_supplied_windows'] for v in per_video)) for m in ['without_history','with_history']}
    return dict(per_video=per_video,totals=totals,matches=matches,
       protocol='Same focus frames, model revision, generation settings, general rules; only prior visual memory differs. Adaptive windows chosen by the common full-video survey.',
       counting_alias='cash_counting and cash_counted_by_hand are compared at the broad counting action family; loading is not counting',
       excluded_metrics=['precision','event_mAP','whole_video_accuracy'],
       limitations=['Small non-independent human-reviewed subset, sparse incomplete references; recovery is not full event recall.',
                    'No label for unreviewed proposals: extra actions cannot be counted as false positives.',
                    'Human labels never enter either prompt; organizer answers enter neither prompt.',
                    'Timing criterion is coarse overlap, not a localization quality claim.'])

def build(root,run,output):
    root=root.resolve();run=run.resolve();out=new_output(root,output);blind=run/'blind'
    claims=read_jsonl(run/'source/organizer_claims_v1.jsonl');mapping=read_jsonl(run/'number_mapping.jsonl')
    human=human_snapshot(root,out,mapping)
    old=read_jsonl(root/'data/autolabel/detailed_train_report/moments.jsonl')
    scenarios=[];conflicts=[];queue=[];train=[]
    for claim in claims:
        group=claim['group_id'];folder=blind/group
        complete=(folder/'complete.json').exists()
        visual=read_jsonl(folder/'visual_observations.jsonl') if complete else []
        rules=read_jsonl(folder/'rule_assessments.jsonl') if complete else []
        h=[r for r in human if r['group_id']==group]
        previous=[r for r in old if r['group_id']==group]
        issues=[]
        def issue(kind,why,start=None,end=None,refs=None):
            r=dict(conflict_id=f'conflict_{len(conflicts):04d}',claim_id=claim['claim_id'],video=claim['video'],group_id=group,
              split=claim['split'],kind=kind,question=why,start_sec=start,end_sec=end,source_references=refs or [],
              status='requires_human_review',legal_violation_target=None)
            conflicts.append(r);issues.append(r)
        if claim['number']==13:
            issue('organizer_scope_ambiguity','В пояснении №13 дан ожидаемый порядок, но нарушение не названо. Какой конкретный этап требуется проверять?',0,min(6,claim['duration_sec']))
        if claim['number']==34:
            issue('organizer_actor_ambiguity','Кто ушёл во время пересчёта: сотрудник или клиент? Уточните видимую роль без установления личности.',0,min(8,claim['duration_sec']))
        if claim['number']==44:
            issue('organizer_explicit_uncertainty','Была ли видна отдельная проверка подлинности или функция проверки встроена в счётную машину? Сохранить сомнение, если установить нельзя.',max(0,claim['duration_sec']-8),claim['duration_sec'])
        if claim['number'] in [2,10]:
            matching=[r for r in h if r['observed_action'] in ['document_transfer','cash_received','phone_put_away','cash_counting']]
            if matching:
                issue('requirement_vs_human_sequence_review','Пояснение задаёт требование, а не точный таймкод ошибки. Проверьте фактический порядок подтверждённых действий; не превращайте требование в положительную метку нарушения.',
                  min(r['start_sec'] for r in matching),max(r['end_sec'] for r in matching),[r['task_id'] for r in matching])
        # Absence of a human action in model proposals is a retrieval gap, not a contradiction.
        for ref in h:
            if ref['observed_action']=='other_observed_action':continue
            matched=[o for o in visual if o['kind']=='event' and canonical(o['action'])==canonical(ref['observed_action']) and overlap(o,ref)>0]
            if complete and not matched:
                issue('human_reference_not_recovered_by_qwen','Подтверждённое действие не восстановлено Qwen. Сохранить ручную метку; проверить пропуск и границы модели.',ref['start_sec'],ref['end_sec'],[ref['task_id']])
        # Compare opposite state proposals only at exactly shared evidence times/track, not merely neighboring windows.
        states=[o for o in visual if o['kind']=='state' and o['visibility']=='clear']
        opposing=[]
        for i,a in enumerate(states):
            for b in states[i+1:]:
                if a['property']==b['property'] and a['track_id']==b['track_id']!='unknown' and {a['value'],b['value']}=={'open','closed'}:
                    common={f['timestamp'] for f in a['evidence']}&{f['timestamp'] for f in b['evidence']}
                    if common:opposing.append((a,b,min(common)))
        for a,b,t in opposing[:2]:issue('qwen_internal_state_disagreement','Qwen дал противоположные состояния одного предполагаемого объекта на общем кадре. Какое состояние действительно видно?',max(0,t-2),min(claim['duration_sec'],t+2),[a['observation_id'],b['observation_id']])
        mismatches=[]
        for older in previous:
            old_objects=set(older['objects'])&{'cash','document','key','stamp'}
            for new in visual:
                if new['kind']=='event' and new['visibility']=='clear' and new['object'] in {'cash','document','key','stamp'} and old_objects and new['object'] not in old_objects and overlap(older,new)>1:
                    mismatches.append((older,new))
        for older,new in mismatches[:2]:
            issue('possible_qwen_object_disagreement','Старый и новый проходы Qwen называют разные предметы в пересекающемся интервале. Это могут быть разные одновременные действия; проверьте предмет и участника.',
                  max(older['start_sec'],new['start_sec']),min(older['end_sec'],new['end_sec']),[older['moment_id'],new['observation_id']])
        raw_findings=read_jsonl(folder/'qwen_rule_proposals.jsonl') if complete else []
        guarded={r['rule_id']:r for r in rules}
        overclaims=[]
        for window in raw_findings:
            for finding in window['findings'] if isinstance(window['findings'],list) else []:
                rid=finding.get('rule_id') if isinstance(finding,dict) else None
                if rid in guarded and finding.get('verdict')=='violation_supported' and guarded[rid]['verdict']=='insufficient_data':
                    overclaims.append(dict(window=window['window'],raw_qwen_finding=finding,final_rule_assessment=guarded[rid]))
        if overclaims:
            windows=json.loads((folder/'focus_plan.json').read_text());index=overclaims[0]['window']
            w=windows[index] if index is not None else dict(start_sec=0,end_sec=min(8,claim['duration_sec']))
            issue('qwen_rule_overclaim_filtered','Qwen предложил нарушение, но обязательных свидетельств недостаточно. Проверьте пропущенные условия; итог оставлен insufficient_data.',w['start_sec'],w['end_sec'],[overclaims[0]['raw_qwen_finding']['rule_id']])
        scenario=dict(video=claim['video'],group_id=group,split=claim['split'],organizer_number=claim['number'],
          organizer_claim=claim,visual_evidence=dict(status='model_proposed_not_human_confirmed',observations=visual,
           full_video_survey_completed=complete,sampling=json.loads((folder/'sampling.json').read_text()) if complete else None),
          memory_and_links=dict(path=str((folder/'memory_trace.jsonl').relative_to(run)),
              last_known_not_current=True,legal_cash_ownership='unknown'),
          rule_assessments=rules,qwen_overclaims_filtered=overclaims,human_confirmed=h,
          insufficient_data=claim['unknowns']+['Визуальные предложения модели требуют проверки человеком; выборка кадров не доказывает отсутствие событий.'],
          comparison_status='post_inference_only_organizer_text_never_sent_to_qwen',conflict_ids=[i['conflict_id'] for i in issues],
          organizer_claim_overrides_human=False,whole_video_binary_target=None)
        scenarios.append(scenario)
        if issues:queue.append(issues[0])
        if claim['split']=='train':
            train.append(dict(video=claim['video'],video_sha256=claim['video_sha256'],group_id=group,split='train',
              source_type='organizer_provided',source_claim_id=claim['claim_id'],original_text=claim['original_text'],
              source_sha256=claim['source_sha256'],source_line=claim['source_line'],
              source_reference=str((run/'source/original_message.txt').relative_to(root)),
              expected_sequence=claim['expected_sequence'],action_state_candidates=claim['observable_action_state_candidates'],
              presumed_violation=claim['presumed_violation'],organizer_certainty=claim['organizer_certainty'],
              required_visual_evidence=claim['required_visual_evidence'],unknowns=claim['unknowns'],
              supervision='weak_whole_video_scenario',trust=dict(provenance='organizer_provided_relayed_by_user',
                   mapping='user_confirmed',time='unlocalized',visual_confirmation='not_confirmed'),
              eligible_for_weak_scenario_training=True,eligible_for_clip_caption_alignment=False,
              eligible_for_strong_event_training=False,event_targets=[],binary_window_target=None,legal_violation_target=None,
              human_source_reference='human_confirmed.jsonl',visual_source_reference=str((folder/'visual_observations.jsonl').relative_to(run)),
              conflict_ids=[i['conflict_id'] for i in issues],must_not_load_full_source_message_as_training_text=True))
    # Prefer actual conflicts; a diverse compact queue, with all conflicts retained separately.
    queue=sorted(queue,key=lambda r:(r['kind'] not in ['human_reference_not_recovered_by_qwen','qwen_rule_overclaim_filtered','requirement_vs_human_sequence_review'],r['claim_id']))[:12]
    previews=out/'evidence';previews.mkdir()
    for q in queue:
        info=next(r for r in mapping if r['group_id']==q['group_id']);duration=info['duration_sec']
        a=q['start_sec'] if q['start_sec'] is not None else 0;b=q['end_sec'] if q['end_sec'] is not None else duration
        q['context_start_sec']=max(0,a-3);q['context_end_sec']=min(duration,b+3);q['frames']=[]
        for role,t in [('before',max(0,a-1)),('during',(a+b)/2),('after',min(duration-.1,b+1))]:
            sampled=next(frames(root/'data/raw/open'/info['video'],max(0,t),duration,1,max_side=960),None)
            if sampled:
                actual,im=sampled;path=previews/f"{q['conflict_id']}_{role}.jpg";im.save(path,quality=94)
                actual_role=('end_of_clip_no_after_frame' if role=='after' and actual<=b else
                             'start_of_clip_no_before_frame' if role=='before' and actual>=a else role)
                q['frames'].append(dict(role=actual_role,requested_role=role,timestamp=actual,path=str(path.relative_to(out)),sha256=digest(path),
                                        interpretation='context_frame_not_automatically_confirmed_evidence'))
    write_jsonl(out/'scenarios.jsonl',scenarios)
    write_jsonl(out/'train_export.jsonl',train)
    write_jsonl(out/'conflicts.jsonl',conflicts);write_jsonl(out/'review_queue.jsonl',queue)
    compare=ablation(blind,mapping,human);write_json(out/'context_ablation.json',compare)
    summary=dict(organizer_claims=len(claims),train_scenarios=len(train),validation_scenarios=len(claims)-len(train),
      completed_visual_surveys=sum(s['visual_evidence']['full_video_survey_completed'] for s in scenarios),
      visual_observations=sum(len(s['visual_evidence']['observations']) for s in scenarios),
      human_confirmed_records=len(human),conflicts=len(conflicts),review_queue=len(queue),
      conflict_types=dict(Counter(c['kind'] for c in conflicts)),
      rule_verdicts=dict(Counter(r['verdict'] for s in scenarios for r in s['rule_assessments'])),
      ablation_totals=compare['totals'],training_source_is_scenario_not_frame_labels=True,
      old_manual_decisions_modified=False,site_stopped=False)
    write_json(out/'summary.json',summary);render(root,run,out,scenarios,queue,summary,compare)
    print(json.dumps(summary,ensure_ascii=False))

def render(root,run,out,scenarios,queue,summary,compare):
    esc=lambda x:html.escape(str(x))
    rel=lambda p:quote(os.path.relpath(p,out),safe='/')
    cards=''
    for s in scenarios:
        c=s['organizer_claim'];g=s['group_id'];folder=run/'blind'/g
        table=''.join(f'<tr><td>{esc(r["rule_id"])}</td><td>{esc(r["verdict"])}</td><td>{esc(r["explanation"])}</td><td>{esc("; ".join(r["missing_evidence"]))}</td></tr>' for r in s['rule_assessments'])
        observations=s['visual_evidence']['observations']
        events=''.join(f'<li>{o["start_sec"]:.2f}–{o["end_sec"]:.2f} с · {esc(o["action"] or o["property"])} · {esc(o["description"])} <em>({esc(o["visibility"])}, предложение модели)</em> '+
          ' '.join(f'<a href="{rel(run/"blind"/f["path"])}">{f["timestamp"]:.2f} с</a>' for f in o['evidence'][:3])+'</li>' for o in observations)
        cards+=f'''<details><summary>№{c['number']} · {esc(s['video'])} · {esc(s['split'])} · {len(observations)} наблюдений</summary>
        <p class="claim"><b>organizer_claim, сценарий целого видео:</b> {esc(c['original_text'])}</p>
        <p>Ожидается: {esc(' → '.join(c['expected_sequence']))}. Предполагаемое нарушение: {esc(c['presumed_violation'] or 'не названо')}.
        Уверенность организатора: {esc(c['organizer_certainty'])}. Это не статья законодательства.</p>
        <p>Неизвестно: {esc('; '.join(s['insufficient_data']))}</p>
        <p>Подтверждений человека: {len(s['human_confirmed'])}. Полный временной обзор: {s['visual_evidence']['full_video_survey_completed']}; исходное видео семантически просматривалось с выборкой 1 кадр/с, а переходы — 4 кадра/с. Это не доказательство отсутствия событий между кадрами.</p>
        <p><a href="{rel(folder/'survey.json')}">Обзор</a> · <a href="{rel(folder/'memory_trace.jsonl')}">Память и связи</a> · <a href="{rel(folder/'qwen_rule_proposals.jsonl')}">Сырые выводы Qwen</a></p>
        <table><tr><th>Правило</th><th>Вердикт по эпизоду</th><th>Обоснование</th><th>Чего не хватает</th></tr>{table}</table>
        <details><summary>Визуальные предложения с таймкодами и кадрами</summary><ol>{events}</ol></details></details>'''
    qs=''
    for q in queue:
        frames_html=''.join(f'<figure><img loading="lazy" src="{esc(f["path"])}"><figcaption>{esc(f["role"])} · {f["timestamp"]:.3f} с</figcaption></figure>' for f in q['frames'])
        qs+=f'<details><summary>{esc(q["video"])} · {esc(q["kind"])}</summary><p>{esc(q["question"])}</p><div class="frames">{frames_html}</div><p>Кадры дают контекст; наличие предмета или нарушение ими автоматически не подтверждаются.</p></details>'
    comparison=''.join(f'<tr><td>{esc(v["video"])}</td><td>{v["without_history"]["confirmed_records_recovered"]}/{v["without_history"]["confirmed_records_in_supplied_windows"]}</td><td>{v["with_history"]["confirmed_records_recovered"]}/{v["with_history"]["confirmed_records_in_supplied_windows"]}</td></tr>' for v in compare['per_video'])
    text=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>ForteBank: контекст и пояснения организаторов</title>
    <style>body{{max-width:1250px;margin:30px auto;padding:0 18px;font:16px/1.5 system-ui;background:#fafbfc;color:#182635}}details{{padding:12px;margin:12px 0;border:1px solid #cad3df;background:white}}summary{{font-weight:650;cursor:pointer}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border:1px solid #cad3df;text-align:left}}.frames{{display:flex;gap:10px}}figure{{margin:0;flex:1;min-width:0}}img{{width:100%}}.claim{{background:#edf3ff;padding:12px}}.notice{{background:#fff0ca;padding:14px}}a{{color:#1554a2}}</style>
    <h1>Контекстный разбор сценариев</h1><p class="notice">Пять раздельных слоёв: визуальные предложения → память → выводы по правилам → утверждения организаторов → подтверждения человека. Автовыводы не являются GT. «Нарушение не замечено» относится только к конкретному эпизоду и правилу.</p>
    <p>Источник: {summary['organizer_claims']} пояснения, нумерация файлов подтверждена пользователем. Train: {summary['train_scenarios']}; раскрытые val: {summary['validation_scenarios']} групп (Х-4/Х-47 — один исходник). Ни ответы о конкретном видео, ни ручные метки не передавались Qwen. Общие правила происходят только из train-пояснений и ранее существовавших шаблонов.</p>
    <p>Визуальных обзоров: {summary['completed_visual_surveys']}; наблюдений: {summary['visual_observations']}; подтверждённых записей человека в снимке: {summary['human_confirmed_records']}. Конфликтов и вопросов: {summary['conflicts']}, компактная очередь: {summary['review_queue']}. Сайт и живая разметка продолжают работать.</p>
    <h2>Помогла ли история на тех же кадрах?</h2><table><tr><th>Видео</th><th>Без истории: найдено / проверяемых записей человека</th><th>С историей: найдено / проверяемых записей человека</th></tr>{comparison}</table>
    <p>Сравниваются одинаковые текущие кадры, Qwen, параметры генерации и общие правила. История — единственное изменение. Отбор окон общий, по полному обзору. Это восстановление небольшой неполной ручной выборки, не точность, не полный recall и не независимый тест; лишние предложения без отрицательной разметки не объявляются ложными.</p>
    <p><a href="context_ablation.json">Проверяемые совпадения, пропуски и ограничения</a>. Сырые ошибки вывода Qwen и нехватка доказательств сохраняются в сценариях и конфликтах; строгие правила могут понизить ответ до insufficient_data.</p>
    <h2>Файлы</h2><p><a href="{rel(run/'source/original_message.txt')}">Оригинальное сообщение</a> · <a href="{rel(run/'number_mapping.csv')}">Соответствие номеров</a> · <a href="{rel(run/'disclosed_validation.jsonl')}">Раскрытые val</a> · <a href="scenarios.jsonl">Все пять слоёв</a> · <a href="train_export.jsonl">Отдельный train-экспорт</a> · <a href="human_confirmed.jsonl">Снимок подтверждений</a> · <a href="conflicts.jsonl">Все вопросы/конфликты</a></p>
    <h2>Очередь проверки: до / во время / после</h2>{qs}<h2>Все сценарии</h2>{cards}</html>'''
    (out/'index.html').write_text(text)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.root,a.run,a.output)
