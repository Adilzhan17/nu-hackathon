"""Continuity guard frozen from organizer finalize v1; no organizer answers read."""
from .observations import rule_assessments

def whole_video_verdict(rules):
    """A supported episode is actionable; sparse negative findings do not clear a video."""
    return 'violation_supported' if any(
        r.get('verdict') == 'violation_supported' and r.get('evidence')
        for r in rules
    ) else 'insufficient_data'

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
