"""Pure validation, sampling plans and conservative evidence gates."""
import math
import copy
import json
from ml.narration_training.organizer.context import ACTIONS, PROPERTIES, RULES, validate_observations, memory
from ml.context_pipeline.rules import guarded_assessments
from .regulatory import regulatory_source

STEP = .5
MAX_DURATION = 180
MAX_SECONDS = 900


def parse_result(raw):
    value=json.loads(raw)
    required={'events','states','rule_checks'}
    if not isinstance(value,dict) or set(value)!=required or any(not isinstance(value[k],list) for k in required):
        raise ValueError('Expected observation arrays; input memory is not an observation result')
    return value


def windows(frames, size=12, overlap=2):
    if size <= overlap or overlap < 0:
        raise ValueError('Invalid window size')
    result=[]
    for i in range(0, len(frames), size-overlap):
        block=frames[i:i+size]
        if result and all(f['id'] in {x['id'] for x in result[-1]} for f in block):
            break
        result.append(block)
    return result


def inspect_answer(answer, refs, group, stage):
    if not isinstance(answer,dict):
        return [],[],[dict(error='Expected JSON object')]
    answer=copy.deepcopy(answer)
    for key in ('events','states'):
        if isinstance(answer.get(key),list):
            for item in answer[key]:
                if isinstance(item,dict):
                    for field in ('actor','role','object','track_id','from_actor','to_actor','recipient','value'):
                        if item.get(field) is None:item[field]='unknown'
    observations,rejected=validate_observations(answer,refs,group,stage,
        (min(f['timestamp'] for f in refs),max(f['timestamp'] for f in refs)))
    for o in observations:
        o['source']='local_gemma_visual_proposal'
    lookup={f['id']:f for f in refs};claims=[]
    rules=answer.get('rule_checks') or []
    if not isinstance(rules,list):
        rejected.append(dict(error='rule_checks must be an array'));rules=[]
    for r in rules:
        if not isinstance(r,dict) or r.get('rule_id') not in RULES:
            rejected.append(dict(error='Unknown rule'));continue
        ids=r.get('evidence_frames',[])
        if not isinstance(ids,list) or not all(isinstance(i,str) and i in lookup for i in ids):
            rejected.append(dict(error='Unknown rule evidence'));continue
        verdict=r.get('verdict')
        if verdict not in {'violation_supported','no_violation_observed','insufficient_data','not_applicable'}:
            rejected.append(dict(error='Unknown verdict'));continue
        missing=r.get('missing_evidence') or []
        if isinstance(missing,str):missing=[missing]
        if not isinstance(missing,list):
            rejected.append(dict(error='Invalid missing_evidence'));missing=['Model returned invalid missing-evidence metadata']
        claims.append(dict(rule_id=r['rule_id'],model_verdict=verdict,
            verdict='insufficient_data' if verdict=='violation_supported' else verdict,
            evidence=[lookup[i] for i in dict.fromkeys(ids)],
            triggering_event=str(r.get('triggering_event',''))[:400],
            explanation=str(r.get('explanation',''))[:1200],
            missing_evidence=[str(v)[:400] for v in missing][:8],
            stage=stage,review_status='proposed',human_confirmed=False,
            validation_note='Rule claims are checked against structured visual observations.'))
    return observations,claims,rejected


def assess(observations,claims):
    """Keep raw VLM claims; promote only evidence-linked supported scenarios."""
    rows=guarded_assessments(observations)
    for r in rows:
        r['source']='local_gemma_evidence_gated_rule'
        r['regulatory_source']=regulatory_source(r['rule_id'])
        related=[c for c in claims if c['rule_id']==r['rule_id']]
        r['model_claims']=related
        if related and r['verdict']=='insufficient_data':
            r['review_priority']='high' if any(c['model_verdict']=='violation_supported' for c in related) else 'normal'
    return rows


def memory_before(observations, timestamp):
    # An overlapping window may begin before a previous interval ends. Retain
    # only its earlier evidence instead of losing all history or leaking future frames.
    past=[]
    for o in observations:
        refs=[f for f in o['evidence'] if f['timestamp']<timestamp]
        if len(refs)<(2 if o['kind']=='event' else 1):continue
        times=[f['timestamp'] for f in refs]
        past.append(dict(o,evidence=refs,start_sec=min(times),end_sec=max(times),last_observed_sec=max(times)))
    return memory(past,timestamp)


def prompt_memory(value):
    value=copy.deepcopy(value)
    limits={'last_known_facts':10,'transfer_history':3,'people_last_seen':4,
            'cash_customer_links':3,'counting_history':3,'authenticity_check_history':3,
            'recent_actions':5,'temporal_links':4}
    result={k:(v[-limits[k]:] if k in limits and isinstance(v,list) else v) for k,v in value.items()}
    for rows in result.values():
        if isinstance(rows,list):
            for row in rows:
                if isinstance(row,dict) and 'evidence_timestamps' in row:
                    row['evidence_timestamps']=row['evidence_timestamps'][:1]+row['evidence_timestamps'][-1:]
    return result


def coverage(frames,inspected):
    all_ids={f['id'] for f in frames};done=all_ids&set(inspected)
    timestamps=sorted({float(f['timestamp']) for f in frames})
    deltas=[b-a for a,b in zip(timestamps,timestamps[1:]) if b>a]
    sampling_step=round(min(deltas),6) if deltas else STEP
    return dict(sampling_step_sec=sampling_step,total_sampled_frames=len(all_ids),
        inspected_sampled_frames=len(done),all_sampled_frames_inspected=done==all_ids and bool(all_ids),
        sampled_coverage_fraction=len(done)/len(all_ids) if all_ids else 0,
        unsampled_action_absence_proven=False,
        missing_frame_ids=sorted(all_ids-done))
