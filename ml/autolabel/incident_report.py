"""Convert explicit positive visual-state evidence into reviewable risk-pattern candidates."""
import json
from collections import Counter
from .normalize_incidents import normalize
from pathlib import Path


def collect(run,plan):
    path=run/'incident_check_summary.json'
    if not path.exists():return [],[],None
    summary=json.loads(path.read_text())
    if not summary['completed']:raise ValueError('Incident visual checks incomplete')
    checks=json.loads((run/'incident_check_plan.json').read_text())['checks']
    states=[];patterns=[];normalized=[]
    for c in checks:
        row=json.loads((run/'incident_check_records'/(c['check_id']+'.json')).read_text());n=normalize(row,c);normalized.append(n);d=n['parsed']
        if d is None:continue
        ts=[f['timestamp'] for f in c['frames']]
        for state in d['states']:
            indices=state['evidence_frames'];times=[ts[i] for i in indices]
            state_id=f'state_{len(states):04d}'
            states.append(dict(state_id=state_id,check_id=c['check_id'],video=c['video'],group_id=c['group_id'],split='train',
                               observed_state=state['state'],start_sec=min(times),end_sec=max(times),evidence_timestamps=times,
                               confidence=state['confidence'],confidence_kind='uncalibrated_model_self_rating',
                               uncertainty_reason=state['uncertainty_reason'],review_status='proposed',
                               model_id=plan['model_lock']['model_id'],model_revision=plan['model_lock']['model_revision']))
            dep=d['departure_frames']
            # Positive evidence on both sides of departure, not absence of a closing action.
            if not dep or min(indices)>min(dep) or max(indices)<max(dep):continue
            evidence=sorted({*times,*(ts[i] for i in dep)})
            patterns.append(dict(pattern_id=f'pattern_{len(patterns):04d}',video=c['video'],group_id=c['group_id'],split='train',
                pattern={'safe_door_open':'safe_open_across_worker_departure','cash_on_work_surface':'cash_visible_across_worker_departure','key_in_safe_lock':'key_in_lock_across_worker_departure'}[state['state']],
                state_proposal_id=state_id,check_id=c['check_id'],start_sec=min(evidence),end_sec=max(evidence),evidence_timestamps=evidence,
                worker_departure_timestamps=[ts[i] for i in dep],confidence=min(state['confidence'],d['departure_confidence']),
                confidence_kind='uncalibrated_model_self_rating',uncertainty_reason='; '.join(x for x in [state['uncertainty_reason'],d['uncertainty_reason']] if x),
                review_status='proposed',organizer_video_verdict=c['organizer_video_verdict'],
                legal_violation_target=None,eligible_for_violation_type_training=False,continuity_between_evidence_frames_verified=False,
                required_review='Confirm worker role, object/state visibility, continuity between evidence frames and applicable rule; this is not a legal verdict.',
                model_id=plan['model_lock']['model_id'],model_revision=plan['model_lock']['model_revision']))
    summary=dict(summary,normalized_status_counts=dict(Counter(n['status'] for n in normalized)),normalized_records=normalized)
    return states,patterns,summary


def attach_assets(states,patterns,run,output):
    """Link exact sampled JPEGs, including departure evidence, for local human review."""
    import os
    checks={c['check_id']:{f['timestamp']:f for f in c['frames']} for c in json.loads((run/'incident_check_plan.json').read_text())['checks']}
    for row in states+patterns:
        frames=checks[row['check_id']]
        row['evidence_frames']=[dict(timestamp=t,path=os.path.relpath(run/frames[t]['path'],output),sha256=frames[t]['sha256']) for t in row['evidence_timestamps']]
