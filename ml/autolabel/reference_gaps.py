"""Retain user observations and flag references not recovered by local visual proposals."""
from .review import ALIASES,bounds
from .schema import ACTIONS

EXTRA={'cash_received_and_unpacked':{'cash_received','cash_unpacking'},
       'safe_closing_and_locking':{'safe_closing','safe_locking'},
       'leave_workstation_take_keys':{'leave_workstation','key_handling'},
       'document_presented_and_inspected':{'document_transfer','document_handling'},
       'safe_locking_keys_taken':{'safe_locking','key_handling'},
       'counted_cash_returned_to_safe':{'cash_placed'},'documents_stored':{'document_handling'}}


def collect(plan,proposals):
    references=[];queue=[]
    for s in plan['scans']:
        group=s['group_id'];duration=s['duration_sec']
        for o in plan['observations'][group]:
            if o['status']=='quarantined':continue
            span=bounds(o,duration)
            if not span:continue
            a,b=span;start=max(0,a-1);end=min(duration,max(a+1,b)+1)
            actions=EXTRA.get(o['action'],{ALIASES.get(o['action'],o['action'])})
            # Approximate source times use context; a match is corroboration to review, not confirmation.
            ids=[p['proposal_id'] for p in proposals if p['group_id']==group and p['observed_action'] in actions and p['start_sec']<end and p['end_sec']>start]
            row=dict(reference_id=o['observation_id'],video=s['video'],group_id=group,split='train',
                     start_sec=start,end_sec=end,source_start_sec=o['start_sec'],source_end_sec=o.get('end_sec'),
                     source_action=o['action'],canonical_action_candidates=sorted(actions & set(ACTIONS)),source_observation=o['observation'],source_status=o['status'],
                     source_authority=o['authority'],source_mapping=o['source_mapping'],
                     source_lines=o.get('source_lines'),timing_precision=o.get('timing_precision','approximate'),
                     matching_proposal_ids=ids,model_support_status='candidate_support_not_confirmation' if ids else 'not_recovered_by_model',
                     source_priority_preserved=True,eligible_for_strong_event_training=False)
            references.append(row)
            if not ids:
                queue.append(dict(video=s['video'],group_id=group,start_sec=start,end_sec=end,context_start_sec=max(0,start-2),
                    context_end_sec=min(duration,end+2),proposal_ids=[],conflict_ids=[],reasons=['user_reference_not_recovered_by_model'],
                    observations=[o],organizer_labels=plan['reviews'][group],aliases=plan['registry'][s['video']]['aliases'],source_reference=True))
    return references,queue
