"""Compare actions on sparse human references, preserving ontology gaps as misses.
A state is not an action; loading a counter is not counting; closing is not locking.
No human label or organizer answer enters inference.
"""
import json
from .context import ACTIONS
from ml.autolabel.dataset import read_jsonl

STATE_CODES={'cash_visible','safe_open_visible'}
UNSPECIFIED_CODES={'other_observed_action','unclear_action'}
FAMILIES={
    'cash_counting':'cash_counting','cash_counter_used':'cash_counting','cash_counted_by_hand':'cash_counting',
    'cash_bundling':'cash_bundle_handling','cash_bundle_split':'cash_bundle_handling',
    'cash_detector_check':'authenticity_device_use','banknote_detector_used':'authenticity_device_use',
    'drink_at_workstation':'drink_handling','drink_used':'drink_handling',
}

def family(action):return FAMILIES.get(action,action)

def partition(human):
    actions=[];states=[];unspecified=[]
    vocabulary={family(a) for a in ACTIONS}
    for r in human:
        code=r['observed_action']
        if code in STATE_CODES:states.append(r)
        elif code in UNSPECIFIED_CODES:unspecified.append(r)
        else:actions.append(dict(r,comparison_action_family=family(code),model_vocabulary_covered=family(code) in vocabulary))
    return actions,states,unspecified

def overlap(a,b):return max(0,min(a['end_sec'],b['end_sec'])-max(a['start_sec'],b['start_sec']))

def protocol_audit(blind,folder):
    rows=[]
    for path in sorted(folder.glob('independent_??.json')):
        plain=json.loads(path.read_text());context=json.loads((folder/path.name.replace('independent','focus')).read_text())
        if plain['frames']!=context['frames'] or plain['model_revision']!=context['model_revision']:raise ValueError('Ablation inputs differ')
        def runtime(r):
            ref=r.get('runtime_ref')
            if not ref:return None
            record=json.loads((blind/ref).read_text())
            return {k:record.get(k) for k in ['dtype','attention_backend','processor_min_pixels','processor_max_pixels']}
        a,b=runtime(context),runtime(plain)
        if a!=b:raise ValueError('Ablation arithmetic or image-processing protocols differ; do not combine')
        rows.append(dict(window=path.stem.rsplit('_',1)[-1],same_frames=True,same_model_revision=True,
          runtime=a,runtime_validation='explicit_per_response' if a else 'legacy_same_preserved_completed_chain_no_per_response_runtime',
          with_history_seconds=context.get('seconds'),without_history_seconds=plain.get('seconds'),
          with_history_input_tokens=context.get('input_tokens'),without_history_input_tokens=plain.get('input_tokens')))
    return rows

def compare(blind,mapping,human):
    actions,states,unspecified=partition(human);per_video=[];matches=[];excluded=[]
    for row in mapping:
        folder=blind/row['group_id'];done=folder/'complete.json'
        if not done.exists() or not json.loads(done.read_text())['ablation']:continue
        windows=json.loads((folder/'focus_plan.json').read_text())
        def in_scope(r):
            return r['group_id']==row['group_id'] and any(min(r['end_sec'],w['context_end_sec'])>max(r['start_sec'],w['context_start_sec']) for w in windows)
        relevant=[r for r in actions if in_scope(r)]
        for r in states+unspecified:
            if in_scope(r):excluded.append(dict(task_id=r['task_id'],video=row['video'],observed_action=r['observed_action'],reason='state_reference_not_action' if r['observed_action'] in STATE_CODES else 'unspecified_action_requires_classification'))
        context=[o for o in read_jsonl(folder/'visual_observations.jsonl') if o['kind']=='event' and o['stage'].startswith('focus')]
        independent=[o for o in read_jsonl(folder/'independent_observations.jsonl') if o['kind']=='event']
        counts={}
        for mode,events in [('without_history',independent),('with_history',context)]:
            found=0
            for h in relevant:
                support=[o['observation_id'] for o in events if family(o['action'])==family(h['observed_action']) and overlap(h,o)/max(.001,min(h['end_sec']-h['start_sec'],o['end_sec']-o['start_sec']))>=.2]
                found+=bool(support)
                matches.append(dict(video=row['video'],group_id=row['group_id'],human_task_id=h['task_id'],action=h['observed_action'],
                  start_sec=h['start_sec'],end_sec=h['end_sec'],condition=mode,recovered=bool(support),supporting_proposal_ids=support,
                  comparison_action_family=family(h['observed_action']),model_vocabulary_covered=h['model_vocabulary_covered']))
            counts[mode]=dict(proposed_actions=len(events),confirmed_records_recovered=found,confirmed_records_in_supplied_windows=len(relevant),
                reference_recovery_fraction=found/len(relevant) if relevant else None,
                references_outside_model_vocabulary=sum(not h['model_vocabulary_covered'] for h in relevant))
        per_video.append(dict(video=row['video'],group_id=row['group_id'],protocol_windows=protocol_audit(blind,folder),**counts))
    totals={m:dict(recovered=sum(v[m]['confirmed_records_recovered'] for v in per_video),references=sum(v[m]['confirmed_records_in_supplied_windows'] for v in per_video),
                  references_outside_model_vocabulary=sum(v[m]['references_outside_model_vocabulary'] for v in per_video)) for m in ['without_history','with_history']}
    return dict(per_video=per_video,totals=totals,matches=matches,excluded_reference_types=excluded,comparison_families=FAMILIES,
      protocol='Same current frames, model revision, decoding settings and general rules; only previous visual memory differs. Shared survey selects windows.',
      excluded_metrics=['precision','event_mAP','whole_video_accuracy'],
      limitations=['Sparse non-independent human subset; this is reference recovery, not full event recall or accuracy.',
        'Unsupported action types such as counter loading or safe locking remain in the denominator as ontology gaps, not normal actions.',
        'Visibility states and unspecified other actions remain separate human sources; no invented action type.',
        'Counting is evaluated as a broad action family; loading is not counting and safe closing is not safe locking.',
        'No negative labels for unreviewed proposals; unmatched proposals cannot be called false positives.',
        'Coarse temporal-overlap criterion is not a boundary-quality metric.'])
