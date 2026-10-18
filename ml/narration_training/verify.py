"""Verify actual run provenance, source preservation, temporal validity and fold isolation."""
import argparse
from collections import Counter
import json
from pathlib import Path
from ml.autolabel.dataset import digest, read_jsonl
from .prepare import valid_interval, write_json, validate_prepared_split

def verify(root,dataset,features,experiment):
    protected=json.loads((dataset/'protected_inputs.json').read_text())
    protected.update(json.loads((dataset/'protected_autolabel.json').read_text()))
    changed=[name for name,expected in protected.items() if not (root/name).is_file() or digest(root/name)!=expected]
    if changed: raise ValueError('Protected inputs changed: '+str(changed))
    bags=read_jsonl(dataset/'bags.jsonl');pairs=read_jsonl(dataset/'pairs.jsonl');clips=read_jsonl(dataset/'clips.jsonl')
    split_policy=validate_prepared_split(root,dataset,bags,pairs,clips)
    source=json.loads((root/'data/transcripts/sources.json').read_text())
    original_lines=(root/'audio.txt').read_text().splitlines()
    for section in source['sections']:
        if section['text'].strip()!='\n'.join(original_lines[section['start_line']-1:section['end_line']]).strip():
            raise ValueError('Source section differs from original transcript lines')
    groups={b['group_id']:b for b in bags}
    by_hash={}
    for b in bags:
        if b['sha256'] in by_hash: raise ValueError('Same bytes represented by multiple groups')
        by_hash[b['sha256']]=b['group_id']
    timestamps=json.loads((features/'frame_timestamps.json').read_text())
    for c in clips:
        bag=groups[c['group_id']]
        if c['split']!=bag['split'] or not valid_interval(c['start_sec'],c['end_sec'],bag['duration_sec']): raise ValueError('Clip scope/bounds')
        if c['window_target'] is not None: raise ValueError('Whole-video verdict leaked onto window')
        ts=timestamps[c['clip_id']]['actual_timestamps']
        if not ts or any(t<c['start_sec']-1e-6 or t>c['end_sec']+1e-6 for t in ts) or ts!=sorted(set(ts)):
            raise ValueError('Bad real frame PTS')
    for p in pairs:
        if p['human_confirmed'] or p['annotation_status']=='quarantined' or p['window_violation_target'] is not None:
            raise ValueError('Weak/strong confusion')
        if not all(p['provenance_checks'].values()): raise ValueError('Provenance check failed')
        start=p['observation_start_sec'];end=p['observation_end_sec'];duration=groups[p['group_id']]['duration_sec']
        if not 0<=start<duration or (end is not None and not start<=end<=duration):
            raise ValueError('Original observation out of bounds, even if context was clamped')
        if not 0<=p['context_start_sec']<=p['start_sec']<p['end_sec']<=p['context_end_sec']<=duration:
            raise ValueError('Context does not enclose the narration interval')
    all_train={g for g,b in groups.items() if b['split']=='train'}
    by_pair={p['pair_id']:p for p in pairs}
    usage=read_jsonl(experiment/'data_usage.jsonl')
    for u in usage:
        fit=set(u['groups']);evaluate=set(u['evaluation_groups'])
        if not fit<=all_train or fit&evaluate: raise ValueError('Train/evaluation leakage')
        ids=u['pair_ids']
        if len(ids)!=len(set(ids)) or len(ids)!=u['unique_text_pairs']: raise ValueError('Usage count mismatch')
        if any(by_pair[k]['group_id'] not in fit for k in ids): raise ValueError('Held-out narration used in fit')
        expected={p['pair_id'] for p in pairs if p['group_id'] in fit and p['eligible_text_loss']} if u['method']=='narration' else set()
        if set(ids)!=expected: raise ValueError('Dropped or unexpected pair')
    # Inner fits must not access the enclosing outer holdout, even though both are train.
    outer={u['scope']:set(u['evaluation_groups']) for u in usage if u['role']=='outer_fit'}
    for u in usage:
        if u['scope'] in outer and u['role']=='inner_selection':
            if (set(u['groups'])|set(u['evaluation_groups']))&outer[u['scope']]: raise ValueError('Nested fold leakage')
    for method in ['baseline','narration']:
        preds=read_jsonl(experiment/f'{method}_oof.jsonl')
        if Counter(p['group_id'] for p in preds)!=Counter({g:1 for g in all_train}): raise ValueError('Incomplete OOF')
        diagnostic=read_jsonl(experiment/f'{method}_reused_val.jsonl')
        expected_diagnostic={g:1 for g,b in groups.items() if b['split']=='val'}
        if Counter(p['group_id'] for p in diagnostic)!=Counter(expected_diagnostic):raise ValueError('Incomplete diagnostic source coverage')
    protocol=json.loads((experiment/'protocol.json').read_text())
    if protocol['threshold']!=.5 or protocol['threshold_selection']!='fixed_before_evaluation': raise ValueError('Threshold protocol changed')
    if split_policy['version']!='original_32_8' and protocol.get('split_policy')!=split_policy:
        raise ValueError('Training protocol omitted or changed versioned split policy')
    result=dict(status='passed',protected_files_checked=len(protected),source_groups=len(bags),pairs=len(pairs),
                clips=len(clips),training_fits_checked=len(usage),temporal_bounds=True,nested_group_isolation=True,
                weak_labels_remain_weak=True,original_artifacts_unchanged=True,source_sections_match_original_lines=len(source['sections']),
                split_version=split_policy['version'],source_split_counts=split_policy['split_counts'],independent_test=False)
    write_json(experiment/'verification.json',result);print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True);p.add_argument('--experiment',type=Path,required=True)
    a=p.parse_args();verify(a.root,a.dataset,a.features,a.experiment)
