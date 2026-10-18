"""User-authored train-only video/text supervision with explicit approximate temporal scope."""
import argparse
from collections import Counter
from pathlib import Path
from .dataset import registry,snapshot
from .pipeline import inside,save,jsonl
from .sampling import metadata


def export(root,output):
    root=Path(root).resolve();output=inside(output,root/'data/autolabel')
    if output.exists():raise FileExistsError('New output directory required')
    reg,reviews,observations=registry(root);pairs=[];excluded=[];duration_cache={}
    for group,obs in observations.items():
        for o in obs:
            source=reg[o['video']]
            if source['split']!='train':continue
            if o['status']=='quarantined':
                excluded.append(dict(observation_id=o['observation_id'],reason='quarantined'));continue
            if o['start_sec'] is None:
                excluded.append(dict(observation_id=o['observation_id'],reason='missing_time'));continue
            video=o['video']
            if video not in duration_cache:duration_cache[video]=metadata(root/'data/raw/open'/video)['duration_sec']
            duration=duration_cache[video];anchor=float(o['start_sec'])
            if not 0<=anchor<duration:
                excluded.append(dict(observation_id=o['observation_id'],reason='time_outside_source'));continue
            observed_end=float(o['end_sec']) if o['end_sec'] is not None else anchor
            if observed_end<anchor or observed_end>duration:
                excluded.append(dict(observation_id=o['observation_id'],reason='invalid_end'));continue
            start=max(0.,anchor-1.5);end=min(duration,max(anchor+1.5,observed_end+1.))
            pairs.append(dict(pair_id=f'user_pair_{len(pairs):04d}',video=video,group_id=group,split='train',
                 start_sec=start,end_sec=end,caption=o['observation'],action_from_source=o['action'],
                 observation_start_sec=o['start_sec'],observation_end_sec=o['end_sec'],
                 observation_id=o['observation_id'],source=o.get('source'),source_lines=o.get('source_lines'),
                 source_mapping=o.get('source_mapping'),supervision='weak_user_video_text_pair',
                 timing_precision=o.get('timing_precision','approximate'),annotation_status=o['status'],
                 eligible_for_weak_video_text_training=True,eligible_for_strong_event_training=False,
                 unknown_actions_are_not_negative_labels=True,organizer_video_verdict=o.get('source_verdict')))
    output.mkdir(parents=True);jsonl(output/'user_video_text_pairs.jsonl',pairs);jsonl(output/'excluded.jsonl',excluded)
    summary=dict(train_pairs=len(pairs),independent_groups=len({p['group_id'] for p in pairs}),
                 source_action_counts=dict(Counter(p['action_from_source'] for p in pairs)),excluded=len(excluded),
                 protected_inputs=snapshot(root),note='User-authored weak supervision, not generated captions; expanded context is not a precise action boundary.')
    save(output/'summary.json',summary);print({k:summary[k] for k in ['train_pairs','independent_groups','excluded']})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,required=True);a=p.parse_args();export(a.root,a.output)
