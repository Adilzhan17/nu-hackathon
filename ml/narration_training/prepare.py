"""Audit immutable narration pairs and prepare source-level bags, never window verdicts."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

from ml.autolabel.dataset import registry, read_jsonl, snapshot, digest, csv_rows
from ml.autolabel.sampling import metadata

PAIR_PATH = 'data/autolabel/user_pairs_v1/user_video_text_pairs.jsonl'
MODEL_ID = 'google/siglip2-so400m-patch14-384'
MODEL_REVISION = 'e8e487298228002f3d8a82e0cd5c8ea9c567f57f'

# Deterministic interpretation of the existing source taxonomy, NOT visual verification.
DECOMPOSE = {
 'cash_removed_and_safe_closed': ['cash_removed','safe_closing'],
 'cash_received_and_unpacked': ['cash_received','cash_unpacking'],
 'document_presented_and_inspected': ['document_transfer','document_checked'],
 'document_and_cash_on_desk': ['document_placed','cash_placed'],
 'cash_preparation_and_loading': ['cash_preparation','cash_counter_loading'],
 'safe_closing_and_locking': ['safe_closing','safe_locking'],
 'safe_locking_keys_taken': ['safe_locking','key_handling'],
 'safe_key_manipulation': ['key_handling'],
 'visitor_enters': ['person_enters'], 'second_visitor_enters': ['person_enters'],
 'visitor_leaves': ['person_leaves'], 'room_exit': ['person_leaves'],
 'person_leaves_view': ['person_leaves_view'],
 'document_presented': ['document_transfer'], 'document_returned': ['document_transfer'],
 'document_inspection': ['document_checked'], 'documents_stored': ['document_stored'],
 'stamps_stored': ['stamp_stored'], 'safe_door_closing': ['safe_closing'],
 'cash_counter_operation': ['cash_counting'], 'cash_placed_on_desk': ['cash_placed'],
 'counted_cash_bundling': ['cash_bundling'], 'cash_bundle_split': ['cash_bundle_split'],
 'partial_cash_return': ['cash_transferred'], 'cash_presented': ['cash_presented'],
 'unclear_handheld_item': [], 'unclear_safe_interaction': [],
}
OBJECT_TERMS = {
 'cash': r'деньг|денег|банкнот|купюр|пачк', 'document': r'документ|удостоверен',
 'key': r'ключ', 'stamp': r'печат|штамп', 'safe': r'сейф',
 'cash_counter': r'сч[её]тн\w* машин|сч[её]тчик', 'computer': r'компьютер|ноутбук',
 'phone': r'телефон', 'drink': r'стакан|напит|пь[её]т', 'notebook': r'блокнот|тетрад',
}
VERBS = r'открыва\w*|закрыва\w*|запира\w*|бер[её]т|доста[её]т|клад[её]т|убира\w*|переда[её]т|возвраща\w*|счита\w*|пересчитыва\w*|загружа\w*|распаковыва\w*|осматрива\w*|проверя\w*|уход\w*|выход\w*|подход\w*'

def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(x, ensure_ascii=False, allow_nan=False)+'\n' for x in rows))

def new_output(root, output):
    output = Path(output).resolve()
    if not output.is_relative_to((root/'runs/narration_training').resolve()):
        raise ValueError('Outputs must be under runs/narration_training')
    output.mkdir(parents=True, exist_ok=False)
    return output

def valid_interval(start, end, duration):
    return all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
               for v in (start,end,duration)) and 0 <= start < end <= duration + 1e-6

def candidates(pair):
    code = pair['action_from_source']; caption = pair['caption'].lower()
    actions = DECOMPOSE.get(code, [code])
    objects = [k for k, pattern in OBJECT_TERMS.items() if re.search(pattern,caption)]
    ambiguous = bool(re.search(r'неяс|неразлич|непонят|возмож|похож|вероят|не видно|то ли|какой-то|какой то',caption)) or code.startswith('unclear_')
    verbs = sorted(set(re.findall(VERBS,caption)))
    cardinality = 'multiple_candidates' if len(actions)>1 else ('unclear' if not actions else
                  ('possibly_multiple' if len(verbs)>1 else 'single_candidate'))
    # Roles are claims from the description, never identities or visually verified attributes.
    participants = ['person_role_unspecified']
    if re.search(r'сотрудник|кассир|оператор',caption): participants = ['employee_candidate']
    if re.search(r'клиент|посетител',caption): participants += ['visitor_candidate']
    inference = bool(re.search(r'наруш|не провер|без провер|не запер|не закрыл|должен|запрещ',caption))
    return actions, objects, participants, ambiguous, cardinality, verbs, inference


def resolve_prepare_split(root, split_version=None, original_split=False):
    """Resolve an explicit version or the active policy, retaining original provenance."""
    root=Path(root).resolve()
    if original_split and split_version:
        raise ValueError('--original-split and --split-version are mutually exclusive')
    active=root/'data/training/active_split.json'
    pointer=None
    if not original_split and split_version is None and active.exists():
        pointer=json.loads(active.read_text())
        split_version=pointer.get('version')
        if not split_version:raise ValueError('Active split version is missing')
    report=None
    if split_version:
        from ml.split95 import load_split
        report=load_split(root,version=split_version)
        if pointer is not None:
            if pointer.get('manifest')!=report['manifest_relative_path'] or pointer.get('source_manifest_sha256')!=report['source_manifest_sha256']:
                raise ValueError('Active split identity differs from verified split version')
            if pointer.get('requires_diagnostic_test_mode') is not True or pointer.get('diagnostic_only') is not True or pointer.get('independent_test') is not False:
                raise ValueError('Active split evaluation policy is invalid')
    source=csv_rows(root/'data/training/manifest.csv')
    identities={}
    for row in source:
        identity=(row['sha256'],row['split'],int(row['target']))
        if row['group_id'] in identities and identities[row['group_id']]!=identity:
            raise ValueError('Original group identity is inconsistent')
        identities[row['group_id']]=identity
    membership={group:(report['group_membership'][group] if report else identity[1]) for group,identity in identities.items()}
    policy=dict(schema_version=1,version=report['version'] if report else 'original_32_8',
        original_manifest='data/training/manifest.csv',original_manifest_sha256=digest(root/'data/training/manifest.csv'),
        canonical_group_membership=membership,
        source_identities={g:dict(sha256=v[0],original_split=v[1],video_target=v[2]) for g,v in identities.items()},
        split_counts=dict(Counter(membership.values())),
        storage_mapping={'train':'train','diagnostic_test':'val','val':'val'},
        validation_role='reused_diagnostic_not_independent_test',independent_test=False,
        selection_uses_train_only=True,original_weak_pair_provenance_preserved=True,
        heldout_captions_used=False,promoted_old_val_captions_added=False,
        prepared_feature_cache_required='new_extraction_for_new_dataset_hashes' if report else 'match_exact_dataset_hashes')
    if report:
        policy['version_report_sha256']=digest(root/f'data/training/{report["version"]}/report.json')
    return policy


def validate_prepared_split(root, dataset, bags=None, pairs=None, clips=None):
    """Check grouped coverage and pair authority independently of cached features."""
    root=Path(root).resolve();dataset=Path(dataset)
    path=dataset/'split_policy.json'
    if path.exists():
        policy=json.loads(path.read_text())
        expected=resolve_prepare_split(root,original_split=True) if policy.get('version')=='original_32_8' else resolve_prepare_split(root,split_version=policy.get('version'))
        if policy!=expected:raise ValueError('Prepared split policy does not match frozen source membership')
    else:
        # Existing dataset_v1 remains reproducible regardless of a later active policy.
        policy=resolve_prepare_split(root,original_split=True)
    bags=read_jsonl(dataset/'bags.jsonl') if bags is None else bags
    pairs=read_jsonl(dataset/'pairs.jsonl') if pairs is None else pairs
    clips=read_jsonl(dataset/'clips.jsonl') if clips is None else clips
    groups={b['group_id']:b for b in bags}
    if len(groups)!=len(bags) or set(groups)!=set(policy['source_identities']):
        raise ValueError('Prepared bags do not cover the frozen source groups exactly once')
    seen_hashes=set()
    for group,bag in groups.items():
        identity=policy['source_identities'][group];partition=policy['canonical_group_membership'][group]
        if bag['sha256']!=identity['sha256'] or bag['sha256'] in seen_hashes:
            raise ValueError('Prepared bags contain changed bytes or duplicate source groups')
        if bag['video_target']!=identity['video_target']:raise ValueError('Whole-video source target changed')
        seen_hashes.add(bag['sha256'])
        if bag['split']!=policy['storage_mapping'][partition]:raise ValueError('Prepared bag uses the wrong partition')
        if policy['version']!='original_32_8' and (bag.get('partition')!=partition or bag.get('original_split')!=identity['original_split']):
            raise ValueError('Versioned bag lost original/effective split provenance')
    coverage=set()
    for clip in clips:
        bag=groups.get(clip['group_id'])
        if not bag or clip['split']!=bag['split']:raise ValueError('Coverage clip crosses source partition')
        if clip['kind']=='video_bag_window':coverage.add(clip['group_id'])
        elif clip['kind']=='weak_narration_interval' and clip['split']!='train':raise ValueError('Held-out narration interval in prepared coverage')
    if coverage!=set(groups):raise ValueError('A source video lacks coverage windows')
    for pair in pairs:
        group=pair['group_id'];identity=policy['source_identities'].get(group)
        if not identity or identity['original_split']!='train' or groups[group]['split']!='train' or pair['split']!='train':
            raise ValueError('Narration caption is outside original and effective train membership')
        if pair['video_sha256']!=identity['sha256'] or not all(pair['provenance_checks'].values()):
            raise ValueError('Narration original provenance failed')
    return policy


def prepare(root, output, split_version=None, original_split=False):
    root=Path(root).resolve(); out=new_output(root,output)
    split_policy=resolve_prepare_split(root,split_version,original_split)
    reg, reviews, grouped_obs = registry(root)
    obs={o['observation_id']:o for rows in grouped_obs.values() for o in rows}
    sources=json.loads((root/'data/transcripts/sources.json').read_text())
    sections={s['section']:s for s in sources['sections']}
    protected=snapshot(root)
    # Freeze hashes of all pre-existing autolabel files; frames remain local.
    old_autolabel={str(p.relative_to(root)):digest(p) for p in sorted((root/'data/autolabel').rglob('*')) if p.is_file()}
    protected[PAIR_PATH]=digest(root/PAIR_PATH)
    protected['audio.txt']=digest(root/'audio.txt')
    if split_policy['version']!='original_32_8':
        directory=root/'data/training'/split_policy['version']
        protected.update({str(p.relative_to(root)):digest(p) for p in directory.iterdir() if p.is_file()})
    if protected['audio.txt'] != sources['source_sha256']:
        raise ValueError('Transcript source hash changed')
    durations={}; bags=[]
    manifest=csv_rows(root/'data/training/manifest.csv')
    for group in sorted({r['group_id'] for r in reg.values()}):
        r=min((r for r in reg.values() if r['group_id']==group),key=lambda r:r['video'])
        duration=metadata(root/'data/raw/open'/r['video'])['duration_sec']
        for alias in r['aliases']: durations[alias]=duration
        labels={int(x['target']) for x in manifest if x['group_id']==group}
        if len(labels)!=1: raise ValueError('Inconsistent source-level verdict')
        bag=dict(**r,duration_sec=duration,video_target=labels.pop(),target_scope='whole_source_video_only')
        if split_policy['version']!='original_32_8':
            partition=split_policy['canonical_group_membership'][group]
            bag.update(original_split=r['split'],partition=partition,split=split_policy['storage_mapping'][partition])
        bags.append(bag)
    rows=[]; excluded=[]; seen=set()
    for pair in read_jsonl(root/PAIR_PATH):
        errors=[]; o=obs.get(pair['observation_id']); r=reg.get(pair['video'])
        if pair['pair_id'] in seen: errors.append('duplicate_pair_id')
        seen.add(pair['pair_id'])
        if not o or not r: errors.append('missing_source')
        else:
            section=sections[o['section']]
            checks={
             'video_group': pair['video']==o['video'] and pair['group_id']==o['group_id']==r['group_id'],
             'train_only': pair['split']==r['split']=='train',
             'caption_exact': pair['caption']==o['observation'],
             'source_action': pair['action_from_source']==o['action'],
             'observation_time': (pair['observation_start_sec'],pair['observation_end_sec'])==(o['start_sec'],o['end_sec']),
             'source_lines': pair['source_lines']==o['source_lines'] and section['start_line']<=pair['source_lines'][0]<=pair['source_lines'][1]<=section['end_line'],
             'source_file': pair['source']==o['source']=='audio.txt',
             'source_section_video': section['video']==o['video'],
             'mapping': pair['source_mapping']==section['mapping'],
             'status': pair['annotation_status']==o['status'] and o['status']!='quarantined',
             'time_bounds': valid_interval(pair['start_sec'],pair['end_sec'],durations[pair['video']]),
            }
            if split_policy['version']!='original_32_8':
                checks['effective_train_only']=split_policy['canonical_group_membership'][r['group_id']]=='train'
            anchor=o['start_sec']; end=o['end_sec'] if o['end_sec'] is not None else anchor
            expected=(max(0.,anchor-1.5), min(durations[pair['video']],max(anchor+1.5,end+1.)))
            checks['context_generation']=all(abs(a-b)<1e-6 for a,b in zip(expected,(pair['start_sec'],pair['end_sec'])))
            errors += [k for k,v in checks.items() if not v]
        if errors:
            excluded.append(dict(pair_id=pair['pair_id'],errors=errors)); continue
        actions, objects, participants, ambiguous, cardinality, verbs, inference=candidates(pair)
        questions=['Укажите начало и конец каждого видимого действия; исходное время приблизительное.']
        if cardinality in {'multiple_candidates','possibly_multiple'}:
            questions.append('Это одно действие или несколько последовательных? Разделите их временные границы.')
        if ambiguous or objects:
            questions.append('Какие из перечисленных предметов действительно различимы? Для остальных отметьте «неразличимо».')
        if inference: questions.append('Какое действие видно непосредственно, а какая часть описания является предположением о нарушении?')
        rows.append(dict(**pair,video_sha256=r['sha256'],source_section=o['section'],
            source_reference=f"audio.txt#L{pair['source_lines'][0]}-L{pair['source_lines'][1]}",
            observation_reference=f"data/transcripts/observations.jsonl#L{int(pair['observation_id'].split('_')[-1])+1}",
            source_sha256=sources['source_sha256'],duration_sec=durations[pair['video']],
            context_start_sec=max(0.,pair['start_sec']-3.),context_end_sec=min(durations[pair['video']],pair['end_sec']+3.),
            action_candidates=actions,object_candidates=objects,participant_candidates=participants,
            object_visibility='not_visually_verified',candidate_provenance='deterministic_source_taxonomy_and_lexical_rules',
            action_cardinality=cardinality,matched_verb_forms=verbs,ambiguous_object_or_action=ambiguous,
            possible_violation_inference=inference,window_violation_target=None,
            human_confirmed=False,eligible_action_positive_loss=bool(actions) and not ambiguous and not inference,
            eligible_text_loss=not inference,text_loss_weight=.25 if ambiguous else 1.,
            human_questions=questions,provenance_checks=checks))
    quarantined=[dict(observation_id=o['observation_id'],video=o['video'],group_id=o['group_id'],reason='quarantined',eligible=False)
                 for o in obs.values() if o['status']=='quarantined' and reg[o['video']]['split']=='train']
    # Short, diverse queue, with the complete questions retained on every audit row.
    queue=[]; chosen=set()
    ranking=sorted(rows,key=lambda r:(-int(r['ambiguous_object_or_action']),-int(r['action_cardinality']=='multiple_candidates'),r['pair_id']))
    for row in ranking:
        if row['group_id'] in chosen: continue
        chosen.add(row['group_id']);queue.append(row)
        if len(queue)==12: break
    clips=[]
    for bag in bags:
        duration=bag['duration_sec']; starts=[0.]
        while starts[-1]+6 < duration: starts.append(min(starts[-1]+4,max(0.,duration-6)))
        for j,start in enumerate(starts):
            clips.append(dict(clip_id=f"{bag['group_id']}_{j:03d}",video=bag['video'],group_id=bag['group_id'],split=bag['split'],
                              start_sec=start,end_sec=min(start+6,duration),kind='video_bag_window',window_target=None))
    for row in rows:
        clips.append(dict(clip_id=row['pair_id'],video=row['video'],group_id=row['group_id'],split='train',
                          start_sec=row['start_sec'],end_sec=row['end_sec'],kind='weak_narration_interval',window_target=None))
    vocab=sorted({a for r in rows for a in r['action_candidates']})
    summary=dict(input_pairs=len(seen),valid_pairs=len(rows),excluded_pairs=len(excluded),quarantined_source_observations=len(quarantined),
      pair_groups=len({r['group_id'] for r in rows}),action_cardinality=dict(Counter(r['action_cardinality'] for r in rows)),
      ambiguous=sum(r['ambiguous_object_or_action'] for r in rows),approximate=sum(r['timing_precision']=='approximate' for r in rows),
      eligible_text_pairs=sum(r['eligible_text_loss'] for r in rows),eligible_action_pairs=sum(r['eligible_action_positive_loss'] for r in rows),
      inference_claim_pairs=sum(r['possible_violation_inference'] for r in rows),human_confirmed_actions=0,
      bag_splits=dict(Counter(b['split'] for b in bags)),review_queue=len(queue),
      canonical_bag_splits=split_policy['split_counts'],split_version=split_policy['version'],
      independent_test=False,diagnostic_only=True,heldout_captions_used=False,
      promoted_old_val_captions_added=False,
      feature_cache_policy=split_policy['prepared_feature_cache_required'],
      notice='Source mapping confirmation is not event confirmation. All window verdicts unknown. No action negatives inferred from omissions.')
    for name,data in [('pairs',rows),('excluded',excluded),('quarantined_sources',quarantined),('review_queue',queue),('bags',bags),('clips',clips)]:
        write_jsonl(out/f'{name}.jsonl',data)
    write_json(out/'audit_summary.json',summary)
    write_json(out/'protected_inputs.json',protected)
    write_json(out/'protected_autolabel.json',old_autolabel)
    write_json(out/'split_policy.json',split_policy)
    write_json(out/'action_vocabulary.json',dict(actions=vocab,prompts=[f'A video of a person: {a.replace("_"," ")}' for a in vocab],source='fixed source-action taxonomy, no validation-label selection'))
    write_json(out/'model_contract.json',dict(model_id=MODEL_ID,model_revision=MODEL_REVISION,license='Apache-2.0',
      model_card=f'https://huggingface.co/{MODEL_ID}',max_parameters=3_000_000_000,local_only=True))
    validate_prepared_split(root,out,bags,rows,clips)
    print(json.dumps(summary,ensure_ascii=False))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,required=True)
    group=p.add_mutually_exclusive_group();group.add_argument('--split-version');group.add_argument('--original-split',action='store_true')
    a=p.parse_args();prepare(a.root,a.output,a.split_version,a.original_split)
