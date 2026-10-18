"""Read-only, original-split audit for Gemma adaptation. Never promotes weak labels.

python -m ml.gemma_lab.adaptation.audit_data --output runs/gemma_lab/adaptation_v1
Writes new files only. No model, GPU, server, or human-review store is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from ml.autolabel.dataset import registry, csv_rows, digest, read_jsonl
from ml.autolabel.sampling import metadata
from ml.autolabel.object_frames import validate_decision
from ml.export_human_dataset import _record, _deduplicate
from ml.human_review import ACTIONS, HumanReview
from ml.narration_training.prepare import candidates, valid_interval

PAIR_PATH = 'data/autolabel/user_pairs_v1/user_video_text_pairs.jsonl'
ORGANIZER = 'runs/narration_training/organizer_reconciled_v1'
FIXED_INPUTS = [
    'data/training/manifest.csv', 'data/review/event_review.csv',
    'data/transcripts/observations.jsonl', 'data/transcripts/sources.json',
    PAIR_PATH, 'audio.txt',
    'data/autolabel/objects_v1/object_review_manifest.jsonl',
    'data/autolabel/objects_v1/summary.json',
    *['data/autolabel/detailed_train_report/' + name for name in
      ('video_timelines.jsonl', 'user_reference_alignment.jsonl', 'moments.jsonl', 'review_queue.jsonl')],
    ORGANIZER + '/source/organizer_claims_v1.jsonl',
    ORGANIZER + '/source/original_message.txt',
]


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def canonical_sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def write_new(path, value, jsonl=False):
    """Fail rather than silently replacing an existing audit or other agent's work."""
    with path.open('x', encoding='utf8') as handle:
        if jsonl:
            for row in value:
                handle.write(encoded(row).decode() + '\n')
        else:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')


def replay_journal(raw):
    """An append may be in progress: use complete lines and record the prefix hash."""
    complete = raw[:raw.rfind(b'\n') + 1]
    current, revision = {}, 0
    for line, text in enumerate(complete.splitlines(), 1):
        if not text.strip():
            continue
        record = json.loads(text)
        prior = current.get(record['task_id'])
        if record['revision'] != revision + 1 or record['version'] != (prior['version'] if prior else 0) + 1:
            raise ValueError('Human action journal revision/version chain is invalid')
        revision = record['revision']
        current[record['task_id']] = dict(record, _journal_line=line)
    return current, dict(revision=revision, journal_complete_prefix_sha256=hashlib.sha256(complete).hexdigest(),
                         complete_bytes=len(complete), incomplete_tail_bytes=len(raw) - len(complete))


def audit_human(root, by_video, durations):
    journal = root / 'data/human_review/decisions.jsonl'
    current, journal_meta = replay_journal(journal.read_bytes())
    # _tasks is read-only. Avoid __init__, store locks, snapshot refresh, export writes.
    reader = HumanReview.__new__(HumanReview)
    reader.root = root
    reader.report = root / 'data/autolabel/detailed_train_report'
    reader.store = root / 'data/human_review'
    task_rows, fingerprint = reader._tasks()
    tasks = {row['id']: row for row in task_rows}
    accepted, excluded = [], []
    for identifier, decision_with_line in sorted(current.items()):
        decision = {k: v for k, v in decision_with_line.items() if k != '_journal_line'}
        reasons = []
        if decision.get('decision') != 'accept':
            reasons.append('not_explicitly_accepted_not_a_negative_example')
        if decision.get('source_fingerprint') != fingerprint or identifier not in tasks:
            reasons.append('stale_or_missing_source_task')
        if not reasons:
            try:
                record = _record(tasks[identifier], decision, by_video, fingerprint)
                if not valid_interval(record['start_sec'], record['end_sec'], durations[record['group_id']]):
                    raise ValueError('Interval exceeds actual source video duration')
                record['provenance']['journal_reference'] = 'data/human_review/decisions.jsonl#L' + str(decision_with_line['_journal_line'])
                accepted.append(record)
            except (ValueError, KeyError) as exc:
                reasons.append(str(exc))
        if reasons:
            excluded.append(dict(source_type='human_action', source_id=identifier, group_id=decision.get('group_id'),
                                 reasons=reasons, decision=decision, usable_as_negative=False))
    accepted = _deduplicate(accepted)
    for row in accepted:
        row.update(annotation_status='human_confirmed', human_confirmed=True,
                   action_label_ru=dict(ACTIONS)[row['observed_action']],
                   context_start_sec=max(0., row['start_sec'] - 3.),
                   context_end_sec=min(row['duration_sec'], row['end_sec'] + 3.),
                   target_text=row['note'].strip() or dict(ACTIONS)[row['observed_action']],
                   boundary_status='human_accepted_interval_not_independently_retimed',
                   interval_precision='not_recorded_by_review_interface',
                   confidence=None, confidence_semantics='no_calibrated_confidence_recorded',
                   window_violation_target=None, omissions_are_negative=False,
                   supervision_scope='visible_action_only')
    return accepted, excluded, dict(**journal_meta, source_fingerprint=fingerprint,
                                   reviewed_current=len(current), tasks_available=len(tasks),
                                   decisions=dict(Counter(x['decision'] for x in current.values())))


def audit_weak(root, reg, durations, protected):
    source = json.loads((root / 'data/transcripts/sources.json').read_text())
    if source['source_sha256'] != protected['audio.txt']:
        raise ValueError('Original transcript source changed')
    observations = read_jsonl(root / 'data/transcripts/observations.jsonl')
    sections = {x['section']: x for x in source['sections']}
    accepted, excluded, seen = [], [], set()
    for line, pair in enumerate(read_jsonl(root / PAIR_PATH), 1):
        reasons = []
        try:
            index = int(pair['observation_id'].removeprefix('observation_'))
            obs = observations[index]
            video = reg[pair['video']]
            section = sections[obs['section']]
            duration = durations[video['group_id']]
            checks = {
                'unique_pair_id': pair['pair_id'] not in seen,
                'matching_video_group': pair['video'] == obs['video'] and pair['group_id'] == obs['group_id'] == video['group_id'],
                'frozen_train_only': pair['split'] == video['split'] == 'train',
                'exact_original_description': pair['caption'] == obs['observation'],
                'exact_original_action': pair['action_from_source'] == obs['action'],
                'exact_original_times': (pair['observation_start_sec'], pair['observation_end_sec']) == (obs['start_sec'], obs['end_sec']),
                'exact_source_lines': pair['source_lines'] == obs['source_lines'] and section['start_line'] <= pair['source_lines'][0] <= pair['source_lines'][1] <= section['end_line'],
                'exact_source_file': pair['source'] == obs['source'] == 'audio.txt',
                'source_video_mapping': section['video'] == obs['video'] and pair['source_mapping'] == section['mapping'],
                'not_quarantined': pair['annotation_status'] == obs['status'] and obs['status'] != 'quarantined',
                'valid_actual_time_bounds': valid_interval(pair['start_sec'], pair['end_sec'], duration),
                'original_weak_only': pair.get('eligible_for_strong_event_training') is False and pair.get('supervision') == 'weak_user_video_text_pair',
            }
            reasons.extend(key for key, ok in checks.items() if not ok)
            seen.add(pair['pair_id'])
        except (KeyError, ValueError, IndexError, TypeError) as exc:
            reasons.append('invalid_source_reference:' + str(exc))
        if reasons:
            excluded.append(dict(source_type='weak_pair', source_id=pair.get('pair_id'), reasons=reasons))
            continue
        actions, objects, participants, ambiguous, cardinality, verbs, inference = candidates(pair)
        accepted.append(dict(pair, video_sha256=video['sha256'], source_path='data/raw/open/' + pair['video'],
            duration_sec=duration, context_start_sec=max(0., pair['start_sec'] - 3.),
            context_end_sec=min(duration, pair['end_sec'] + 3.),
            action_candidates=actions, object_candidates=objects, participant_candidates=participants,
            action_cardinality=cardinality, ambiguous_object_or_action=ambiguous,
            contains_violation_inference=inference, annotation_status='weak_transcribed_not_human_confirmed',
            human_confirmed=False, window_violation_target=None, confidence=None,
            object_visibility='not_visually_verified', provenance_checks=checks,
            source_reference=f'audio.txt#L{pair["source_lines"][0]}-L{pair["source_lines"][1]}',
            observation_reference=f'data/transcripts/observations.jsonl#L{index + 1}',
            pair_reference=PAIR_PATH + '#L' + str(line), source_sha256=source['source_sha256'],
            source_mapping_confirmation_is_action_confirmation=False,
            eligible_for_supervised_action_gt=False, eligible_for_weak_text_alignment=not inference))
    quarantined = [dict(x, observation_reference=f'data/transcripts/observations.jsonl#L{i + 1}',
                        eligible_for_training=False) for i, x in enumerate(observations) if x['status'] == 'quarantined']
    return accepted, excluded, quarantined


def audit_objects(root, reg, durations):
    catalog = {r['image_id']: r for r in read_jsonl(root / 'data/autolabel/objects_v1/object_review_manifest.jsonl')}
    # SQLite read-only transaction captures latest committed decisions and frames consistently.
    with sqlite3.connect((root / 'data/human_review/objects.sqlite').as_uri() + '?mode=ro', uri=True, timeout=10) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        decisions = [json.loads(raw) for raw, in db.execute('SELECT payload FROM decisions ORDER BY image_id')]
        frames = [json.loads(raw) for raw, in db.execute('SELECT payload FROM frames ORDER BY image_id')]
        db.rollback()
    catalog.update({r['image_id']: r for r in frames})
    accepted, excluded = [], []
    for decision in decisions:
        reasons = []
        try:
            frame = catalog[decision['image_id']]
            source = reg[frame['video']]
            if (frame['group_id'], frame['split'], frame['source_sha256']) != (source['group_id'], 'train', source['sha256']) or source['split'] != 'train':
                raise ValueError('Frame identity or frozen train membership mismatch')
            if not math.isfinite(frame['timestamp']) or not 0 <= frame['timestamp'] <= durations[frame['group_id']]:
                raise ValueError('Frame timestamp outside real video')
            if not validate_decision(decision, frame) or decision.get('eligible_for_detector_training') is not True:
                raise ValueError('Incomplete/uncertain annotation is not a negative example')
            if decision.get('annotation_origin') != 'explicit_human_decision':
                raise ValueError('Missing explicit human provenance')
            base = root / ('data/human_review' if frame.get('origin') == 'selected_video_frame' else 'data/autolabel/objects_v1')
            image = (base / frame['image']).resolve()
            if not image.is_relative_to((base / ('frames' if frame.get('origin') == 'selected_video_frame' else 'images')).resolve()):
                raise ValueError('Frame image path escapes reviewed image directory')
            if digest(image) != frame['image_sha256']:
                raise ValueError('Reviewed frame image SHA changed')
            accepted.append(dict(frame, image_path=str(image.relative_to(root)), human_decision=decision,
                                 label_scope='objects_in_single_frame', annotation_status='human_confirmed',
                                 human_confirmed=True, eligible_for_detector_training=True,
                                 eligible_for_violation_training=False, confidence=None,
                                 provenance=dict(source='data/human_review/objects.sqlite', table='decisions',
                                                 row_id=decision['image_id'], version=decision['version'],
                                                 decision_sha256=canonical_sha(decision))))
        except (KeyError, ValueError, OSError, TypeError) as exc:
            reasons.append(str(exc))
        if reasons:
            excluded.append(dict(source_type='human_object', source_id=decision.get('image_id'), reasons=reasons,
                                 usable_as_negative=False))
    return accepted, excluded, dict(current_decisions=len(decisions), committed_snapshot_sha256=canonical_sha(dict(decisions=decisions, frames=frames)))


def run(root: Path, output: Path):
    root = root.resolve()
    output = output.resolve()
    if not output.is_relative_to(root / 'runs/gemma_lab'):
        raise ValueError('Outputs must be under runs/gemma_lab')
    output.mkdir(parents=True, exist_ok=True)
    protected = {name: digest(root / name) for name in FIXED_INPUTS}
    reg, _, _ = registry(root)  # verifies all actual video bytes, including exact duplicate aliases
    manifest = csv_rows(root / 'data/training/manifest.csv')
    by_video = {row['video']: {k: row[k] for k in ('video', 'group_id', 'split', 'sha256')} for row in manifest}
    groups = sorted({x['group_id'] for x in reg.values()})
    durations, video_targets, duplicates = {}, [], []
    for group in groups:
        row = min((r for r in reg.values() if r['group_id'] == group), key=lambda r: r['video'])
        # Decode only container metadata; audio is never opened or analyzed.
        duration = metadata(root / 'data/raw/open' / row['video'])['duration_sec']
        durations[group] = duration
        labels = {int(x['target']) for x in manifest if x['group_id'] == group}
        if len(labels) != 1 or not labels.issubset({0, 1}):
            raise ValueError('Inconsistent original whole-video label')
        video_targets.append(dict(video=row['video'], aliases=row['aliases'], group_id=group, split=row['split'],
                                  source_path='data/raw/open/' + row['video'], video_sha256=row['sha256'],
                                  duration_sec=duration, label=labels.pop(), label_scope='whole_video',
                                  label_source='organizer_original_video_label',
                                  source_reference='data/training/manifest.csv',
                                  source_sha256=protected['data/training/manifest.csv'],
                                  window_violation_target=None, independent_test=False,
                                  evaluation_role='reused_diagnostic_not_independent_test' if row['split'] == 'val' else 'train_group'))
        if len(row['aliases']) > 1:
            duplicates.append(dict(group_id=group, split=row['split'], sha256=row['sha256'], aliases=row['aliases']))
    human, human_exclusions, human_meta = audit_human(root, by_video, durations)
    weak, weak_exclusions, quarantine = audit_weak(root, reg, durations, protected)
    objects, object_exclusions, object_meta = audit_objects(root, reg, durations)
    claims, disclosed, organizer_exclusions = [], [], []
    organizer_file = ORGANIZER + '/source/organizer_claims_v1.jsonl'
    for line, claim in enumerate(read_jsonl(root / organizer_file), 1):
        source = reg.get(claim.get('video'))
        if not source or any(claim.get(key) != source.get(alias) for key, alias in
                             [('group_id', 'group_id'), ('split', 'split'), ('video_sha256', 'sha256')]):
            organizer_exclusions.append(dict(source_type='organizer_claim', source_id=claim.get('claim_id'), reasons=['source_identity_mismatch']))
            continue
        row = dict(claim, label_scope='whole_video_scenario_claim', annotation_status='organizer_provided_not_visual_gt',
                   source_reference=organizer_file + '#L' + str(line),
                   source_path='data/raw/open/' + claim['video'],
                   eligible_for_supervised_action_gt=False, eligible_for_clip_caption_alignment=False,
                   eligible_for_training=claim['split'] == 'train',
                   confidence=None, window_violation_target=None)
        (claims if claim['split'] == 'train' else disclosed).append(row)
    exclusions = human_exclusions + weak_exclusions + object_exclusions + organizer_exclusions
    # Candidate overlap is for human review; different actions can coexist, so never invent conflicts.
    overlaps = []
    for pair in weak:
        matches = [h for h in human if h['group_id'] == pair['group_id'] and min(h['end_sec'], pair['end_sec']) > max(h['start_sec'], pair['start_sec'])]
        if matches:
            overlaps.append(dict(pair_id=pair['pair_id'], human_task_ids=[h['task_id'] for h in matches],
                                 rule='human_target_takes_precedence; coexistence_does_not_imply_conflict'))
    # Immutable source files must not change during audit. Human stores remain live and are snapshotted above.
    if protected != {name: digest(root / name) for name in FIXED_INPUTS}:
        raise ValueError('Protected source changed during audit; rerun into a new version')
    journal_latest = (root / 'data/human_review/decisions.jsonl').read_bytes()
    if hashlib.sha256(journal_latest[:human_meta['complete_bytes']]).hexdigest() != human_meta['journal_complete_prefix_sha256']:
        raise ValueError('Authoritative human journal prefix changed during audit')
    results = {
        'human_actions_train.jsonl': human, 'weak_pairs_train.jsonl': weak,
        'human_objects_train.jsonl': objects, 'organizer_claims_train.jsonl': claims,
        'video_level_targets.jsonl': video_targets, 'disclosed_validation.jsonl': disclosed,
        'quarantined_excluded.jsonl': quarantine, 'exclusions.jsonl': exclusions,
        'human_weak_overlaps.jsonl': overlaps,
    }
    counts = dict(original_manifest_rows=len(manifest), original_groups=len(groups), raw_files=len(reg),
                  original_group_splits=dict(Counter(x['split'] for x in video_targets)),
                  confirmed_action_intervals=len(human), confirmed_action_groups=len({x['group_id'] for x in human}),
                  confirmed_action_classes=dict(Counter(x['observed_action'] for x in human)),
                  confirmed_action_groups_detail=dict(Counter(x['group_id'] for x in human)),
                  weak_pairs=len(weak), weak_groups=len({x['group_id'] for x in weak}),
                  weak_cardinality=dict(Counter(x['action_cardinality'] for x in weak)),
                  weak_ambiguous=sum(x['ambiguous_object_or_action'] for x in weak),
                  weak_approximate_boundaries=sum(x['timing_precision'] == 'approximate' for x in weak),
                  human_object_frames=len(objects), human_boxes=sum(len(x['human_decision']['boxes']) for x in objects),
                  human_box_classes=dict(Counter(b['class'] for x in objects for b in x['human_decision']['boxes'])),
                  explicitly_reviewed_object_negative_frames=sum(x['human_decision']['image_assessment'] == 'confirmed_no_target' for x in objects),
                  organizer_train_claims=len(claims), organizer_disclosed_val_groups=len(disclosed),
                  quarantined_observations=len(quarantine), excluded_records=len(exclusions),
                  confirmed_violation_events=sum(x['violation_confirmed'] for x in human))
    audit = dict(schema_version=1, created_at=dt.datetime.now(dt.timezone.utc).isoformat(), counts=counts,
                 original_split_only=True, active_split_ignored=True, protected_input_sha256=protected,
                 source_video_sha256_verified=True, duplicate_aliases=duplicates,
                 human_journal_snapshot=human_meta, human_objects_snapshot=object_meta,
                 sources_changed=False, source_files_modified=False,
                 policy=dict(weak_descriptions_are_ground_truth=False, organizer_claims_are_frame_labels=False,
                             unlabeled_or_rejected_are_negative=False, video_verdict_is_window_target=False,
                             validation_is_independent_test=False, evidence_of_legal_violation=False),
                 training_consumers_must_exclude_all_inner_heldout_groups_across_all_sources=True,
                 human_review_fields=sorted(human[0]) if human else [],
                 evaluation_limits=['8 original validation groups are repeatedly used diagnostics, not an independent test.',
                                    'Sparse accepted actions support matched-reference recovery, not complete event precision/recall.',
                                    'No human-confirmed violation-event set exists.',
                                    'Accepted event bounds may be inherited from suggestions; independent retiming is not recorded.',
                                    'Object no-target applies only to explicitly reviewed target classes in that frame.'])
    for name, rows in results.items():
        write_new(output / name, rows, jsonl=True)
    audit['output_sha256'] = {name: digest(output / name) for name in results}
    write_new(output / 'audit.json', audit)
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)


if __name__ == '__main__':
    main()
