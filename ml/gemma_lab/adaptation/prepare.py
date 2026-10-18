"""Prepare local native-resolution frames and answer-only fit/evaluation samples.

No GPU, network, audio decoding, external API, human store writes or pseudo-GT.
python -m ml.gemma_lab.adaptation.prepare --run runs/gemma_lab/adaptation_v1
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import av

from .taxonomy import ACTION_CODES
from .audit_data import canonical_sha, write_new


ACTION_VOCAB = ', '.join(ACTION_CODES)
# The weak-caption task retains original source candidate names. Its generic
# vocabulary is separate from the confirmed human action taxonomy.
WEAK_EXTRA_CODES = ('laptop_lid_lowering', 'approach_safe', 'looks_at_counter',
                    'safe_handle_touch', 'switch_operation', 'cash_preparation', 'cash_presented')
WEAK_VOCAB = ACTION_VOCAB + ', ' + ', '.join(WEAK_EXTRA_CODES)
OPPOSITES = {'safe_opening': {'safe_closing', 'safe_locking'},
             'safe_closing': {'safe_opening'}, 'safe_locking': {'safe_opening'},
             'cash_placed': {'cash_removed', 'cash_stored'}, 'cash_removed': {'cash_placed'},
             'phone_used': {'phone_put_away'}, 'phone_put_away': {'phone_used'}}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    with path.open('rb') as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def weak_exclusion(pair, human):
    """Human corrections win. Mere coexistence of different actions is not conflict."""
    if pair.get('adaptation_partition') != 'fit' or pair.get('split') != 'train':
        return 'not_frozen_fit'
    if pair.get('contains_violation_inference'):
        return 'description_contains_violation_inference'
    if 'quarantin' in str(pair.get('annotation_status', '')).lower():
        return 'quarantined'
    if pair.get('ambiguous_object_or_action') or not pair.get('action_candidates'):
        return 'ambiguous_object_or_action'
    if pair.get('eligible_for_weak_text_alignment') is not True:
        return 'not_eligible_for_weak_alignment'
    for known in human:
        if known['group_id'] != pair['group_id']:
            continue
        overlap = min(known['end_sec'], pair['end_sec']) - max(known['start_sec'], pair['start_sec'])
        if overlap <= 0:
            continue
        same_source = known['provenance'].get('source_kind') == 'user_reference' and known['provenance'].get('source_id') == pair.get('observation_id')
        if same_source and known['observed_action'] not in set(pair['action_candidates']) | {'other_observed_action'}:
            return 'human_corrected_same_source_action:' + known['task_id']
        substantial = overlap >= .5 * min(known['end_sec'] - known['start_sec'], pair['end_sec'] - pair['start_sec'])
        if substantial and any(known['observed_action'] in OPPOSITES.get(action, set()) for action in pair['action_candidates']):
            return 'potential_opposing_action_overlap_requires_review:' + known['task_id']
        if substantial and known['observed_action'] == 'object_handling' and pair.get('object_candidates'):
            return 'human_cannot_identify_overlapping_object:' + known['task_id']
    return None


def generic_prompt(task, timestamps):
    times = ', '.join(f'{index + 1}: {value:.3f}с' for index, value in enumerate(timestamps))
    shared = (f'Кадры идут по времени: {times}. Используй только видимое; имена, лица, аудио и текст удостоверений не анализируй. '
              'Два состояния не доказывают непрерывность или отсутствие промежуточного действия. '
              'Не называй предмет деньгами или удостоверением, если он неразличим. '
              'Не выдумывай нарушение, невидимое действие или достоверную вероятность. ')
    if task == 'video':
        return shared + (
            'Это выборочные кадры целого видео, не полный просмотр. Обрабатывай последовательно окна по четыре кадра. Проверь правила: удостоверение до денег; '
            'деньги предыдущего клиента в сейф до следующего; убрать телефон до пересчёта; пересчёт и проверка устройством — разные действия; '
            'уход сотрудника требует видимости сейфа, денег, ключей и документов. Тёмный экран не доказывает выключение. '
            'Повторный пересчёт допустим. Отсутствие действия в этих кадрах ничего не доказывает. '
            'Верни JSON: {"whole_video_verdict":"violation_supported|no_violation_observed|insufficient_data",'
            '"observations":[],"rules":[],"missing_evidence":[],"review_status":"proposed"}. '
            'Каждое наблюдение связывай с временем кадра. При недостаточном контексте выбирай insufficient_data.')
    ending = ('Верни JSON {"unconfirmed_candidates":[],"caption":"краткое видимое описание",'
              '"review_status":"proposed"}. Кандидаты не подтверждены человеком.' if task == 'weak' else
              'Верни JSON {"observed_action":"код действия", "description":"краткое видимое описание",'
              '"review_status":"proposed"}. При неясности используй object_handling или other_observed_action и объясни ограничение.')
    return shared + 'Допустимые коды: ' + (WEAK_VOCAB if task == 'weak' else ACTION_VOCAB) + '. ' + ending


def checked_target(value):
    text = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    if len(text.split()) > 100:
        raise ValueError('target_exceeds_100_words_not_truncated')
    return text


def sample_base(row, kind, identifier, protocol, count):
    group = row['group_id']
    identity = protocol['source_identity'].get(group)
    if identity is None or row['video_sha256'] != identity['video_sha256'] or row['split'] != identity['split']:
        raise ValueError('sample_source_identity_mismatch')
    partition = protocol['group_membership'][group]
    if row.get('adaptation_partition', partition) != partition:
        raise ValueError('sample_partition_mismatch')
    start, end, duration = float(row.get('start_sec', 0)), float(row.get('end_sec', row['duration_sec'])), float(row['duration_sec'])
    if not all(math.isfinite(v) for v in (start, end, duration)) or not 0 <= start < end <= duration + 1e-6:
        raise ValueError('invalid_source_interval')
    # Last requested instant lies inside the annotated interval. Actual native
    # PTS is separately recorded and must also stay inside the interval.
    last = max(start, end - min(.001, (end - start) / 100))
    requested = [start + (last - start) * index / (count - 1) for index in range(count)]
    return dict(sample_id=identifier, kind=kind, task='video' if kind == 'whole_video_eval' else 'action',
                partition=partition, group_id=group, split=row['split'], video=row['video'],
                video_sha256=row['video_sha256'], source_path=row['source_path'],
                start_sec=start, end_sec=end, duration_sec=duration,
                context_start_sec=row.get('context_start_sec', start), context_end_sec=row.get('context_end_sec', end),
                requested_timestamps=requested, frames=[None] * count, frame_paths=[], target=None,
                loss_weight=0., trust_level='evaluation_reference_only',
                labels_in_prompt=False, organizer_answers_in_prompt=False,
                source_protocol_sha256=protocol['protocol_sha256'])


def prepare_records(run, protocol):
    human = read_rows(run / 'human_actions_fit.jsonl')
    weak = read_rows(run / 'weak_pairs_fit.jsonl')
    training, evaluation_video, evaluation_action, excluded = [], [], [], []
    for row in human:
        try:
            if row.get('adaptation_partition') != 'fit' or row.get('action_confirmed') is not True or row.get('eligible_for_action_training') is not True:
                raise ValueError('not_confirmed_fit_action')
            value = sample_base(row, 'human_action', 'human_' + row['task_id'], protocol, 2)
            if value['partition'] != 'fit':
                raise ValueError('non_fit_action')
            target = checked_target(dict(observed_action=row['observed_action'], description=row['target_text'], review_status='proposed'))
            value.update(target=target, loss_weight=1., trust_level='human_confirmed',
                         provenance=dict(row['provenance'], source_manifest='human_actions_fit.jsonl'),
                         reference=row, prompt_task='action')
            training.append(value)
        except (ValueError, KeyError) as exc:
            excluded.append(dict(kind='human_action', source_id=row.get('task_id'), reason=str(exc)))
    for row in weak:
        try:
            reason = weak_exclusion(row, human)
            if reason:
                raise ValueError(reason)
            if set(row['action_candidates']) - set(ACTION_CODES) - set(WEAK_EXTRA_CODES):
                raise ValueError('weak_candidate_outside_versioned_generic_taxonomy')
            value = sample_base(row, 'weak_caption', 'weak_' + row['pair_id'], protocol, 2)
            if value['partition'] != 'fit':
                raise ValueError('non_fit_weak_caption')
            target = checked_target(dict(unconfirmed_candidates=row['action_candidates'], caption=row['caption'], review_status='proposed'))
            value.update(target=target, loss_weight=.2, trust_level='weak_unconfirmed',
                         provenance=dict(source_manifest='weak_pairs_fit.jsonl', pair_reference=row['pair_reference'],
                                         observation_reference=row['observation_reference'], source_reference=row['source_reference'],
                                         source_sha256=row['source_sha256']),
                         reference=row, prompt_task='weak')
            training.append(value)
        except (ValueError, KeyError) as exc:
            excluded.append(dict(kind='weak_caption', source_id=row.get('pair_id'), reason=str(exc)))
    for row in read_rows(run / 'video_level_targets.jsonl'):
        if protocol['group_membership'][row['group_id']] not in {'inner_holdout', 'diagnostic_val'}:
            continue
        value = sample_base(row, 'whole_video_eval', 'video_' + row['group_id'], protocol, 16)
        value.update(reference=row, provenance=dict(source_manifest='video_level_targets.jsonl'), prompt_task='video')
        evaluation_video.append(value)
    for row in read_rows(run / 'human_actions_inner_holdout.jsonl'):
        if protocol['group_membership'][row['group_id']] != 'inner_holdout':
            raise ValueError('Evaluation action outside inner holdout')
        value = sample_base(row, 'human_action_eval', 'eval_' + row['task_id'], protocol, 2)
        value.update(reference=row, provenance=dict(row['provenance'], source_manifest='human_actions_inner_holdout.jsonl'), prompt_task='action')
        evaluation_action.append(value)
    return training, evaluation_video, evaluation_action, excluded


def extract_video(path, requests, out, group, run):
    """One native video-only decode. Match nearest legal PTS; never fabricate time."""
    requests.sort(key=lambda request: request['requested'])
    saved, next_request = {}, 0
    def consume(request, candidates):
        legal = [(time_sec, frame) for time_sec, frame in candidates
                 if request['sample']['start_sec'] - 1e-7 <= time_sec <= request['sample']['end_sec'] + 1e-7]
        if not legal:
            raise ValueError('No actual decoded frame lies within requested annotation bounds')
        actual, frame = min(legal, key=lambda item: (abs(item[0] - request['requested']), item[0]))
        key = frame.pts
        if key not in saved:
            image = frame.to_image().convert('RGB')
            dest = out / 'frames' / f'{group}_{key}.jpg'
            image.save(dest, quality=95, subsampling=0)
            saved[key] = dict(path=dest.relative_to(run).as_posix(), sha256=sha(dest),
                              width=image.width, height=image.height, timestamp=actual, actual_sec=actual,
                              source_pts=frame.pts, source_time_base=str(frame.time_base), native_resolution=True)
        request['sample']['frames'][request['index']] = dict(saved[key], requested_sec=request['requested'])
    previous = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 2
        stream.thread_type = 'SLICE'
        origin = float((stream.start_time or 0) * stream.time_base)
        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            actual = float(frame.pts * frame.time_base) - origin
            current = (actual, frame)
            while next_request < len(requests) and requests[next_request]['requested'] <= actual:
                consume(requests[next_request], ([previous] if previous is not None else []) + [current])
                next_request += 1
            previous = current
            if next_request == len(requests):
                break
    if previous is not None:
        while next_request < len(requests):
            consume(requests[next_request], [previous])
            next_request += 1
    if next_request != len(requests):
        raise ValueError('Source video had insufficient decodable frames')
    return len(saved)


def run(root, run_dir):
    started = time.monotonic()
    root, run_dir = root.resolve(), run_dir.resolve()
    if not run_dir.is_relative_to(root / 'runs/gemma_lab'):
        raise ValueError('Preparation must stay under runs/gemma_lab')
    protocol = json.loads((run_dir / 'frozen_group_protocol.json').read_text())
    audit = json.loads((run_dir / 'audit.json').read_text())
    partition_audit = json.loads((run_dir / 'partition_export_audit.json').read_text())
    if sha(root / 'data/training/manifest.csv') != protocol['original_manifest_sha256']:
        raise ValueError('Original frozen manifest changed')
    for name in ('human_actions_fit.jsonl', 'weak_pairs_fit.jsonl', 'human_actions_inner_holdout.jsonl'):
        if sha(run_dir / name) != partition_audit['output_sha256'][name]:
            raise ValueError('Partitioned source changed: ' + name)
    if sha(run_dir / 'video_level_targets.jsonl') != audit['output_sha256']['video_level_targets.jsonl']:
        raise ValueError('Video-level references changed')
    training, video_eval, action_eval, excluded = prepare_records(run_dir, protocol)
    out = run_dir / 'prepared'
    out.mkdir(exist_ok=False)
    (out / 'frames').mkdir()
    groups = defaultdict(list)
    all_samples = training + video_eval + action_eval
    for sample in all_samples:
        for index, requested in enumerate(sample['requested_timestamps']):
            groups[sample['group_id']].append(dict(sample=sample, index=index, requested=requested))
    saved = 0
    for group, requests in sorted(groups.items()):
        sample = requests[0]['sample']
        source = (root / sample['source_path']).resolve()
        if not source.is_relative_to((root / 'data/raw/open/videos').resolve()) or sha(source) != sample['video_sha256']:
            raise ValueError('Actual video identity changed or escaped source directory')
        saved += extract_video(source, requests, out, group, run_dir)
        print(json.dumps(dict(group_id=group, frames_saved_total=saved, elapsed_sec=round(time.monotonic() - started, 2))), flush=True)
    for sample in all_samples:
        sample['frame_paths'] = [row['path'] for row in sample['frames']]
        times = [row['timestamp'] for row in sample['frames']]
        if any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError('Action/video sample does not contain distinct chronological frames')
        sample['prompt'] = generic_prompt(sample.pop('prompt_task'), times)
        sample['prompt_sha256'] = hashlib.sha256(sample['prompt'].encode()).hexdigest()
        if len(sample['prompt'].split()) > 180:
            raise ValueError('Generic prompt exceeds 180-word preparation cap')
        sample['sampled_coverage'] = dict(frame_count=len(times), actual_timestamps=times,
                                         continuous_video_observed=False, absence_can_be_inferred=False)
    for name, data in [('training_samples.jsonl', training), ('eval_video_samples.jsonl', video_eval),
                       ('eval_action_samples.jsonl', action_eval), ('preparation_exclusions.jsonl', excluded)]:
        write_new(out / name, data, jsonl=True)
    report = dict(schema_version=1, elapsed_sec=round(time.monotonic() - started, 3),
                  input_human_fit=len(read_rows(run_dir / 'human_actions_fit.jsonl')),
                  input_weak_fit=len(read_rows(run_dir / 'weak_pairs_fit.jsonl')),
                  accepted_human=sum(row['kind'] == 'human_action' for row in training),
                  accepted_weak=sum(row['kind'] == 'weak_caption' for row in training),
                  training_samples=len(training), training_groups=len({row['group_id'] for row in training}),
                  excluded=len(excluded), exclusions_by_reason=dict(Counter(row['reason'] for row in excluded)),
                  evaluation_videos=len(video_eval), evaluation_video_partitions=dict(Counter(row['partition'] for row in video_eval)),
                  evaluation_human_actions=len(action_eval), native_frames_saved=saved,
                  original_manifest_sha256=protocol['original_manifest_sha256'],
                  protocol_sha256=protocol['protocol_sha256'],
                  output_sha256={name: sha(out / name) for name in ('training_samples.jsonl', 'eval_video_samples.jsonl', 'eval_action_samples.jsonl', 'preparation_exclusions.jsonl')},
                  model_training_performed=False, audio_decoded=False, external_ai_used=False,
                  limitations=['Two frames are a bounded adaptation pilot, not full action sequence coverage.',
                               '16 sparse video frames in four chronological windows can miss events; absence is not normality.',
                               'Human source interval acceptance does not certify every boundary was independently retimed.',
                               'Weak targets remain a separate unconfirmed-caption task with weight 0.2.'])
    write_new(out / 'preparation_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.run)
