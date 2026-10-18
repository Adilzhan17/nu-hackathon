"""Factual local report from training ledgers, real checkpoints and paired eval.

No GPU/model calls, publication, source mutation or failed-pilot cleanup.
python -m ml.gemma_lab.adaptation.report --run runs/gemma_lab/adaptation_v1 \
  --training runs/gemma_lab/adaptation_v1/train_v1 \
  --video-eval runs/gemma_lab/adaptation_v1/eval_inner_video_v1 \
  --action-eval runs/gemma_lab/adaptation_v1/eval_inner_actions_v1 \
  --output runs/gemma_lab/adaptation_v1/final_report_v1
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

from .run_eval import metrics as compute_metrics
from .eval import classify

CONDITIONS = ('base_v2prompt', 'lora_v2prompt')
REMOTE_ROOT = Path('/home/ubuntu/nuhack-training')
FORCED_CONDITIONS = ('base_same_prompt', 'lora_same_prompt')
FORCED_PROTOCOL = 'gemma_all16_digit_logits_diagnostic_v1'


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    # Preserve in-flight append tails instead of trying to repair partial JSON.
    if raw and not raw.endswith(b'\n'):
        raw = raw[:raw.rfind(b'\n') + 1]
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def record_digest(row):
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def local_path(value, root):
    """Resolve own local/remote run artifacts without reading outside this run."""
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    elif path.is_relative_to(REMOTE_ROOT):
        marker = Path('runs/gemma_lab') / root.name
        relative = path.relative_to(REMOTE_ROOT)
        if not relative.is_relative_to(marker):
            raise ValueError('Remote artifact is outside this adaptation run')
        path = root / relative.relative_to(marker)
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError('Artifact path escapes the isolated adaptation run')
    return path


def checkpoint_details(path, expected_consumed=()):
    if path is None:
        return dict(available=False, reason='no_checkpoint_recorded')
    if not path.is_dir():
        return dict(available=False, path=str(path), reason='checkpoint_not_copied_to_main_server')
    tensor_path = path / 'adapter_model.safetensors'
    if not tensor_path.is_file():
        return dict(available=False, path=str(path), reason='adapter_tensor_file_missing')
    report = dict(available=True, path=str(path), adapter_sha256=sha(tensor_path), adapter_bytes=tensor_path.stat().st_size,
                  file_sha256={file.name: sha(file) for file in sorted(path.iterdir())
                               if file.is_file() and file.suffix in {'.json', '.safetensors'}},
                  changed_parameter_elements=None, exact_all_parameter_diff_available=False,
                  initialization_seed_caveat='Initialization seed scope is not attested by a seed number alone; source/run provenance verification is required.')
    state_path = path / 'trainer_state.json'
    state = read_json(state_path) if state_path.exists() else {}
    ids = state.get('consumed_sample_ids', [])
    report.update(saved_optimizer_updates=state.get('optimizer_updates'), saved_unique_samples=len(set(ids)),
                  saved_sample_ids=ids, current_successful_samples_missing_from_checkpoint=sorted(set(expected_consumed) - set(ids)))
    config = read_json(path / 'adapter_config.json') if (path / 'adapter_config.json').exists() else {}
    try:
        import numpy as np
        from safetensors.numpy import load_file
        arrays = load_file(str(tensor_path))
        b_tensors = {name: value for name, value in arrays.items() if '.lora_B.' in name}
        zeros_known = config.get('init_lora_weights', True) is True and config.get('use_dora', False) is False
        report.update(adapter_tensor_count=len(arrays), adapter_parameter_elements=sum(value.size for value in arrays.values()),
                      all_adapter_values_finite=all(bool(np.isfinite(value).all()) for value in arrays.values()),
                      lora_B_tensor_count=len(b_tensors), nonzero_lora_B_elements=sum(int(np.count_nonzero(value)) for value in b_tensors.values()),
                      zero_initialized_B_supported_by_adapter_config=zeros_known)
        if zeros_known and b_tensors:
            report['changed_parameter_elements'] = report['nonzero_lora_B_elements']
            report['change_count_scope'] = 'lower_bound: LoRA B elements changed from documented zero initialization; A changes not measured'
        else:
            report['change_count_scope'] = 'initial tensors unavailable; nonzero parameters do not establish change'
    except (ImportError, ValueError, TypeError, OSError) as exc:
        report['tensor_audit_unavailable'] = type(exc).__name__ + ': ' + str(exc)
    return report


def training_summary(root, training, eligible):
    if not (training / 'status.json').exists():
        return dict(status='not_run', actual_unique_samples=0, used_by_kind={}, eligible=len(eligible), checkpoint=dict(available=False)), []
    state = read_json(training / 'status.json')
    if state.get('dataset_sha256') is not None and state['dataset_sha256'] != sha(root / 'prepared/training_samples_v2.jsonl'):
        raise ValueError('Training status refers to a different prepared dataset version')
    eligible_by_id = {row['sample_id']: row for row in eligible}
    # A resumed run may reference successful updates in an earlier pilot's ledger.
    # Require each counted update to have an actual durable ledger entry.
    ledgers, successful = {}, {}
    for path in sorted(root.glob('*/consumption.jsonl')):
        entries = read_jsonl(path)
        ledgers[path.relative_to(root).as_posix()] = dict(sha256=sha(path), records=len(entries))
        for entry in entries:
            if entry.get('result', {}).get('updated') is True:
                successful[record_digest(entry)] = (entry, path.relative_to(root).as_posix())
    rows, problems = [], []
    for entry in state.get('consumed_records', []):
        sid = entry['sample_id']
        reference = eligible_by_id.get(sid)
        if record_digest(entry) not in successful:
            problems.append('consumed_record_missing_from_durable_ledger:' + sid)
            continue
        if reference is None or entry.get('group_id') != reference['group_id'] or entry.get('video_sha256') != reference['video_sha256']:
            raise ValueError('Consumed record differs from audited eligible source')
        if reference['partition'] != 'fit' or entry.get('result', {}).get('updated') is not True:
            raise ValueError('Only successful fit updates may count as used training data')
        for name, expected in [('prompt_sha256', hashlib.sha256(reference.get('prompt', '').encode()).hexdigest()),
                               ('target_sha256', hashlib.sha256(reference.get('target', '').encode()).hexdigest()),
                               ('frame_sha256', [frame['sha256'] for frame in reference.get('frames', [])])]:
            if name in entry and entry[name] != expected:
                raise ValueError('Consumed training prompt, target or frames differ from the audited sample')
        rows.append(entry)
    if len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('Repeated successful sample IDs in one-pass training state')
    ids = {row['sample_id'] for row in rows}
    if set(state.get('consumed_sample_ids', [])) != ids:
        problems.append('state_and_verified_ledger_consumption_differ')
    skipped = [entry for entry in read_jsonl(training / 'consumption.jsonl') if entry.get('result', {}).get('updated') is not True]
    checkpoint_path = local_path(state['checkpoint'], root) if state.get('checkpoint') else None
    checkpoint = checkpoint_details(checkpoint_path, ids)
    forensics_path = training / 'adapter_forensics.json'
    if forensics_path.exists():
        forensics = read_json(forensics_path)
        if forensics.get('adapter_sha256') != checkpoint.get('adapter_sha256'):
            raise ValueError('Adapter forensics belongs to a different checkpoint')
        checkpoint['forensics'] = forensics
        checkpoint['forensics_sha256'] = sha(forensics_path)
    histories = []
    for path in sorted(root.glob('*/status.json')):
        old = read_json(path)
        if 'optimizer_updates' not in old:
            continue
        histories.append(dict(run=path.parent.name, status=old.get('status'), phase=old.get('phase'),
                              optimizer_updates=old.get('optimizer_updates', 0), error=old.get('error'),
                              stop_reason=old.get('stop_reason'), successful_samples=len(set(old.get('consumed_sample_ids', []))),
                              elapsed_sec=old.get('elapsed_sec'), source_sha256=sha(path)))
    result = dict(status=state['status'], phase=state.get('phase'), stop_reason=state.get('stop_reason'), error=state.get('error'),
                  source_run=str(training), source_status_sha256=sha(training / 'status.json'),
                  eligible=len(eligible), eligible_by_kind=dict(Counter(row['kind'] for row in eligible)),
                  actual_unique_samples=len(ids), used_by_kind=dict(Counter(row['kind'] for row in rows)),
                  actual_groups=len({row['group_id'] for row in rows}), successful_update_records=len(rows),
                  optimizer_updates_reported=state.get('optimizer_updates'), elapsed_sec=state.get('elapsed_sec'),
                  samples_not_used=len(eligible) - len(ids), not_used_sample_ids=sorted(set(eligible_by_id) - ids),
                  skipped_attempts=len(skipped), skipped_reasons=dict(Counter(row['result'].get('reason', 'unknown') for row in skipped)),
                  skipped_records=[dict(sample_id=row['sample_id'], reason=row['result'].get('reason'), error=row['result'].get('error')) for row in skipped],
                  verified_ledger_sources=ledgers, consumption_validation_problems=problems,
                  checkpoint=checkpoint, model_id=state.get('model_id'), model_revision=state.get('model_revision'),
                  backend=state.get('backend'), successful_sample_ids=sorted(ids))
    return result, histories


def evaluation_summary(root, folder, checkpoint):
    if folder is None or not (folder / 'config.json').exists():
        return dict(status='not_run', fully_paired_samples=0, paired_only_conditions={})
    config = read_json(folder / 'config.json')
    if config.get('protocol_sha256') is not None and config['protocol_sha256'] != sha(root / 'frozen_group_protocol.json'):
        raise ValueError('Evaluation protocol differs from frozen source membership')
    if checkpoint.get('available') and checkpoint['adapter_sha256'] != config['adapter_checksums'].get('adapter_model.safetensors'):
        raise ValueError('Evaluation used a different adapter checkpoint; report its corresponding training run explicitly')
    samples = []
    for name, expected_sha in config['manifests'].items():
        manifest = local_path(name, root)
        if sha(manifest) != expected_sha:
            raise ValueError('Evaluation references changed since inference')
        samples.extend(row for row in read_jsonl(manifest) if row['sample_id'] in set(config['sample_ids']))
    lookup = {row['sample_id']: row for row in samples}
    if set(lookup) != set(config['sample_ids']):
        raise ValueError('Frozen evaluation sample IDs no longer match manifests')
    predictions = [row for row in read_jsonl(folder / 'predictions.jsonl') if row['condition'] in CONDITIONS]
    keys = [(row['sample_id'], row['condition']) for row in predictions]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate prediction for one evaluation condition/sample')
    by_sample = {}
    for row in predictions:
        sample = lookup.get(row['sample_id'])
        if sample is None or row.get('group_id') != sample['group_id']:
            raise ValueError('Prediction source differs from evaluation manifest')
        if row.get('organizer_answers_in_prompt') is not False:
            raise ValueError('Blind evaluation provenance is not explicit')
        if row.get('frame_sha256') != [frame['sha256'] for frame in sample['frames']]:
            raise ValueError('Inference frame identities differ from evaluation manifest')
        by_sample.setdefault(row['sample_id'], {})[row['condition']] = row
    for conditions in by_sample.values():
        if set(conditions) == set(CONDITIONS):
            if conditions[CONDITIONS[0]]['prompt_sha256'] != conditions[CONDITIONS[1]]['prompt_sha256']:
                raise ValueError('Paired base/adapter prompts or memory differ')
    report = compute_metrics(samples, predictions, CONDITIONS)
    state = read_json(folder / 'status.json') if (folder / 'status.json').exists() else {}
    report.update(status=state.get('status', 'unknown'), elapsed_sec=state.get('elapsed_sec'), source_run=str(folder),
                  config_sha256=sha(folder / 'config.json'), predictions_sha256=sha(folder / 'predictions.jsonl') if (folder / 'predictions.jsonl').exists() else None,
                  attempted_condition_samples=len(predictions), parse_status_counts=dict(Counter(row.get('parse_status') for row in predictions)),
                  adapter_checksums=config['adapter_checksums'], comparison_conditions=list(CONDITIONS),
                  checkpoint_matches_current_training=checkpoint.get('adapter_sha256') == config['adapter_checksums'].get('adapter_model.safetensors'),
                  state_error=state.get('error'))
    report['predicted_action_distribution_all_supplied_intervals'] = {
        condition: dict(Counter(row.get('observed_action', 'unknown') for row in predictions
                                if row['condition'] == condition and row['task'] == 'action'))
        for condition in CONDITIONS}
    return report


def forced_choice_summary(root, folder, checkpoint):
    """Keep the additional digit-logit protocol distinct from sequential JSON.

    Verify the full frozen eight-group schedule, actual paired inputs and adapter
    identity. Recompute scores from raw predictions and original references;
    never replace the primary evaluation with this later diagnostic.
    """
    if folder is None or not (folder / 'config.json').exists():
        return dict(status='not_run', protocol=FORCED_PROTOCOL, scheduled_groups=0,
                    fully_paired_groups=0, paired_only_conditions={}, comparison_complete=False)
    config = read_json(folder / 'config.json')
    protocol = read_json(root / 'frozen_group_protocol.json')
    if config.get('protocol') != FORCED_PROTOCOL or config.get('additional_protocol_not_replacement') is not True:
        raise ValueError('Unknown or conflated forced-choice protocol')
    if config.get('frozen_protocol_file_sha256') != sha(root / 'frozen_group_protocol.json'):
        raise ValueError('Forced-choice protocol differs from frozen source membership')
    if config.get('original_manifest_sha256') != protocol['original_manifest_sha256']:
        raise ValueError('Forced-choice original manifest hash mismatch')
    if config.get('organizer_answers_in_prompt') is not False or config.get('threshold_tuned') is not False:
        raise ValueError('Forced-choice provenance or threshold policy is not blind/frozen')
    if config.get('scores_are_calibrated_probabilities') is not False:
        raise ValueError('Raw digit logits must not be presented as calibrated probabilities')
    if checkpoint.get('available') and checkpoint['adapter_sha256'] != config['adapter_checksums'].get('adapter_model.safetensors'):
        raise ValueError('Forced-choice evaluation used another adapter checkpoint')
    manifest = root / 'prepared/eval_video_samples.jsonl'
    if config['samples_sha256'] != sha(manifest):
        raise ValueError('Forced-choice frame/reference manifest changed')
    partition = config.get('partition')
    if partition not in {'inner_holdout', 'diagnostic_val'}:
        raise ValueError('Forced-choice sample partition is outside frozen evaluation groups')
    expected = set(protocol['inner_holdout_groups' if partition == 'inner_holdout' else 'diagnostic_val_groups'])
    samples = [row for row in read_jsonl(manifest) if row['task'] == 'video' and row['partition'] == partition]
    if len(expected) != 8 or len(samples) != 8 or {row['group_id'] for row in samples} != expected:
        raise ValueError('Forced-choice must schedule all eight frozen source groups exactly once')
    by_id = {row['sample_id']: row for row in samples}
    if len(config.get('sample_ids', [])) != 8 or set(config['sample_ids']) != set(by_id):
        raise ValueError('Forced-choice sample schedule dropped or duplicated source groups')
    for sample in samples:
        identity = protocol['source_identity'][sample['group_id']]
        if sample['video_sha256'] != identity['video_sha256'] or sample['split'] != identity['split']:
            raise ValueError('Forced-choice source video identity mismatch')
        hashes = [frame['sha256'] for frame in sample['frames']]
        if len(hashes) != 16 or config['frame_sha256'].get(sample['sample_id']) != hashes:
            raise ValueError('Forced-choice schedule is not the same frozen sixteen frames')
    predictions = read_jsonl(folder / 'predictions.jsonl')
    by_sample, seen = {}, set()
    for row in predictions:
        key = (row.get('sample_id'), row.get('condition'))
        if key in seen or key[0] not in by_id or key[1] not in FORCED_CONDITIONS:
            raise ValueError('Unknown or duplicate forced-choice prediction')
        seen.add(key)
        sample = by_id[key[0]]
        if row.get('group_id') != sample['group_id'] or row.get('decision_scope') != 'whole_video':
            raise ValueError('Forced-choice prediction source/scope mismatch')
        if row.get('organizer_answers_in_prompt') is not False:
            raise ValueError('Forced-choice prediction lacks blind provenance')
        if row.get('frame_sha256') != [frame['sha256'] for frame in sample['frames']]:
            raise ValueError('Forced-choice model did not receive the same frozen frames')
        label = row.get('predicted_label')
        scores = row.get('candidate_logits')
        if label is not None:
            if type(label) is not int or label not in {0, 1} or not isinstance(scores, dict) or set(scores) != {'0', '1'}:
                raise ValueError('Forced-choice prediction must have finite raw logits and binary label')
            if not all(type(value) in {float, int} and math.isfinite(value) for value in scores.values()):
                raise ValueError('Forced-choice nonfinite logits cannot count as a classification')
            if label != int(scores['1'] > scores['0']):
                raise ValueError('Forced-choice label differs from frozen digit-logit argmax')
            if row.get('scores_are_calibrated_probabilities') is not False:
                raise ValueError('Forced-choice logits are not calibrated probabilities')
        by_sample.setdefault(key[0], {})[key[1]] = row
    paired_ids, input_failures = [], []
    for sid, conditions in by_sample.items():
        if set(conditions) != set(FORCED_CONDITIONS):
            continue
        base, lora = (conditions[name] for name in FORCED_CONDITIONS)
        if not base.get('prompt_sha256') or not lora.get('prompt_sha256'):
            # Encoding failures did not produce comparable model inputs. They
            # remain scheduled failures, not scored pairs or normal examples.
            if base.get('predicted_label') is not None or lora.get('predicted_label') is not None:
                raise ValueError('Forced-choice classification lacks a prompt fingerprint')
            input_failures.append(sid)
            continue
        if base['prompt_sha256'] != lora['prompt_sha256'] or base['frame_sha256'] != lora['frame_sha256']:
            raise ValueError('Forced-choice paired prompt or frames differ')
        paired_ids.append(sid)
    def targets(selected):
        return [dict(video=row['video'], group_id=row['group_id'], split=row['split'],
                     label=row['reference']['label'], label_scope='whole_video') for row in selected]
    paired = set(paired_ids)
    paired_samples = [row for row in samples if row['sample_id'] in paired]
    scheduled = {condition: classify(targets(samples), [row for row in predictions if row['condition'] == condition])
                 for condition in FORCED_CONDITIONS}
    paired_metrics = {condition: classify(targets(paired_samples), [row for row in predictions if row['condition'] == condition and row['sample_id'] in paired])
                      for condition in FORCED_CONDITIONS}
    base = paired_metrics[FORCED_CONDITIONS[0]]['correct_fraction_all_groups']
    lora = paired_metrics[FORCED_CONDITIONS[1]]['correct_fraction_all_groups']
    delta = lora - base if base is not None and lora is not None else None
    state = read_json(folder / 'status.json') if (folder / 'status.json').exists() else {}
    return dict(status=state.get('status', 'unknown'), elapsed_sec=state.get('elapsed_sec'), source_run=str(folder),
                protocol=FORCED_PROTOCOL, additional_protocol_not_replacement=True, partition=partition,
                scheduled_groups=8, fully_paired_groups=len(paired), fully_paired_sample_ids=sorted(paired),
                comparison_complete=len(paired) == 8, encoding_failed_sample_ids=input_failures,
                attempted_condition_samples=len(predictions), conditions=scheduled, paired_only_conditions=paired_metrics,
                effect=dict(base=base, lora=lora, absolute_change=delta,
                            conclusion='not_measured' if delta is None else 'partial_comparison_not_final' if len(paired) != 8 else
                            'no_improvement' if delta == 0 else 'improvement_in_this_additional_diagnostic' if delta > 0 else 'regression_in_this_additional_diagnostic'),
                validation_is_independent_test=False, threshold_tuned=False, scores_are_calibrated_probabilities=False,
                supported_rule_verdicts_measured=False, event_type_accuracy_measured=False,
                same_frames_prompt_and_checkpoint_verified=checkpoint.get('available', False), original_schedule_eight_groups_verified=True,
                adapter_checksums=config['adapter_checksums'], config_sha256=sha(folder / 'config.json'),
                predictions_sha256=sha(folder / 'predictions.jsonl') if (folder / 'predictions.jsonl').exists() else None,
                checkpoint_matches_current_training=checkpoint.get('adapter_sha256') == config['adapter_checksums'].get('adapter_model.safetensors'),
                model_id=config.get('model_id'), model_revision=config.get('model_revision'),
                input_protocol='all sixteen frozen frames jointly; direct comparison of 0/1 token logits')


def effects(video, actions):
    result = {}
    for name, source, section, key in [('whole_video_tentative', video, 'whole_video', 'correct_fraction_all_groups'),
                                       ('conditional_action_top1', actions, 'supplied_action_intervals', 'top1_correct_fraction_all')]:
        paired = source.get('paired_only_conditions', {})
        base = paired.get(CONDITIONS[0], {}).get('inner_holdout', {}).get(section, {})
        tuned = paired.get(CONDITIONS[1], {}).get('inner_holdout', {}).get(section, {})
        a, b = base.get(key), tuned.get(key)
        delta = b - a if a is not None and b is not None else None
        result[name] = dict(base=a, lora=b, absolute_change=delta,
                            conclusion='not_measured' if delta is None else 'improvement_in_this_small_paired_diagnostic' if delta > 0 else 'regression_in_this_small_paired_diagnostic' if delta < 0 else 'no_improvement',
                            metric=key, independent_test=False)
    return result


def percent(value):
    return 'не измерено' if value is None else f'{100 * value:.1f}%'


def format_outputs(summary):
    train, data, video, actions = (summary[key] for key in ('training', 'data', 'video_evaluation', 'action_evaluation'))
    used = train['used_by_kind']
    lines = [
        'Gemma 4 — фактическое дообучение и парная проверка', '',
        f"Подготовлено: {data['eligible_human']} подтверждённых и {data['eligible_weak']} слабых примеров; всего {data['eligible_total']}.",
        f"Разделение: {data['fit_groups']} исходные группы fit, из них {data['fit_groups_with_samples']} с обучающими примерами; {data['inner_holdout_groups']} inner holdout; {data['diagnostic_val_groups']} повторно использованных val.",
        f"Статус обучения: {train['status']}. Реально использованы {train['actual_unique_samples']} уникальных примеров: human {used.get('human_action', 0)}, weak {used.get('weak_caption', 0)}; групп {train.get('actual_groups', 0)}.",
        f"Пропуски попыток: {train.get('skipped_attempts', 0)}; причины: {json.dumps(train.get('skipped_reasons', {}), ensure_ascii=False)}.",
        f"Не поступили в успешные обновления: {train.get('samples_not_used', data['eligible_total'])}. Не считать подготовку данных обучением.",
    ]
    if train.get('error'):
        lines.append('Ошибка обучения: ' + train['error'])
    checkpoint = train.get('checkpoint', {})
    lines.append('Checkpoint: ' + (checkpoint.get('path', '') + ', SHA256 ' + checkpoint['adapter_sha256'] if checkpoint.get('available') else 'нет доступного сохранённого обученного adapter'))
    if checkpoint.get('changed_parameter_elements') is not None:
        lines.append(f"Как минимум {checkpoint['changed_parameter_elements']} параметров LoRA B отличаются от нулевой инициализации. Полный diff LoRA A без исходного snapshot не измерялся.")
    if checkpoint.get('forensics'):
        value = checkpoint['forensics']
        lines.append(f"Фактически ненулевые эффективные LoRA-обновления B@A: {value.get('nonzero_effective_delta_modules')}/{value.get('module_count')} модулей; конечные значения: {value.get('finite')}.")
    if checkpoint.get('available'):
        lines.append('Одного seed в metadata недостаточно для доказательства воспроизводимой инициализации: требуется проверка версии запущенного кода.')
    if train.get('consumption_validation_problems'):
        lines.append('Проблемы подтверждения потребления: ' + '; '.join(train['consumption_validation_problems']))
    lines += ['', 'Парное сравнение на одних кадрах и одинаковом v2 prompt:']
    metric_rows = []
    paired = video.get('paired_only_conditions', {})
    for condition, label in [(CONDITIONS[0], 'Gemma + инструкция'), (CONDITIONS[1], 'Gemma + инструкция + LoRA')]:
        section = paired.get(condition, {}).get('inner_holdout', {})
        values = section.get('whole_video', {})
        supported = section.get('rule_supported_whole_video', {})
        total, answered = values.get('total_source_groups', 0), values.get('answered', 0)
        correct = sum(values.get('confusion_matrix', {}).get(key, {}).get(value, 0) for key, value in
                      [('true_negative', 'predicted_negative'), ('true_positive', 'predicted_positive')])
        lines.append(f"{label}: tentative whole-video {correct}/{total}; ответов {answered}/{total}; accuracy среди ответов {percent(values.get('accuracy_on_answered'))}; верно среди всех {percent(values.get('correct_fraction_all_groups'))}.")
        lines.append(f"  Поддержанные моделью правила (не человеческое подтверждение): coverage {percent(supported.get('decision_coverage'))}. Это отдельный показатель, не качество бинарной догадки.")
        if total:
            metric_rows.append(dict(model=label + ' · предварительный класс видео', videos=total, answered=f'{answered}/{total}',
                                    accuracy=percent(values.get('accuracy_on_answered')), correct_all=f'{correct}/{total} ({percent(values.get("correct_fraction_all_groups"))})'))
    lines.append(f"Парно проверенных видео: {video.get('fully_paired_samples', 0)}; запланировано {video.get('scheduled_samples', 0)}. Непроверенные ролики не считаются правильными.")
    action_details = []
    for condition, label in [(CONDITIONS[0], 'База'), (CONDITIONS[1], 'LoRA')]:
        values = actions.get('paired_only_conditions', {}).get(condition, {}).get('inner_holdout', {}).get('supplied_action_intervals', {})
        text = f"{label}, действия на предоставленных человеком интервалах: {values.get('correct', 0)}/{values.get('scorable', 0)}; top-1 {percent(values.get('top1_correct_fraction_all'))}."
        lines.append(text)
        action_details.append(text)
    distribution = actions.get('predicted_action_distribution_all_supplied_intervals', {})
    if any(distribution.values()):
        for condition, label in [(CONDITIONS[0], 'База'), (CONDITIONS[1], 'LoRA')]:
            values = distribution.get(condition, {})
            if values:
                total = sum(values.values())
                most_common, count = max(values.items(), key=lambda pair: pair[1])
                text = f"{label}: самый частый ответ — {most_common}, {count}/{total} интервалов; различных ответов {len(values)}."
                lines.append(text + ' Полное распределение: ' + json.dumps(values, ensure_ascii=False, sort_keys=True))
                action_details.append(text)
        note = 'Распределение ответов включает все предоставленные интервалы, в том числе исключённые из top-1; сужение ответов — наблюдаемый симптом, не доказательство причины регрессии.'
        lines.append(note)
        action_details.append(note)
    lines += ['Это классификация действия в заданном интервале, не обнаружение событий и не accuracy типов нарушений.',
              'Неполная разметка не позволяет получить полные event precision/recall/mAP. Подтверждённых нарушений с интервалами нет.', '']
    for name, effect in summary['effects'].items():
        names = {'whole_video_tentative': 'Предварительный класс целого видео', 'conditional_action_top1': 'Действие в заданном интервале'}
        result = {'not_measured': 'не измерено', 'no_improvement': 'улучшения нет',
                  'improvement_in_this_small_paired_diagnostic': 'рост на этой небольшой парной диагностике',
                  'regression_in_this_small_paired_diagnostic': 'ухудшение на этой небольшой парной диагностике'}[effect['conclusion']]
        lines.append(f"{names[name]}: {result}; {percent(effect['base'])} → {percent(effect['lora'])}.")
    forced = summary.get('forced_choice_evaluation', {})
    additional = None
    if forced.get('status', 'not_run') != 'not_run':
        complete = forced['comparison_complete']
        additional_rows = []
        additional_details = [
            f"Сопоставлены {forced['fully_paired_groups']}/8 групп. " + ('Все восемь групп имеют парные результаты.' if complete else 'Сравнение ещё неполное; окончательная accuracy на восьми группах не заявляется.'),
            'Другой протокол: все 16 кадров одновременно; сравнение логитов ответов 0 и 1, без JSON и последовательных окон.',
            'Это предварительный бинарный классификатор, не проверка доказательств нарушений. Логиты не являются откалиброванными вероятностями.',
            'Дополнительный эксперимент не заменяет основные результаты JSON и действий и не является независимым тестом.',
        ]
        lines += ['', 'Дополнительный протокол — direct 0/1 по всем 16 кадрам одновременно:']
        for condition, label in [(FORCED_CONDITIONS[0], 'База · direct 0/1'), (FORCED_CONDITIONS[1], 'LoRA · direct 0/1')]:
            value = forced['paired_only_conditions'][condition]
            total, answered = value['total_source_groups'], value['answered']
            correct = sum(value['confusion_matrix'][key][item] for key, item in
                          [('true_negative', 'predicted_negative'), ('true_positive', 'predicted_positive')])
            lines.append(f"{label}: {correct}/{total}; ответов {answered}/{total}; accuracy среди ответов {percent(value['accuracy_on_answered'])}; верно среди всех парных групп {percent(value['correct_fraction_all_groups'])}.")
            if total:
                additional_rows.append(dict(model=label, videos=total, answered=f'{answered}/{total}',
                                            accuracy=percent(value['accuracy_on_answered']), correct_all=f'{correct}/{total} ({percent(value["correct_fraction_all_groups"])})'))
        lines += additional_details
        conclusion = forced['effect']['conclusion']
        if conclusion == 'no_improvement':
            extra_message = 'В дополнительном direct 0/1 сравнении улучшения нет.'
        elif conclusion == 'improvement_in_this_additional_diagnostic':
            extra_message = 'В дополнительном direct 0/1 сравнении есть рост; это отдельный протокол и небольшая внутренняя диагностика.'
        elif conclusion == 'regression_in_this_additional_diagnostic':
            extra_message = 'В дополнительном direct 0/1 сравнении результат после LoRA ухудшился.'
        else:
            extra_message = 'Дополнительное direct 0/1 сравнение ещё не завершено на всех восьми группах.'
        lines.append(extra_message)
        additional = dict(title='Дополнительная проверка · direct 0/1', message=extra_message,
                          metrics=additional_rows, details=additional_details)
    diagnostic_ran = any('diagnostic_val' in sections for source in (video, actions)
                         for sections in source.get('paired_only_conditions', {}).values()) or (
                         forced.get('partition') == 'diagnostic_val' and forced.get('fully_paired_groups', 0) > 0)
    lines += ['', 'Исходные val: ' + ('повторно использованная диагностика, не независимый test.' if diagnostic_ran else 'не запускались в этом сравнении; они раскрыты и не являются независимым test.'),
              'Протокол: HF NF4 база, а не deployed GGUF QAT Q4_0; 16 выборочных кадров в четырёх последовательных окнах, два кадра на действие.',
              'Это ограниченный пилот, не оценка полной production обработки каждые 0,5 секунды.',
              'Человеческая разметка, source train/val, production checkpoint и история не заменялись.',
              'История запусков:']
    for history in summary['training_history']:
        lines.append(f"- {history['run']}: {history['status']}, обновлений {history['optimizer_updates']}, ошибка {history.get('error') or 'нет'}")
    conclusions = [effect['conclusion'] for effect in summary['effects'].values()]
    if all(value == 'not_measured' for value in conclusions):
        message = 'Дообучение ещё не имеет завершённого парного измерения качества. Accuracy после обучения не заявляется.'
    elif all(value in {'not_measured', 'no_improvement', 'regression_in_this_small_paired_diagnostic'} for value in conclusions):
        message = 'Парное сравнение проведено: улучшение после LoRA не подтверждено. Фактические результаты приведены ниже.'
    else:
        message = 'Есть измеренный результат на небольшой внутренней проверке; он не доказывает качество на независимых видео.'
    details = [f"В успешных обновлениях: {train['actual_unique_samples']}/{data['eligible_total']} примеров (подтверждённых {used.get('human_action', 0)}, слабых {used.get('weak_caption', 0)}).",
               f"Сравнение видео: {video.get('fully_paired_samples', 0)} пар; бинарный класс — предварительная догадка, не подтверждённое нарушение.",
               *action_details,
               'Точность типов нарушений и полный recall событий не измерены: соответствующей подтверждённой разметки недостаточно.',
               'Прежние val раскрыты и не являются независимым тестом. Production checkpoint автоматически не менялся.']
    public = dict(schema_version=2, title='Дообучение Gemma · фактические результаты', message=message,
                  metrics=metric_rows, details=details)
    if additional is not None:
        public['additional_protocol'] = additional
    return '\n'.join(lines) + '\n', public


def run(run_root, training, video_eval, action_eval, output, forced_choice_eval=None):
    root = run_root.resolve()
    if not output.resolve().is_relative_to(root):
        raise ValueError('Report output must stay inside this isolated experiment')
    protocol = read_json(root / 'frozen_group_protocol.json')
    prepared = read_json(root / 'prepared/preparation_report.json')
    eligible = read_jsonl(root / 'prepared/training_samples_v2.jsonl')
    train, histories = training_summary(root, local_path(training, root), eligible)
    video = evaluation_summary(root, local_path(video_eval, root) if video_eval else None, train['checkpoint'])
    actions = evaluation_summary(root, local_path(action_eval, root) if action_eval else None, train['checkpoint'])
    forced = forced_choice_summary(root, local_path(forced_choice_eval, root) if forced_choice_eval else None, train['checkpoint'])
    if forced.get('status') != 'not_run' and (forced.get('model_id'), forced.get('model_revision')) != (train.get('model_id'), train.get('model_revision')):
        raise ValueError('Forced-choice base model differs from the training experiment')
    summary = dict(schema_version=2, generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                   data=dict(fit_groups=len(protocol['fit_groups']), inner_holdout_groups=len(protocol['inner_holdout_groups']),
                             diagnostic_val_groups=len(protocol['diagnostic_val_groups']), fit_groups_with_samples=len({r['group_id'] for r in eligible}),
                             eligible_total=len(eligible), eligible_human=sum(r['kind'] == 'human_action' for r in eligible),
                             eligible_weak=sum(r['kind'] == 'weak_caption' for r in eligible), preparation_exclusions=prepared['excluded'],
                             protocol_sha256=protocol['protocol_sha256'], original_manifest_sha256=protocol['original_manifest_sha256']),
                   training=train, training_history=histories, video_evaluation=video, action_evaluation=actions,
                   forced_choice_evaluation=forced,
                   effects=effects(video, actions), production_checkpoint_changed=False,
                   publisher_invoked=False, original_labels_modified=False, failed_histories_preserved=True,
                   verified_violation_type_accuracy=None, independent_test_accuracy=None)
    report, public = format_outputs(summary)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('summary.json', summary), ('public_status.json', public)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (output / 'REPORT.txt').write_text(report)
    print(json.dumps(dict(report=str(output / 'REPORT.txt'), successful_samples=train['actual_unique_samples'],
                          paired_videos=video.get('fully_paired_samples', 0), paired_action_samples=actions.get('fully_paired_samples', 0),
                          additional_forced_choice_pairs=forced.get('fully_paired_groups', 0),
                          effects=summary['effects']), ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--training', required=True, type=Path)
    parser.add_argument('--video-eval', type=Path)
    parser.add_argument('--action-eval', type=Path)
    parser.add_argument('--forced-choice-eval', type=Path, help='Additional direct-0/1 diagnostic; never replaces primary results')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.run, args.training.resolve(), args.video_eval.resolve() if args.video_eval else None,
        args.action_eval.resolve() if args.action_eval else None, args.output.resolve(),
        args.forced_choice_eval.resolve() if args.forced_choice_eval else None)
