"""Read-only detailed view of trained runs and the separate autolabel pilot."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


@lru_cache(maxsize=4)
def _plan_summary(path, modified_ns, size):
    raw = Path(path).read_bytes()
    plan = json.loads(raw)
    candidates = {c['candidate_id']: c['video'] for c in plan.get('candidates', [])}
    videos = {}
    for video in candidates.values():
        videos[video] = videos.get(video, 0) + 1
    return dict(plan_sha256=hashlib.sha256(raw).hexdigest(), candidates=candidates,
                videos=videos, sampling=plan.get('sampling', {}),
                model=plan.get('model_lock', {}).get('model_id'),
                source_seconds=sum(plan.get('durations', {}).values()))


def detailed_snapshot(root):
    """Only report locally received evidence; no inference about remote liveness."""
    folder = Path(root) / 'data/autolabel/detailed_train'
    plan_path = folder / 'plan.json'
    try:
        stat = plan_path.stat()
        plan = _plan_summary(str(plan_path), stat.st_mtime_ns, stat.st_size)
    except (OSError, ValueError, KeyError, TypeError):
        return dict(stage='not_received', message='План подробной авторазметки ещё не получен.')
    counts = {v: 0 for v in plan['videos']}
    proposals = invalid = 0
    seen = set()
    latest = stat.st_mtime
    for path in (folder / 'inference_records').glob('*.json'):
        record = read_json(path, {})
        cid = record.get('candidate_id')
        if cid not in plan['candidates'] or cid in seen:
            continue
        seen.add(cid)
        counts[plan['candidates'][cid]] += 1
        proposals += len(record.get('proposals', []))
        invalid += record.get('inference_status') in ('invalid', 'partially_invalid')
        try:
            latest = max(latest, path.stat().st_mtime)
        except OSError:
            pass
    progress = {}
    for name in ('inference_progress.json', 'inference.json'):
        path = folder / name
        data = read_json(path, {})
        if data.get('plan_sha256') == plan['plan_sha256']:
            progress = data
            try:
                latest = max(latest, path.stat().st_mtime)
            except OSError:
                pass
    total = len(plan['candidates'])
    count = progress.get('completed_windows', 0)
    count = count if type(count) is int and 0 <= count <= total else 0
    received = max(len(seen), count)
    stage = 'completed' if total and received == total else 'partial' if received else 'prepared'
    return dict(stage=stage, planned_windows=total, received_windows=received,
                videos=[dict(video=v, planned=n, records_received=counts[v]) for v, n in plan['videos'].items()],
                proposals_in_local_records=proposals, invalid_local_records=invalid,
                local_records=len(seen), last_received_at=latest, sampling=plan['sampling'],
                source_seconds=plan['source_seconds'], model=plan['model'],
                inference_seconds=progress.get('inference_seconds'),
                training_package=read_json(Path(root) / 'data/autolabel/training_v1/summary.json', {}),
                message={'prepared': 'План готов. Результаты подробного прохода на основной сервер ещё не поступили.',
                         'partial': 'Получена часть результатов. Состояние удалённого процесса по этим файлам неизвестно.',
                         'completed': 'Получены записи обо всех окнах. Предложения ещё требуют проверки.'}[stage])


def snapshot(jobs):
    root = jobs.root
    models = jobs.models()
    for model in models:
        folder = root / 'artifacts/models' / model['key']
        model['settings'] = read_json(folder / 'settings.json', {})
        model['validation'] = read_json(folder / 'validation.json', {})
        model['receipt'] = read_json(folder / 'receipt.json', {})
        model['sync'] = read_json(root / 'runs/suite' / model['key'] / 'sync.json', {})
        cpu = read_json(folder / 'cpu_validation.json', {})
        if model.get('metrics_device') == 'CPU':
            model['validation'] = cpu
        model['cpu_validation'] = {k: v for k, v in cpu.items() if k != 'predictions'}
        model['artifacts'] = [dict(name=name, bytes=(folder/name).stat().st_size,
                                 url=f"/api/models/{model['key']}/{name}")
                              for name in ('model.pt', 'validation.json', 'settings.json', 'cpu_validation.json')
                              if (folder/name).is_file()]
    dataset = read_json(root / 'data/training/summary.json', {})
    manifest = root / 'data/training/manifest.csv'
    dataset['current_manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None
    return dict(models=models, dataset=dataset,
                quality_experiment=dict(status=read_json(root / 'runs/quality_v2/status.json', {}),
                                        report=read_json(root / 'runs/quality_v2/report.json', {}),
                                        sequence_status=read_json(root / 'runs/quality_v2/sequence_status.json', {}),
                                        sequence_report=read_json(root / 'runs/quality_v2/sequence_report.json', {}),
                                        semantic_status=read_json(root / 'runs/quality_v2/semantic_status.json', {}),
                                        semantic_report=read_json(root / 'runs/quality_v2/semantic_report.json', {}),
                                        retrieval_report=read_json(root / 'runs/quality_v2/retrieval_report.json', {}),
                                        baseline_report=read_json(root / 'runs/quality_v2/baseline_video_report.json', {})),
                dataset_release={k:v for k,v in read_json(root / 'data/autolabel/release.json', {}).items()
                                 if k not in ('code_sha256','artifact_sha256','protected_inputs')},
                sync=read_json(root / 'runs/suite/sync.json', {}),
                autolabel=read_json(root / 'data/autolabel/pilot_report.json', {}),
                detailed_autolabel=detailed_snapshot(root))
