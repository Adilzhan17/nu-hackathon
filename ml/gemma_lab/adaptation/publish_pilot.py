"""Publish a completed local fit demonstration in the existing Gemma viewer.

Copies a video-only preview and verified frames into a NEW job. No inference,
service restart, changes to old jobs, or promotion to human ground truth.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid


def read(path):
    return json.loads(path.read_text())


def publish(candidate, baseline, uploads, record, name):
    if record.exists():
        raise FileExistsError('Publication already recorded; preserve its existing job ID')
    candidate, baseline = candidate.resolve(), baseline.resolve()
    result = read(candidate / 'result.json')
    plan, old_plan = read(candidate / 'plan.json'), read(baseline / 'plan.json')
    if result.get('status') != 'completed':
        raise ValueError('Only a finished demonstration can be published by this helper')
    if result.get('organizer_answers_in_prompt') is not False or result.get('cloud_ai_used') is not False:
        raise ValueError('Expected blind, local-only provenance')
    if result.get('ground_truth') is not False or result.get('review_status') != 'proposed':
        raise ValueError('Model output must remain an unconfirmed proposal')
    if plan != old_plan:
        raise ValueError('Preview and candidate must share the original source and frame plan')
    for frame in plan['detail_frames']:
        path = (candidate / frame['path']).resolve()
        if not path.is_relative_to(candidate / 'frames'):
            raise ValueError('Frame path escaped the candidate frame directory')
        if hashlib.sha256(path.read_bytes()).hexdigest() != frame['sha256']:
            raise ValueError('Frame checksum mismatch')
    protocol = read(candidate.parent / 'frozen_group_protocol.json')
    group = 'video_' + plan['source_sha256'][:16]
    if protocol['group_membership'].get(group) != 'fit':
        raise ValueError('This helper publishes only the predeclared fit demonstration')
    job_id = uuid.uuid4().hex
    folder = uploads / job_id
    folder.mkdir(mode=0o700)
    shutil.copytree(candidate / 'frames', folder / 'frames')
    shutil.copy2(candidate / 'plan.json', folder / 'plan.json')
    shutil.copy2(baseline / 'preview.mp4', folder / 'preview.mp4')
    state = dict(id=job_id, name=name, status='completed', created_at=time.time(),
                 updated_at=time.time(), mode='gemma_lab', result=result,
                 review_status='proposed', ground_truth=False, stage='done',
                 message='Демонстрация на train-видео; не оценка accuracy. Выводы требуют проверки.',
                 duration_sec=plan['duration_sec'], sampled_frames=len(plan['frames']),
                 preview_ready=True, experiment_kind='fit_demonstration_not_accuracy',
                 group_id=group, adapter_loaded=False)
    temp = folder / 'job.json.tmp'
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    temp.replace(folder / 'job.json')
    provenance = dict(job_id=job_id, url='/demo/gemma?job=' + job_id,
                      candidate=str(candidate), result_sha256=hashlib.sha256(
                          (candidate / 'result.json').read_bytes()).hexdigest(),
                      original_job_unchanged=True, adapter_loaded=False,
                      experiment_kind=state['experiment_kind'], group_id=group)
    record.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(provenance, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--baseline', type=Path, default=Path(
        'runs/gemma_lab/uploads/f22640e7b6e34f69a6d6e55d7f314132'))
    parser.add_argument('--uploads', type=Path, default=Path('runs/gemma_lab/uploads'))
    parser.add_argument('--record', required=True, type=Path)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    publish(args.candidate, args.baseline, args.uploads, args.record, args.name)
