"""Snapshot ambiguous human review items. Read-only; never calls an AI provider."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request


def prepare(base):
    def get(route):
        with urllib.request.urlopen(base + route, timeout=30) as response:
            return json.load(response)

    objects = get('/api/human-objects')
    events = get('/api/human-review')
    queue = []
    for row in objects['rows']:
        decision = objects['decisions'].get(row['image_id'], {})
        if row['split'] != 'train' or not decision:
            continue
        if decision.get('review_status') != 'needs_context' and decision.get('image_assessment') != 'uncertain':
            continue
        queue.append(dict(kind='object_frame', video=row['video'], group_id=row['group_id'],
            split='train', timestamp_sec=row['timestamp'], review_status='pending_model_review',
            eligible_for_training=False, human_version=decision.get('version'),
            source_sha256=row['source_sha256'],
            request=dict(image_id=row['image_id'], image_sha256=row['image_sha256'], detail_tiles=True),
            review_url='/label/objects?image_id=' + row['image_id'],
            question='Какие предметы различимы? Предложить рамки и явно указать неопределённость.'))
    for row in events['tasks']:
        decision = events['decisions'].get(row['id'], {})
        if row['split'] != 'train' or decision.get('decision') != 'needs_context':
            continue
        queue.append(dict(kind='event_context', video=row['video'], group_id=row['group_id'],
            split='train', start_sec=row['start_sec'], end_sec=row['end_sec'],
            context_start_sec=row['context_start_sec'], context_end_sec=row['context_end_sec'],
            review_status='pending_context_review', eligible_for_training=False,
            task_id=row['id'], source_fingerprint=row['source_fingerprint'],
            source_sha256=row['video_sha256'],
            question='Просмотреть контекст до и после: что видно, что остаётся неизвестным?',
            single_frame_is_sufficient=False))
    return queue, dict(created_at=datetime.now(timezone.utc).isoformat(),
        object_counts=objects['counts'], event_counts=events['counts'],
        queue_items=len(queue), object_frames=sum(r['kind']=='object_frame' for r in queue),
        event_intervals=sum(r['kind']=='event_context' for r in queue),
        provider_requests=0, pending_external_processing_authorization=True,
        note='Only explicitly uncertain train items. Unreviewed and rejected items are not negatives; model output is not GT.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:8765')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows, summary = prepare(args.base.rstrip('/'))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'queue.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False))
