"""Bounded, sequential annotation pilot through the existing frame assistant."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.request


def run(args):
    base = args.base.rstrip('/')
    def api(route, payload=None):
        request = urllib.request.Request(base + route,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json', 'Origin': base})
        with urllib.request.urlopen(request, timeout=35) as response:
            return json.load(response)

    if not args.external_annotation_authorized:
        raise SystemExit('Explicit external annotation authorization is required.')
    if api('/api/frame-assist/status')['busy']:
        raise SystemExit('Frame assistant is busy; no jobs submitted.')
    out = args.queue.parent / args.run_name
    out.mkdir(exist_ok=False)
    (out / 'authorization.json').write_text(json.dumps(dict(
        authorized_at=datetime.now(timezone.utc).isoformat(),
        scope='Selected ambiguous train frames and minimal context, annotation only',
        production_inference=False, automatic_ground_truth=False, max_requests=args.limit), indent=2))
    results = []
    for item in map(json.loads, args.queue.read_text().splitlines()):
        if len(results) >= args.limit:
            break
        if item['kind'] != 'object_frame':
            continue  # Temporal context cannot be resolved by this single-frame tool.
        current = api('/api/human-objects')['decisions'].get(item['request']['image_id'], {})
        if current.get('version') != item['human_version']:
            continue  # Live human work takes precedence over this snapshot.
        submitted = api('/api/frame-assist', item['request'])
        job_id = submitted['job_id']
        (out / (job_id + '.submitted.json')).write_text(json.dumps(submitted))
        deadline = time.monotonic() + 310
        while True:
            job = api('/api/frame-assist/jobs/' + job_id)
            if job['status'] in ('completed', 'failed'):
                break
            if time.monotonic() >= deadline:
                raise SystemExit('Wait expired; existing job may still run. Do not submit a duplicate.')
            time.sleep(2)
        (out / (job_id + '.json')).write_text(json.dumps(job, ensure_ascii=False, indent=2))
        result = dict(image_id=item['request']['image_id'], video=item['video'],
            timestamp_sec=item['timestamp_sec'], job_id=job_id, status=job['status'],
            boxes=len((job.get('result') or {}).get('candidates', [])),
            eligible_for_training=False, review_status='proposed',
            review_url=item['review_url'] + '&assist_job=' + job_id)
        results.append(result)
        (out / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if job['status'] != 'completed':
            break


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue', required=True, type=Path)
    p.add_argument('--base', default='http://127.0.0.1:8765')
    p.add_argument('--run-name', default='execution_v1')
    p.add_argument('--limit', type=int, choices=range(1, 5), default=4)
    p.add_argument('--external-annotation-authorized', action='store_true')
    run(p.parse_args())
