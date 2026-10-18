"""Read-only provenance registry; exact duplicates and declared derivatives share groups."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

INPUTS = ['data/training/manifest.csv', 'data/review/event_review.csv',
          'data/transcripts/observations.jsonl', 'data/transcripts/sources.json']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def snapshot(root):
    return {p: digest(root / p) for p in INPUTS}


def registry(root, lineage=None):
    root = Path(root).resolve()
    base = root / 'data/raw/open'
    manifest = csv_rows(root / INPUTS[0])
    by_hash, by_group = {}, defaultdict(set)
    for row in manifest:
        if row['split'] not in {'train', 'val'}:
            raise ValueError('Unknown split')
        identity = (row['group_id'], row['split'])
        if row['sha256'] in by_hash and by_hash[row['sha256']] != identity:
            raise ValueError('Same source hash assigned to multiple groups/splits')
        by_hash[row['sha256']] = identity
        by_group[row['group_id']].add(row['split'])
    if any(len(s) != 1 for s in by_group.values()):
        raise ValueError('Group leakage between train and val')
    paths = {p.relative_to(base).as_posix(): p for p in (base / 'videos').rglob('*')
             if p.suffix.lower() in {'.mov', '.mp4', '.mkv', '.avi'} and p.is_file()}
    hashes = {}
    for name, p in paths.items():
        if not p.resolve().is_relative_to(base.resolve()):
            raise ValueError('Video escapes data root')
        hashes[name] = digest(p)
    for row in manifest:
        if hashes.get(row['video']) != row['sha256']:
            raise ValueError('Missing or changed manifest video: ' + row['video'])
    links = json.loads(Path(lineage).read_text()) if lineage else {}
    if not isinstance(links, dict) or any(k not in paths or v not in paths for k, v in links.items()):
        raise ValueError('Lineage must map existing derivative paths to existing source paths')
    result = {}

    def resolve(name, seen):
        if name in seen:
            raise ValueError('Cyclic lineage')
        identity = by_hash.get(hashes[name])
        if name in links:
            inherited = resolve(links[name], seen | {name})
            if identity and identity != inherited:
                raise ValueError('Declared source relationship conflicts with frozen groups/splits')
            identity = inherited
        if identity is None:
            raise ValueError('Unknown source: declare lineage, never guess split: ' + name)
        return identity

    reviews = csv_rows(root / INPUTS[1])
    observations = read_jsonl(root / INPUTS[2])
    sources = json.loads((root / INPUTS[3]).read_text())
    sections = {s['section']: s for s in sources['sections']}
    for name in paths:
        group, split = resolve(name, set())
        result[name] = dict(video=name, group_id=group, split=split, sha256=hashes[name])
    grouped_reviews, grouped_obs = defaultdict(list), defaultdict(list)
    for row in reviews:
        if row['video'] not in result:
            raise ValueError('Review refers to missing video')
        grouped_reviews[result[row['video']]['group_id']].append(row)
    for i, obs in enumerate(observations):
        if obs['video'] not in result:
            raise ValueError('Observation refers to missing video')
        section = sections.get(obs.get('section'), {})
        # user_confirmed is source-to-video mapping confirmation, not frame-level GT.
        confirmed = obs.get('status') in {'confirmed', 'reviewed'} or section.get('mapping') == 'user_confirmed'
        row = dict(obs, observation_id=f'observation_{i:04d}',
                   authority='quarantined' if obs['status'] == 'quarantined' else
                   ('user_confirmed_source' if confirmed else 'user_transcribed'),
                   source_mapping=section.get('mapping', 'unknown'))
        grouped_obs[result[obs['video']]['group_id']].append(row)
    for row in result.values():
        row['aliases'] = sorted(n for n, r in result.items() if r['group_id'] == row['group_id'])
    return result, grouped_reviews, grouped_obs
