"""Create a reproducible binary baseline from organizer labels; never rewrite review CSV."""
import csv
import hashlib
import json
import random
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def prepare(root=ROOT):
    source = root / 'data/raw/open'
    with (root / 'data/review/event_review.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(list)
    for row in rows:
        path = (source / row['video']).resolve()
        if not path.is_relative_to(source.resolve()):
            raise ValueError('Unsafe video path')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        groups[digest].append(row)
    classes = defaultdict(list)
    for digest, items in groups.items():
        labels = {r['source_verdict'] for r in items}
        if len(labels) != 1 or not labels <= {'нарушение', 'без нарушения'}:
            raise ValueError('Conflicting or unknown source labels')
        classes[int(items[0]['source_verdict'] == 'нарушение')].append(digest)
    rng = random.Random(42)
    split = {}
    for label, hashes in sorted(classes.items()):
        hashes.sort()
        rng.shuffle(hashes)
        count = max(2, round(len(hashes) * .2))
        if len(hashes) <= count:
            raise ValueError('Not enough videos for independent validation')
        split.update({h: 'val' if i < count else 'train' for i, h in enumerate(hashes)})
    clips = []
    for digest, items in sorted(groups.items()):
        row = items[0]
        duration = float(row['duration_sec'])
        label = int(row['source_verdict'] == 'нарушение')
        # Keep both organizer anchors of duplicated videos, but only in the same split.
        anchors = sorted({float(r['anchor_sec']) for r in items}) if label else [duration * f for f in (.25, .5, .75)]
        for anchor in anchors:
            if not 0 <= anchor <= duration:
                raise ValueError('Anchor outside video')
            start = max(0, min(duration - 8, anchor - 4))
            end = min(duration, start + 8)
            clips.append(dict(video=row['video'], group_id='video_' + digest[:16], sha256=digest,
                              split=split[digest], start_sec=round(start, 3), end_sec=round(end, 3),
                              label='violation_unspecified' if label else 'normal', target=label,
                              anchor_sec=anchor, review_status='source_labeled',
                              annotation_mode='point_window' if label else 'source_video_label'))
    output = root / 'data/training'
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / 'manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(clips[0]))
        writer.writeheader()
        writer.writerows(clips)
    summary = dict(source_files=len({r['video'] for r in rows}), unique_videos=len(groups),
                   clips=len(clips), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                   splits={s: dict(videos=sum(v == s for v in split.values()),
                                   clips=sum(c['split'] == s for c in clips),
                                   classes=dict(Counter('violation' if items[0]['source_verdict'] == 'нарушение' else 'normal'
                                                        for h, items in groups.items() if split[h] == s))) for s in ('train', 'val')},
                   limitations=['Приблизительные окна по отметкам организатора; точные границы не размечены.',
                                'Проверка классификации фрагментов, не качества поиска событий в полном видео.',
                                'Разделение по SHA-256 исключает точные дубли; похожие постановочные сцены могут пересекаться.',
                                'Малая проверочная выборка; независимый тестовый набор не выделен.'])
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
