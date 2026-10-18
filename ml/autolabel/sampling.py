"""Video-only decoding; first pass at 1–2 fps, second pass at a higher rate."""
import math
import cv2
import numpy as np
import av


def metadata(path):
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        duration = float(s.duration * s.time_base) if s.duration else float(c.duration / av.time_base)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('Invalid video duration')
        return dict(duration_sec=duration, width=s.width, height=s.height, fps=float(s.average_rate))


def frames(path, start, end, fps, max_side=640):
    """Return true decoded PTS, never invented timestamps. Audio is never decoded."""
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        if start:
            c.seek(int((start + origin) / float(stream.time_base)), stream=stream, backward=True)
        target = start
        sample_index = 0
        previous = -1.
        for frame in c.decode(video=0):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base) - origin
            if t < start - 1e-6:
                continue
            if t > end + 1e-6:
                break
            if t + 1e-6 < target or t <= previous:
                continue
            image = frame.to_image()
            image.thumbnail((max_side, max_side))
            yield round(t, 6), image
            previous = t
            sample_index += 1
            target = start + sample_index / fps
            while target <= t + 1e-6:
                sample_index += 1
                target = start + sample_index / fps


def coarse_scan(path, duration, fps=1.5):
    if not 1 <= fps <= 2:
        raise ValueError('Coarse scan must use 1–2 fps')
    rows, prev = [], None
    for t, im in frames(path, 0, duration, fps, max_side=192):
        gray = cv2.resize(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY), (160, 90))
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        score = float(np.abs(gray.astype(float)-prev).mean() / 255) if prev is not None else 0.
        rows.append(dict(timestamp=t, change=round(score, 6)))
        prev = gray.astype(float)
    if len(rows) < 2:
        raise ValueError('Insufficient decodable video frames')
    return rows


def candidates(scan, duration, limit=3, window=4.):
    """Change peaks plus a quiet-window audit. Limit bounds pilot cost, not recall."""
    if limit < 1 or window <= 0:
        raise ValueError('Invalid candidate budget')
    def interval(t):
        a = max(0., min(max(0., duration-window), t-window/2))
        return round(a, 6), round(min(duration, a+window), 6)
    ranked = sorted(scan[1:], key=lambda r: (-r['change'], r['timestamp']))
    result = []
    # Reserve one window for low-motion activity when the budget allows.
    motion_budget = limit-1 if limit >= 3 else limit
    for row in ranked:
        a, b = interval(row['timestamp'])
        if any(max(0, min(b, c['end_sec'])-max(a, c['start_sec'])) > .4*(b-a) for c in result):
            continue
        result.append(dict(start_sec=a, end_sec=b, reason='visual_change', change_score=row['change']))
        if len(result) >= motion_budget:
            break
    if limit >= 3:
        for row in sorted(scan, key=lambda r: (r['change'], r['timestamp'])):
            a, b = interval(row['timestamp'])
            if any(max(0, min(b, c['end_sec'])-max(a, c['start_sec'])) > .4*(b-a) for c in result):
                continue
            result.append(dict(start_sec=a, end_sec=b, reason='low_motion_audit', change_score=row['change']))
            break
    return sorted(result, key=lambda c: c['start_sec'])
