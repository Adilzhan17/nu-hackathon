"""CLI: prepare (CPU), infer (local GPU), finalize (CPU), validate (CPU)."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from .dataset import digest, registry, snapshot, read_jsonl
from .schema import validate_proposal


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def inside(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Path outside authorized artifact directory')
    return path


def prepare(args):
    from .sampling import metadata, coarse_scan, candidates, frames
    root = args.root.resolve()
    output = inside(args.run, root / 'data/autolabel')
    if output.exists():
        raise FileExistsError('Use a NEW run directory; existing results are never overwritten')
    if not 1 <= args.coarse_fps <= 2 or not args.coarse_fps < args.dense_fps <= 8:
        raise ValueError('Require 1–2 coarse fps and higher dense fps (<=8)')
    if not 0 < args.window <= 8 or not 1 <= args.max_candidates <= 20:
        raise ValueError('Bound window and candidate count')
    before = snapshot(root)
    reg, reviews, observations = registry(root, args.lineage)
    wanted = args.videos or sorted({r['video'] for r in reg.values() if r['split'] == 'train'})
    selected, seen = [], set()
    for video in wanted:
        if video not in reg or reg[video]['split'] != 'train':
            raise ValueError('Only existing TRAIN videos may be proposed: ' + video)
        if reg[video]['group_id'] not in seen:
            selected.append(video)
            seen.add(reg[video]['group_id'])
    output.mkdir(parents=True)
    (output / 'frames').mkdir()
    durations, all_candidates, scan_stats = {}, [], []
    for video in selected:
        info = reg[video]
        path = root / 'data/raw/open' / video
        meta = metadata(path)
        duration = meta['duration_sec']
        durations[info['group_id']] = duration
        scan = coarse_scan(path, duration, args.coarse_fps)
        save(output / f"scan_{info['group_id']}.json", scan)
        selected_intervals = candidates(scan, duration, args.max_candidates, args.window)
        scan_stats.append(dict(video=video, group_id=info['group_id'], **meta,
                               coarse_samples=len(scan), candidate_intervals=len(selected_intervals)))
        for candidate in selected_intervals:
            cid = f'candidate_{len(all_candidates):04d}'
            # Dense frames include the interval's whole before/after progression.
            sampled = list(frames(path, candidate['start_sec'], candidate['end_sec'], args.dense_fps, max_side=960))
            if len(sampled) < 2:
                continue
            frame_rows = []
            for i, (ts, image) in enumerate(sampled):
                rel = f'frames/{cid}_{i:03d}.jpg'
                image.save(output / rel, quality=92)
                frame_rows.append(dict(timestamp=ts, path=rel, sha256=digest(output/rel), width=image.width, height=image.height))
            all_candidates.append(dict(candidate, candidate_id=cid, video=video, group_id=info['group_id'],
                                       split='train', dense_fps=args.dense_fps, frames=frame_rows))
        print(json.dumps(dict(prepared=video, candidates=len(selected_intervals), coarse_samples=len(scan))), flush=True)
    if snapshot(root) != before:
        raise RuntimeError('Source inputs changed during preparation; discard run')
    lock = json.loads((root / 'data/autolabel/model_lock.json').read_text())
    plan = dict(version=1, protected_inputs=before, registry=reg,
                reviews={g: reviews[g] for g in seen}, observations={g: observations[g] for g in seen},
                durations=durations, scans=scan_stats, selected_videos=selected,
                sampling=dict(coarse_fps=args.coarse_fps, dense_fps=args.dense_fps, window=args.window,
                              max_candidates=args.max_candidates, selection='change peaks plus low-motion audit'),
                model_lock=lock, candidates=all_candidates,
                lineage_sha256=digest(args.lineage) if args.lineage else None)
    save(output / 'plan.json', plan)
    print(json.dumps(dict(prepared_videos=len(selected), candidates=len(all_candidates), run=str(output))), flush=True)


@contextlib.contextmanager
def gpu_lock(path, wait_sec=30):
    # Cooperates with the other agent; only our own process owns/releases this lock.
    fd = open(path, 'a')
    start = time.monotonic()
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - start >= wait_sec:
                    raise TimeoutError('Shared GPU lock busy; no inference started')
                time.sleep(.25)
        yield
    finally:
        fd.close()


def infer(args):
    from .vlm import LocalVLM, parse_events, offline
    offline()
    from PIL import Image
    run = inside(args.run, args.root / 'data/autolabel')
    if (run / 'inference.json').exists() or (run / 'proposals.jsonl').exists():
        raise FileExistsError('Inference artifacts already exist; use a new run')
    plan = json.loads((run / 'plan.json').read_text())
    # Validate plan before touching GPU, including every frozen group and sampled frame.
    group_splits = {}
    for info in plan['registry'].values():
        group_splits.setdefault(info['group_id'], set()).add(info['split'])
    if any(len(s) != 1 for s in group_splits.values()):
        raise ValueError('Group leakage in inference plan')
    for c in plan['candidates']:
        info = plan['registry'][c['video']]
        if info['split'] != 'train' or c['group_id'] != info['group_id']:
            raise ValueError('Validation video cannot reach the model')
        ts = [f['timestamp'] for f in c['frames']]
        if len(ts) < 2 or ts != sorted(set(ts)) or not 0 <= c['start_sec'] < c['end_sec'] <= plan['durations'][c['group_id']]:
            raise ValueError('Invalid candidate sequence')
        for f in c['frames']:
            if not c['start_sec'] <= f['timestamp'] <= c['end_sec'] or digest(inside(run / f['path'], run / 'frames')) != f['sha256']:
                raise ValueError('Candidate frame provenance mismatch')
    code_dir = run / 'inference_code'
    code_dir.mkdir(exist_ok=False)
    code_hashes = {}
    for name in ('pipeline.py', 'vlm.py', 'schema.py'):
        src = Path(__file__).with_name(name)
        shutil.copy2(src, code_dir / name)
        code_hashes[name] = digest(code_dir / name)
    proposals, audit, rejected = [], [], []
    started = time.monotonic()
    with gpu_lock(args.gpu_lock) if args.device.startswith('cuda') else contextlib.nullcontext():
        model = LocalVLM(args.model_path, plan['model_lock'], device=args.device)
        for c in plan['candidates']:
            sample = [(f['timestamp'], Image.open(run / f['path']).convert('RGB')) for f in c['frames']]
            t0 = time.monotonic()
            raw = model.predict(sample)
            status, errors, accepted = 'ok', [], 0
            try:
                events = parse_events(raw)
                if not events:
                    status = 'empty'
                for event in events:
                    try:
                        times = event['evidence_timestamps']
                        p = dict(event, video=c['video'], group_id=c['group_id'],
                                 start_sec=min(times), end_sec=max(times),
                                 model_id=plan['model_lock']['model_id'], model_revision=plan['model_lock']['model_revision'],
                                 review_status='proposed', split='train', candidate_id=c['candidate_id'],
                                 confidence_kind='uncalibrated_model_self_rating')
                        validate_proposal(p, plan['registry'], plan['durations'][c['group_id']], [t for t, _ in sample])
                        p['proposal_id'] = f'proposal_{len(proposals):04d}'
                        proposals.append(p)
                        accepted += 1
                    except (ValueError, TypeError, KeyError) as exc:
                        errors.append(str(exc))
                        rejected.append(dict(candidate_id=c['candidate_id'], event=event, reason=str(exc)))
                if errors:
                    status = 'partially_invalid' if accepted else 'invalid'
            except (ValueError, TypeError, KeyError) as exc:
                status = 'invalid'
                errors.append(str(exc))
                rejected.append(dict(candidate_id=c['candidate_id'], reason=str(exc)))
            audit.append(dict(candidate_id=c['candidate_id'], raw_response=raw, inference_status=status,
                              errors=errors, accepted=accepted, elapsed_sec=round(time.monotonic()-t0, 3)))
            jsonl(run / 'raw_responses.jsonl', audit)
            jsonl(run / 'proposals.jsonl', proposals)
            jsonl(run / 'rejected.jsonl', rejected)
            print(json.dumps(dict(candidate=c['candidate_id'], status=status, accepted=accepted,
                                  elapsed_sec=audit[-1]['elapsed_sec'])), flush=True)
    import torch, transformers
    save(run / 'inference.json', dict(elapsed_sec=round(time.monotonic()-started, 3),
                                     proposals=len(proposals), rejected=len(rejected), candidates=len(audit),
                                     torch=torch.__version__, transformers=transformers.__version__,
                                     model_id=plan['model_lock']['model_id'], model_revision=plan['model_lock']['model_revision'],
                                     device=args.device, plan_sha256=digest(run/'plan.json'), code_sha256=code_hashes,
                                     max_pixels_per_frame=256*28*28, max_new_tokens=650, do_sample=False,
                                     prompt_sha256=hashlib.sha256(__import__('ml.autolabel.vlm', fromlist=['PROMPT']).PROMPT.encode()).hexdigest()))


def make_clip(root, run, item):
    # Local PyAV encode uses video frames only; no audio track can be copied.
    import av
    import numpy as np
    from fractions import Fraction
    from .sampling import frames
    relative = 'context/' + item['review_id'] + '.mp4'
    dst = run / relative
    dst.parent.mkdir(exist_ok=True)
    if dst.exists():
        raise FileExistsError('Context clip already exists')
    samples = frames(root / 'data/raw/open' / item['video'], item['context_start_sec'], item['context_end_sec'], 8., 768)
    first = None
    with av.open(str(dst), 'w', options={'movflags': '+faststart'}) as out:
        stream = out.add_stream('libx264', rate=8)
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '24', 'preset': 'fast'}
        for ts, image in samples:
            if first is None:
                first = ts
                stream.width, stream.height = image.width // 2 * 2, image.height // 2 * 2
            frame = av.VideoFrame.from_ndarray(np.asarray(image), format='rgb24')
            frame = frame.reformat(width=stream.width, height=stream.height, format='yuv420p')
            frame.pts = round((ts-first)*1000)
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    if first is None:
        raise ValueError('No frames for review context')
    item['context_clip'] = relative
    item['context_first_frame_sec'] = first


def finalize(args):
    from .review import reconcile, write_html
    root, run = args.root.resolve(), inside(args.run, args.root / 'data/autolabel')
    if (run / 'summary.json').exists():
        raise FileExistsError('Finalized run already exists')
    plan = json.loads((run/'plan.json').read_text())
    if snapshot(root) != plan['protected_inputs']:
        raise RuntimeError('Frozen inputs changed; rebuild with current inputs')
    if not (run/'inference.json').exists():
        raise ValueError('Inference did not complete')
    props = read_jsonl(run/'proposals.jsonl')
    audit = {a['candidate_id']: a for a in read_jsonl(run/'raw_responses.jsonl')}
    cs = [dict(c, inference_status=audit[c['candidate_id']]['inference_status']) for c in plan['candidates']]
    conflicts, queue = reconcile(props, cs, plan['observations'], plan['reviews'], plan['durations'], plan['registry'])
    for q in queue:
        make_clip(root, run, q)
    jsonl(run/'proposals.jsonl', props)
    jsonl(run/'conflicts.jsonl', conflicts)
    jsonl(run/'review_queue.jsonl', queue)
    write_html(run/'review.html', queue, props)
    # Clear proposals remain pending in proposals.jsonl; no automatic training export.
    summary = dict(proposals=len(props), conflicts=len(conflicts), review_intervals=len(queue),
                   videos_with_proposals=len({p['group_id'] for p in props}),
                   videos_coarse_scanned=len(plan['scans']), videos_dense_scanned=len({c['group_id'] for c in cs}),
                   candidates=len(cs), rejected_events=len(read_jsonl(run/'rejected.jsonl')),
                   split='train', ground_truth_created=0, input_hashes_unchanged=snapshot(root)==plan['protected_inputs'],
                   source_files_in_registry=len(plan['registry']), unique_groups_in_registry=len({r['group_id'] for r in plan['registry'].values()}),
                   inference=json.loads((run/'inference.json').read_text()),
                   limitations=['Candidate budget limits recall; coarse scanning is not exhaustive action detection.',
                                'VLM object/action hypotheses can be wrong even at high self-ratings.',
                                'Small keys, stamps, money versus paper require human verification.',
                                'Opposite overlapping actions may be successive phases, not true contradictions.',
                                'Transcribed observations and confirmed source mapping are not exact temporal ground truth.',
                                'Exact hashes plus explicit lineage cannot discover every undeclared re-encode.',
                                'No boxes: this backend does not run a specialized object detector.'])
    save(run/'summary.json', summary)
    validate(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def validate(args):
    run = inside(args.run, args.root / 'data/autolabel')
    plan = json.loads((run/'plan.json').read_text())
    if snapshot(args.root) != plan['protected_inputs']:
        raise ValueError('Protected source inputs changed')
    current, _, _ = registry(args.root, getattr(args, 'lineage', None))
    if current != plan['registry']:
        raise ValueError('Source registry changed')
    inference = json.loads((run/'inference.json').read_text())
    for name, expected in inference.get('code_sha256', {}).items():
        if digest(inside(run/'inference_code'/name, run/'inference_code')) != expected:
            raise ValueError('Archived inference code changed')
    if inference['plan_sha256'] != digest(run/'plan.json'):
        raise ValueError('Inference plan changed')
    by_candidate = {c['candidate_id']: c for c in plan['candidates']}
    props = read_jsonl(run/'proposals.jsonl')
    ids = set()
    for p in props:
        if p['proposal_id'] in ids:
            raise ValueError('Duplicate proposal id')
        ids.add(p['proposal_id'])
        if any(p[k] != plan['model_lock'][k] for k in ('model_id', 'model_revision')):
            raise ValueError('Proposal model provenance does not match pinned run')
        c = by_candidate[p['candidate_id']]
        if p['video'] != c['video'] or p['group_id'] != c['group_id']:
            raise ValueError('Proposal/candidate mismatch')
        validate_proposal(p, current, plan['durations'][p['group_id']], [f['timestamp'] for f in c['frames']])
    for c in plan['candidates']:
        for f in c['frames']:
            if digest(inside(run/f['path'], run/'frames')) != f['sha256']:
                raise ValueError('Evidence file changed')
    for name in ('review_queue.jsonl', 'conflicts.jsonl'):
        if not (run/name).exists():
            continue
        for q in read_jsonl(run/name):
            r = current[q['video']]
            if r['split'] != 'train' or q['group_id'] != r['group_id']:
                raise ValueError('Non-train review item')
            if not 0 <= q['start_sec'] < q['end_sec'] <= plan['durations'][q['group_id']]:
                raise ValueError('Invalid review bounds')
            if name == 'review_queue.jsonl':
                if not 0 <= q['context_start_sec'] <= q['start_sec'] < q['end_sec'] <= q['context_end_sec'] <= plan['durations'][q['group_id']]:
                    raise ValueError('Invalid context bounds')
                if not set(q['proposal_ids']) <= ids:
                    raise ValueError('Dangling proposal reference')
                clip = inside(run/q['context_clip'], run/'context')
                if not clip.is_file():
                    raise ValueError('Missing review clip')
    print(json.dumps(dict(validation='passed', proposals=len(props), group_leakage=False, protected_inputs_unchanged=True)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd())
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'infer', 'finalize', 'validate'):
        s = sub.add_parser(name)
        s.add_argument('--run', type=Path, required=True)
        if name in {'prepare', 'finalize', 'validate'}:
            s.add_argument('--lineage', type=Path)
        if name == 'prepare':
            s.add_argument('--videos', nargs='+')
            s.add_argument('--coarse-fps', type=float, default=1.5)
            s.add_argument('--dense-fps', type=float, default=4.)
            s.add_argument('--window', type=float, default=4.)
            s.add_argument('--max-candidates', type=int, default=3)
        if name == 'infer':
            s.add_argument('--model-path', type=Path, required=True)
            s.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
            s.add_argument('--gpu-lock', type=Path, default=Path('/home/ubuntu/nuhack-training/.gpu-work.lock'))
    args = p.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
