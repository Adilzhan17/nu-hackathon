"""Conservative disagreement flags; actions and organizer verdicts are separate axes."""
import html
import json
from collections import defaultdict
from pathlib import Path

# Differences within one broad window can be successive phases, not contradictions.
OPPOSITES = {frozenset(('safe_opening', 'safe_closing')), frozenset(('cash_received', 'cash_transferred'))}
ALIASES = {
    'safe_door_closing': 'safe_closing', 'safe_closing_and_locking': 'safe_closing',
    'cash_received_and_unpacked': 'cash_received', 'second_cash_received': 'cash_received',
    'partial_cash_return': 'cash_transferred', 'cash_placed_on_desk': 'cash_placed',
    'cash_stored': 'cash_placed', 'cash_removed_from_safe': 'cash_removed',
    'cash_counter_operation': 'cash_counting', 'keys_taken': 'key_handling',
    'safe_key_manipulation': 'key_handling', 'room_exit': 'leave_workstation',
    'leave_workstation_take_keys': 'leave_workstation',
    'document_inspection': 'document_handling', 'document_returned': 'document_handling',
    'document_presented': 'document_handling', 'stamps_stored': 'stamp_handling',
}


def interaction_object(action):
    # Only interactions, not object presence on a desk or a location inferred from context.
    if action.startswith('cash_') or action.startswith('counted_cash') or action in {'second_cash_received', 'partial_cash_return'}:
        return 'cash'
    if action in {'document_handling', 'document_inspection', 'document_presented',
                  'document_presented_and_inspected', 'document_returned', 'documents_stored'}:
        return 'document'
    if action in {'key_handling', 'keys_taken', 'safe_key_manipulation', 'safe_locking_keys_taken'}:
        return 'key'
    if action in {'stamp_handling', 'stamps_stored'}:
        return 'stamp'
    return None


def overlap(a, b, c, d):
    return max(a, c) <= min(b, d)


def bounds(observation, duration):
    a = observation.get('start_sec')
    if a is None or a == '':
        return None
    b = observation.get('end_sec')
    # Point/approximate observations receive narrow context, not fabricated exact ends.
    a = float(a)
    b = float(b) if b is not None and b != '' else a + 1
    return max(0, min(a, duration)), max(0, min(b, duration))


def related(observations, a, b, duration):
    return [o for o in observations if (r := bounds(o, duration)) and overlap(a, b, *r)]


def reconcile(proposals, candidates, observations, reviews, durations, registry, threshold=.8):
    conflicts, items = [], []
    # Quarantine is retained as a source issue, never used to seed model predictions.
    for group in sorted(durations):
        duration = durations[group]
        for o in observations[group]:
            if o['status'] != 'quarantined':
                continue
            interval = bounds(o, duration)
            a, b = interval if interval else (0., duration)
            a, b = max(0, a-2), min(duration, b+2)
            conflict = dict(conflict_id=f'conflict_{len(conflicts):04d}', kind='quarantined_source',
                            group_id=group, video=o['video'], start_sec=a, end_sec=b,
                            observation_ids=[o['observation_id']], reason=o.get('reason', ''),
                            resolution='Keep quarantined; reviewer must resolve source attribution.')
            conflicts.append(conflict)
            items.append(dict(video=o['video'], group_id=group, start_sec=a, end_sec=b,
                              reasons=['quarantined_source'], proposal_ids=[], conflict_ids=[conflict['conflict_id']]))
    for p in proposals:
        group = p['group_id']
        refs = related(observations[group], p['start_sec'], p['end_sec'], durations[group])
        authoritative = [o for o in refs if o['authority'] != 'quarantined']
        # Exact reviewed annotations also have priority. Binary verdicts are not actions.
        manual = []
        for i, row in enumerate(reviews[group]):
            if row.get('review_status') == 'reviewed' and row.get('label_scope') != 'binary':
                manual.append(dict(row, action=row.get('label'), observation_id=f'review_row_{group}_{i}', authority='manual_reviewed'))
        for o in authoritative + manual:
            r = bounds(o, durations[group])
            if not r or not overlap(p['start_sec'], p['end_sec'], *r):
                continue
            other = ALIASES.get(o.get('action'), o.get('action'))
            kind = None
            if frozenset((p['observed_action'], other)) in OPPOSITES:
                kind = 'possible_temporal_disagreement'
            expected_object = interaction_object(o.get('action') or '')
            proposed_object = interaction_object(p['observed_action'])
            if (expected_object and proposed_object and expected_object != proposed_object
                    and expected_object not in p['objects']
                    and min(p['end_sec'], r[1]) - max(p['start_sec'], r[0]) > 0):
                kind = 'possible_object_disagreement'
            if kind is None:
                continue
            reason = ('Different handled objects in overlapping intervals; may be simultaneous actions. Check source observation.'
                      if kind == 'possible_object_disagreement' else
                      'Opposite actions overlap; may be successive phases. Check sequence.')
            conflict = dict(conflict_id=f'conflict_{len(conflicts):04d}', kind=kind,
                            video=p['video'], group_id=group, start_sec=p['start_sec'], end_sec=p['end_sec'],
                            proposal_id=p['proposal_id'], observation_ids=[o['observation_id']],
                            authoritative_action=other, proposed_action=p['observed_action'],
                            authority=o['authority'], reason=reason,
                            resolution='Original annotation retained; proposal remains unapproved.')
            conflicts.append(conflict)
        p['source_observation_ids'] = [o['observation_id'] for o in refs]
        p['conflict_ids'] = [c['conflict_id'] for c in conflicts if c.get('proposal_id') == p['proposal_id']]
        reasons = []
        if p['confidence'] < threshold:
            reasons.append('low_model_self_rating')
        if p['uncertainty_reason']:
            reasons.append('model_visibility_or_timing_uncertainty')
        if p['observed_action'] == 'unclear_action' or 'unidentified_item' in p['objects']:
            reasons.append('unclear_action_or_object')
        if p['conflict_ids']:
            reasons.extend(sorted({c['kind'] for c in conflicts if c.get('proposal_id') == p['proposal_id']}))
        if any(o['authority'] == 'quarantined' for o in refs):
            reasons.append('quarantined_context')
        if reasons:
            items.append(dict(video=p['video'], group_id=group, start_sec=p['start_sec'], end_sec=p['end_sec'],
                              reasons=reasons, proposal_ids=[p['proposal_id']], conflict_ids=p['conflict_ids']))
    for c in candidates:
        if c.get('inference_status') not in {'ok', 'empty'}:
            reason = 'invalid_or_failed_model_output'
        elif c.get('inference_status') == 'empty':
            reason = 'model_abstained_not_a_negative_label'
        else:
            continue
        items.append(dict(video=c['video'], group_id=c['group_id'], start_sec=c['start_sec'], end_sec=c['end_sec'],
                          reasons=[reason], proposal_ids=[], conflict_ids=[]))
    # Merge overlapping flagged intervals, but retain all references. Never queue clear proposals by default.
    merged = []
    for item in sorted(items, key=lambda x: (x['group_id'], x['start_sec'])):
        if merged and merged[-1]['group_id'] == item['group_id'] and item['start_sec'] <= merged[-1]['end_sec'] + .5:
            last = merged[-1]
            last['end_sec'] = max(last['end_sec'], item['end_sec'])
            for key in ('reasons', 'proposal_ids', 'conflict_ids'):
                last[key] = sorted(set(last[key] + item[key]))
        else:
            merged.append(dict(item))
    for i, item in enumerate(merged):
        duration = durations[item['group_id']]
        item.update(review_id=f'review_{i:04d}', review_status='pending',
                    context_start_sec=max(0, item['start_sec']-2), context_end_sec=min(duration, item['end_sec']+2))
        item['observations'] = related(observations[item['group_id']], item['context_start_sec'], item['context_end_sec'], duration)
        item['organizer_labels'] = reviews[item['group_id']]
        item['aliases'] = registry[item['video']]['aliases']
    return conflicts, merged


def write_html(path, queue, proposals):
    by_id = {p['proposal_id']: p for p in proposals}
    cards = []
    for q in queue:
        title = f"{q['video']} · {q['start_sec']:.2f}–{q['end_sec']:.2f} s"
        media = html.escape(q.get('context_clip', ''), quote=True)
        body = html.escape(json.dumps(dict(reasons=q['reasons'], proposals=[by_id[i] for i in q['proposal_ids']],
                                          observations=q['observations'], organizer_labels=q['organizer_labels']), ensure_ascii=False, indent=2))
        cards.append(f'<article><h2>{html.escape(title)}</h2><p>Контекст начинается с {q["context_start_sec"]:.2f} с исходника.</p>'
                     f'<video controls muted preload="metadata" src="{media}"></video><pre>{body}</pre></article>')
    Path(path).write_text('<!doctype html><meta charset="utf-8"><title>Локальная очередь проверки</title>'
        '<style>body{font:16px system-ui;max-width:1000px;margin:auto}video{max-width:100%;width:720px}'
        'article{border-top:1px solid #aaa;padding:12px}pre{white-space:pre-wrap;font-size:13px}</style>'
        '<h1>Интервалы для проверки</h1><p>Все метки — предложения. Оценка модели не является вероятностью. '
        'Исходная разметка сохраняется. Видео без аудио. Сохранение ground truth здесь не выполняется.</p>' + ''.join(cards), encoding='utf-8')
