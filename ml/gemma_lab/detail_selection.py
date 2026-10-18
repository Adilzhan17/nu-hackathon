"""Budgeted temporal reinspection, selected only from model observations.

Prioritize security transitions across the entire overview before routine early
events. This is a sampling heuristic, not ground truth or a violation detector.
"""
import math

EVENT_PRIORITY = {
    'employee_leaves': 0,
    'employee_absent': 1,
    'key_handling': 1,
    'stamp_handling': 1,
    'safe_opening': 2,
    'safe_closing': 2,
    'person_leaves': 3,
    'cash_received': 4,
    'cash_counted_by_hand': 4,
    'cash_counter_used': 4,
    'phone_used': 4,
    'drink_used': 4,
    'person_enters': 5,
}


def select_detail_centers(observations, frames, *, limit=2, separation=2.0):
    if limit <= 0:
        return []
    candidates = []
    for observation in observations:
        priority = EVENT_PRIORITY.get(observation.get('action'))
        if (observation.get('kind') == 'state'
                and observation.get('property') in {'key_location', 'stamp_location', 'safe_door'}
                and observation.get('value') in {'safe_top', 'in_lock', 'work_surface', 'desk', 'open'}):
            priority = 2
        start, end = observation.get('start_sec'), observation.get('end_sec')
        if priority is None or any(type(t) not in (int, float) or not math.isfinite(t) for t in (start, end)):
            continue
        if not 0 <= start <= end:
            continue
        # Before/after evidence bounds the transition; do not always recheck
        # only its first frame, which can precede the actual movement.
        center = (start + end) / 2
        candidates.append((priority, center, observation.get('observation_id', '')))
    selected = []
    for priority, center, identifier in sorted(candidates):
        if all(abs(center - old['timestamp']) > separation for old in selected):
            selected.append(dict(timestamp=center, priority=priority, observation_id=identifier))
        if len(selected) >= limit:
            break
    if not selected and frames and limit > 0:
        selected = [dict(timestamp=frames[len(frames) // 2]['timestamp'], priority=None,
                         observation_id=None, reason='midpoint_fallback_no_candidate')]
    return selected
