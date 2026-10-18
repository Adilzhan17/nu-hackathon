"""Strict semantic validation in addition to the published JSON schema."""
import math
import re

ACTIONS = ('cash_received', 'cash_transferred', 'cash_counting', 'cash_placed',
           'cash_removed', 'safe_opening', 'safe_closing', 'leave_workstation',
           'key_handling', 'document_handling', 'stamp_handling', 'unclear_action',
           'cash_unpacking', 'cash_bundling', 'cash_counter_loading', 'safe_locking',
           'document_transfer', 'document_stamping', 'return_to_workstation',
           'computer_work', 'person_enters', 'person_leaves', 'object_handling')
OBJECTS = ('cash', 'safe', 'key', 'document', 'stamp', 'person', 'cash_counter', 'unidentified_item', 'computer', 'bag')
REQUIRED = {'video', 'group_id', 'start_sec', 'end_sec', 'observed_action', 'objects',
            'evidence_timestamps', 'confidence', 'uncertainty_reason', 'model_id',
            'model_revision', 'review_status'}


def number(value):
    return type(value) in (float, int) and math.isfinite(value)


def validate_proposal(p, registry, duration, sampled=None):
    if not isinstance(p, dict) or not REQUIRED <= p.keys():
        raise ValueError('Missing required proposal fields')
    allowed = REQUIRED | {'split', 'candidate_id', 'proposal_id', 'confidence_kind', 'source_observation_ids', 'conflict_ids'}
    if not p.keys() <= allowed:
        raise ValueError('Unknown proposal fields, including unsupported coordinates or verdicts')
    if p.get('split', 'train') != 'train':
        raise ValueError('Proposal declares non-train split')
    if p.get('confidence_kind', 'uncalibrated_model_self_rating') != 'uncalibrated_model_self_rating':
        raise ValueError('Unsupported confidence interpretation')
    source = registry.get(p['video'])
    if not source or source['split'] != 'train' or source['group_id'] != p['group_id']:
        raise ValueError('Proposal is not in the frozen train group')
    a, b = p['start_sec'], p['end_sec']
    if not all(number(t) for t in [a, b, duration]) or not 0 <= a < b <= duration:
        raise ValueError('Invalid temporal bounds')
    if p['observed_action'] not in ACTIONS:
        raise ValueError('Action outside observable taxonomy')
    if not isinstance(p['objects'], list) or any(x not in OBJECTS for x in p['objects']):
        raise ValueError('Invalid object taxonomy')
    ts = p['evidence_timestamps']
    if not isinstance(ts, list) or len(ts) < 2 or any(not number(t) or not a <= t <= b for t in ts):
        raise ValueError('Need at least two temporal evidence frames within the event')
    if ts != sorted(set(ts)) or ts[-1] - ts[0] < .1:
        raise ValueError('Evidence must describe a sequence, not one frame')
    if sampled is not None and any(min(abs(t-s) for s in sampled) > .002 for t in ts):
        raise ValueError('Evidence timestamp was not actually sampled')
    if not number(p['confidence']) or not 0 <= p['confidence'] <= 1:
        raise ValueError('Invalid model self-rating')
    if (not isinstance(p['uncertainty_reason'], str) or not isinstance(p['model_id'], str) or not p['model_id']
            or not isinstance(p['model_revision'], str) or not re.fullmatch(r'[0-9a-f]{40}', p['model_revision'])):
        raise ValueError('Missing uncertainty/model provenance')
    if p['review_status'] != 'proposed':
        raise ValueError('Pseudo-labels cannot become ground truth')
    # This backend is a VLM, not a verified bounding-box detector.
    if any(k in p for k in ('bbox', 'boxes', 'bounding_boxes', 'detections')):
        raise ValueError('VLM coordinates are not detector measurements')
    return p
