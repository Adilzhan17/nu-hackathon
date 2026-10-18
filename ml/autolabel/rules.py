"""Evidence-gated operational hypotheses, separate from actions and legal labels.

Rules are project review templates, not bank policy or legal conclusions. Missing
annotations never establish an absence. Each assessment keeps insufficient_data
until explicit required evidence exists.
"""
RULES={
    'safe_open_across_worker_departure':dict(required=['worker_role_visible','worker_departure_sequence','safe_identity_visible','safe_door_open_before_and_after_departure'],absence_based=False),
    'cash_visible_across_worker_departure':dict(required=['worker_role_visible','worker_departure_sequence','cash_identity_visible','cash_on_surface_before_and_after_departure'],absence_based=False),
    'key_in_lock_across_worker_departure':dict(required=['worker_role_visible','worker_departure_sequence','key_and_safe_lock_visible','key_in_lock_before_and_after_departure'],absence_based=False),
    'banknote_check_not_observed':dict(required=['human_verified_operation_boundaries','whole_check_stage_visible','no_relevant_occlusion','human_verified_check_absence','applicable_bank_rule_verified'],absence_based=True),
}


def absence_assessment(verified_facts):
    """Only explicit human-verified facts may support a missing-step suspicion."""
    required=RULES['banknote_check_not_observed']['required']
    missing=[k for k in required if verified_facts.get(k) is not True]
    return dict(rule_id='banknote_check_not_observed',assessment='insufficient_data' if missing else 'requires_rule_review',
                missing_required_evidence=missing,legal_violation_target=None,eligible_for_violation_training=False)


def assessments(checks,patterns):
    rows=[]
    for c in checks:
        for rule_id,rule in RULES.items():
            matched=[p for p in patterns if p['check_id']==c['check_id'] and p['pattern']==rule_id]
            row=dict(check_id=c['check_id'],video=c['video'],group_id=c['group_id'],split='train',start_sec=c['start_sec'],end_sec=c['end_sec'],
                     rule_id=rule_id,assessment='requires_visual_and_rule_review' if matched else 'insufficient_data',
                     rule_authority='project_hypothesis_not_verified_bank_policy',
                     supporting_pattern_ids=[p['pattern_id'] for p in matched],required_evidence=rule['required'],
                     evidence_origin='local_model_proposal' if matched else None,
                     uncertainty_reason='Positive visual evidence is proposed; worker role, object identity and rule applicability require human review.' if matched else 'No complete verified evidence for this rule. Missing model annotations do not establish absence.',
                     legal_violation_target=None,eligible_for_violation_training=False,review_status='proposed')
            if rule['absence_based']:row.update(absence_assessment({}))
            rows.append(row)
    return rows
