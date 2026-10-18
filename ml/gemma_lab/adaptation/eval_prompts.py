"""Frozen compact prompts for a reduced, paired HF/NF4 evaluation protocol.

These are not the production 2 fps prompt and not a replacement for its long
instructions. They never interpolate labels, notes, names or organizer answers.
"""

OLD = """Inspect these silent chronological bank-camera frames. Describe only visible activity.
Use anonymous participants; do not infer identity, audio or intentions. General operational
requirements: examine ID before accepting cash; secure previous customer's cash before the
next customer; put phone away before counting; distinguish counting from authenticity-device
use; secure safe, cash, keys, documents and stamp when employee leaves. Inspect room-door
closure and equipment interactions. Never infer shutdown from screen color. Recounting is
not itself an error. Sample gaps and unseen actions remain unknown. Outputs are proposals,
not human confirmation or legal findings."""

V2 = """Inspect ALL current timestamped frames in order, comparing before, transition and after.
Report the action actually visible, not an expected action or a room description. Track the
same anonymous people, cash bundle and device only while visual continuity is supported.
Roles and objects can be unknown. An indistinct card is not automatically ID; a rectangle is
not automatically money. Document transfer differs from examination. Separate hand counting,
counter use and authenticity-device use; never judge banknote authenticity by appearance.
Rules: ID examination before cash acceptance for the same customer; previous cash secured
before a different customer's arrival; phone put away before counting; counting then checking;
safe and valuables secured when employee leaves; inspect room door after entry; counted cash
secured before drinking. Show positive simultaneous or ordered evidence for a conflict.
Leaving camera view is not room departure. Closed is not locked. Dark screen is not shutdown.
After occlusion current state is unknown, not a carried-forward certainty. Missing sampled
actions do not prove absence; repeated counting is not itself an error. These sparse images
cannot clear a whole video. No audio, identities, legal claims or organizer-specific answers.
Return a concise visual proposal; never invent evidence to fit a rule."""

VIDEO_CONTRACT = """Return ONLY JSON: predicted_label (1=tentative violation class, 0=tentative normal
class, null=unable), rule_verdict (violation_supported or insufficient_data), evidence_frames
(supplied IDs), reason (one short sentence). predicted_label is an UNCONFIRMED sampled-video
classification for measurement: 0 does NOT certify this video normal. Keep rule_verdict
insufficient_data unless >=2 frames positively support a concrete offending sequence;
classification 1 alone is not evidence. State a visible change and uncertainty. Earlier
model observations are fallible hypotheses, never established current facts."""

ACTION_CONTRACT = """Task: classify the principal visible action in this supplied interval.
Return ONLY JSON with observed_action (one allowed code, or unknown), description (one
short sentence grounded in the images), review_status (exactly proposed).
This is action recognition, not a violation verdict or human confirmation.
If the action/object cannot be identified, use unknown. Allowed codes: """


def prompt(condition, task, frames, action_codes, memory=None):
    if condition not in {"base_oldprompt", "base_v2prompt", "lora_v2prompt"}:
        raise ValueError("Unknown comparison condition")
    instruction = OLD if condition == "base_oldprompt" else V2
    if task == "video":
        contract = VIDEO_CONTRACT
    elif task == "action":
        contract = ACTION_CONTRACT + ",".join(action_codes)
    else:
        raise ValueError("Unknown task")
    times = "; ".join(f"{frame.get('id', f'f{i}')}={float(frame['timestamp']):.3f}s" for i, frame in enumerate(frames))
    history = ""
    if memory:
        import json
        history = "\nEarlier MODEL proposals; current states unknown until reobserved: " + json.dumps(memory, ensure_ascii=False, separators=(",", ":"))
    return instruction + "\nImages are supplied in this order: " + times + history + "\n" + contract
