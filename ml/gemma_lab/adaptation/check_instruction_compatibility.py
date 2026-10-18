"""Read-only checks of candidate GGUF instructions against the live output contract.

No model, GPU, network or server change. Writes a NEW compatibility artifact only.
This checks schema/protocol mechanics, not recognition quality or token fit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from ml.gemma_lab.schema import output_schema
from ml.gemma_lab.protocol import ACTIONS, PROPERTIES, RULES, parse_result, inspect_answer, memory_before, assess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def declared_fields(text, name):
    match = re.search(r"^" + re.escape(name) + r" fields: ([^.]+)\.$", text, re.MULTILINE)
    if not match:
        raise ValueError("No explicit output fields for " + name)
    return set(match.group(1).split(","))


def contract_check(path, schema):
    text = path.read_text()
    details = {}
    for field in ("events", "states", "rule_checks"):
        declared = declared_fields(text, field)
        actual = set(schema["properties"][field]["items"]["properties"])
        details[field] = dict(extra_fields=sorted(declared - actual), missing_fields=sorted(actual - declared),
                              exact_match=declared == actual)
    rules_section = text.split("GENERAL OPERATIONAL RULES; APPLY ONLY WHEN TRIGGERED", 1)[1].split("VERDICT DISCIPLINE", 1)[0]
    mentioned_rules = set(re.findall(r"^([a-z_]+):", rules_section, re.MULTILINE))
    rule_match = mentioned_rules == set(RULES)
    return dict(path=str(path), sha256=sha(path), bytes=len(path.read_bytes()),
                field_checks=details, exact_rule_ids=rule_match,
                extra_rule_ids=sorted(mentioned_rules - set(RULES)),
                missing_rule_ids=sorted(set(RULES) - mentioned_rules),
                structural_contract_compatible=rule_match and all(row["exact_match"] for row in details.values()))


def probe_protocol():
    refs = [dict(id="f00000", timestamp=0.0), dict(id="f00001", timestamp=0.5),
            dict(id="f00002", timestamp=1.0)]
    shared = dict(actor="person_1", role="employee", object="unknown", track_id="unknown",
                  visibility="partial", description="Synthetic contract probe", uncertainty="Unclear object")
    for action in ACTIONS:
        answer = dict(events=[dict(action=action, from_actor="unknown", to_actor="unknown", **shared,
                                   evidence_frames=["f00000", "f00001"])], states=[], rule_checks=[])
        parsed = parse_result(json.dumps(answer))
        observations, claims, rejected = inspect_answer(parsed, refs, "synthetic_contract_probe", "overview_000")
        assert len(observations) == 1 and not claims and not rejected
        assert observations[0]["start_sec"] == 0 and observations[0]["end_sec"] == 0.5
        assert observations[0]["source"] == "local_gemma_visual_proposal"
        assert observations[0]["human_confirmed"] is False
    for prop in PROPERTIES:
        answer = dict(events=[], states=[dict(property=prop, value="unknown", **shared,
                                             evidence_frames=["f00001"])], rule_checks=[])
        observations, _, rejected = inspect_answer(answer, refs, "synthetic_contract_probe", "overview_000")
        assert len(observations) == 1 and not rejected
    for rule in RULES:
        answer = dict(events=[], states=[], rule_checks=[dict(rule_id=rule, verdict="violation_supported",
                      evidence_frames=["f00000", "f00001"], triggering_event="unknown",
                      explanation="Synthetic unsupported claim", missing_evidence=["Actual visual evidence"])])
        observations, claims, rejected = inspect_answer(answer, refs, "synthetic_contract_probe", "overview_000")
        assert not rejected and len(claims) == 1
        assert claims[0]["model_verdict"] == "violation_supported"
        assert claims[0]["verdict"] == "insufficient_data"
        assert next(row for row in assess(observations, claims) if row["rule_id"] == rule)["verdict"] == "insufficient_data"
    answer = dict(events=[dict(action="phone_used", from_actor="unknown", to_actor="unknown", **shared,
                              evidence_frames=["f00000", "f00001", "f00002"])], states=[], rule_checks=[])
    observations, _, _ = inspect_answer(answer, refs, "synthetic_contract_probe", "overview_000")
    past = memory_before(observations, 1.0)
    fact = past["recent_actions"][0]
    assert fact["last_observed_sec"] == 0.5
    assert fact["current_state"] == "unknown_until_reobserved"
    assert fact["uncertainty"] == "Unclear object"
    assert fact["human_confirmed"] is False
    answer["events"][0]["evidence_frames"] = ["old_memory_frame", "f00001"]
    observations, _, rejected = inspect_answer(answer, refs, "synthetic_contract_probe", "overview_000")
    assert not observations and rejected
    return dict(all_allowed_action_codes_accepted=len(ACTIONS), all_allowed_state_properties_accepted=len(PROPERTIES),
                raw_rule_claims_not_promoted_without_observations=len(RULES),
                evidence_times_derived_from_current_video_frames=True, future_and_past_outside_refs_not_accepted=True,
                memory_preserves_last_seen_uncertainty_and_unknown_current_state=True,
                all_outputs_are_unconfirmed_proposals=True,
                synthetic_probes_are_not_video_annotations=True)


def build(root, candidate):
    frames = [dict(id="f00000", timestamp=0), dict(id="f00001", timestamp=0.5)]
    schema = output_schema(frames)
    live = root / "ml/gemma_lab"
    previous = candidate.with_name("instructions_v2.txt")
    before = contract_check(previous, schema)
    after = contract_check(candidate, schema)
    return dict(schema_version="gemma_instruction_compatibility_v1", previous=before, candidate=after,
                inspected_code={str(path.relative_to(root)): sha(path) for path in
                                [live / "schema.py", live / "protocol.py", live / "runner.py",
                                 root / "ml/narration_training/organizer/context.py",
                                 root / "ml/context_pipeline/rules.py"]},
                output_schema=schema, runtime_probes=probe_protocol(),
                v2_blocker="v2 asks for event.recipient; active schema forbids all additional fields and has no recipient property.",
                v3_changes=["Removed recipient from output contract.",
                            "Documented exact active character/array limits and unknown-value location.",
                            "Kept server-derived timestamps/source/review status outside model JSON.",
                            "Distinguished scenario action vocabulary from human-training taxonomy."],
                drop_in_instruction_file_without_server_code_change=after["structural_contract_compatible"],
                activated=False, model_or_gpu_loaded=False, quality_improvement_established=False,
                limits=dict(context_tokens=16384, output_tokens=2200, maximum_overview_frames=12,
                            image_max_tokens_configured=512, output_events=3, output_states=3, output_rules=2,
                            actual_token_count_measured=False),
                limitations=[
                    "Schema compatibility does not establish quality, speed or fit of a large accumulated memory into the runtime token budget.",
                    "runner.py also says 'and unknowns'; the final schema and v3 explicitly prohibit a separate unknowns top-level key.",
                    "Rule engine can positively support only safe_on_departure and valuables_on_departure; other candidate violations stay model claims/insufficient_data.",
                    "Phone/counting, previous-customer cash and other unsupported rule gates cannot become supported site findings by changing instructions alone.",
                    "At most three events and three states per window can omit simultaneous actions. Better instructions do not remove this output-cap limit.",
                    "Current generic tracking/memory is model-proposed, not a verified object tracker or legal ownership record.",
                    "A short fit-group GGUF pilot with the same frames/runtime must precede any activation; no inference quality was measured here."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--candidate", type=Path, default=Path("ml/gemma_lab/adaptation/instructions_v3.txt"))
    parser.add_argument("--output", type=Path, default=Path("runs/gemma_lab/adaptation_v1/instruction_compatibility.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new compatibility artifact path")
    result = build(args.root.resolve(), args.candidate.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), v2_compatible=result["previous"]["structural_contract_compatible"],
                          v3_compatible=result["candidate"]["structural_contract_compatible"],
                          probes=result["runtime_probes"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
