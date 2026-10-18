"""CPU-only posthoc fit-demo comparison; never an accuracy measurement.

Reads a saved candidate result, the original completed phone pilot and frozen
human references. It never runs inference or places references in a model prompt.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from ml.gemma_lab.protocol import ACTIONS
from .eval import temporal_iou

GROUP = "video_57a0e46c7a45f0e8"
REFERENCE_TASKS = {
    "hr_476f7bd6de9e96149e9e": ("phone_put_away", 0.0, 4.0),
    "hr_5e65de11e8d5dcb59749": ("cash_counter_loading", 6.16, 8.17),
}
INTERESTING_ACTIONS = {"phone_used", "phone_put_away", "cash_counted_by_hand", "cash_counter_used"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def selected_references(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    result = []
    for task, (action, start, end) in REFERENCE_TASKS.items():
        matches = [row for row in rows if row.get("task_id") == task and row.get("group_id") == GROUP]
        if len(matches) != 1:
            raise ValueError("Expected exactly one frozen human reference for " + task)
        row = matches[0]
        if (row.get("action_confirmed") is not True or row.get("split") != "train"
                or row["observed_action"] != action or row["start_sec"] != start or row["end_sec"] != end):
            raise ValueError("Human reference differs from the predeclared fit-demo protocol")
        result.append({key: row.get(key) for key in ("task_id", "group_id", "observed_action", "start_sec", "end_sec",
                      "target_text", "boundary_status", "interval_precision", "action_confirmed", "provenance")})
    return result


def frame_signature(plan, name):
    return [(row["id"], row["timestamp"], row["sha256"], row.get("width"), row.get("height"))
            for row in plan[name]]


def preflight(baseline_job, baseline_plan, candidate_dir, human_path, protocol_path):
    baseline = read(baseline_job)["result"]
    left, right = read(baseline_plan), read(candidate_dir / "plan.json")
    protocol = read(protocol_path)
    if protocol["group_membership"].get(GROUP) != "fit":
        raise ValueError("Demo source must remain in the frozen fit partition")
    if left["source_sha256"] != right["source_sha256"] or "video_" + right["source_sha256"][:16] != GROUP:
        raise ValueError("Pilot source videos differ")
    if frame_signature(left, "frames") != frame_signature(right, "frames"):
        raise ValueError("Overview inputs differ; do not compare this pilot")
    if frame_signature(left, "detail_frames") != frame_signature(right, "detail_frames"):
        raise ValueError("Available detail-frame pools differ")
    for frame in right["detail_frames"]:
        file = (candidate_dir / frame["path"]).resolve()
        if not file.is_relative_to(candidate_dir.resolve()) or sha(file) != frame["sha256"]:
            raise ValueError("Candidate frame changed")
    return baseline, dict(source_sha256=right["source_sha256"], group_id=GROUP, partition="fit",
                          duration_sec=right["duration_sec"], same_overview_frames=True,
                          same_detail_frame_pool=True, overview_frames=len(right["frames"]),
                          detail_pool_frames=len(right["detail_frames"]),
                          same_adaptively_selected_detail_windows="not_established_from_frame_pool"), selected_references(human_path)


def observation_row(row):
    return {key: row.get(key) for key in ("observation_id", "kind", "action", "property", "value", "actor", "role",
            "object", "start_sec", "end_sec", "stage", "description", "uncertainty", "visibility", "evidence")}


def summarize(result, references):
    observations = result.get("observations", [])
    events = [row for row in observations if row.get("kind") == "event"]
    overview = [row for row in events if row.get("stage", "").startswith("overview_")]
    rejections = result.get("rejections", [])
    compared = []
    for reference in references:
        start, end = reference["start_sec"], reference["end_sec"]
        nearby = [row for row in events if min(row["end_sec"], end + 1) > max(row["start_sec"], max(0, start - 1))]
        matches = [row for row in overview if row.get("action") == reference["observed_action"]
                   and temporal_iou(row, reference) >= 0.3]
        compared.append(dict(reference_task=reference["task_id"], reference_action=reference["observed_action"],
                             scenario_action_code_exists=reference["observed_action"] in ACTIONS,
                             exact_action_overview_temporal_matches=[dict(observation_id=row["observation_id"],
                                  temporal_iou=temporal_iou(row, reference)) for row in matches],
                             nearby_visual_proposals=[observation_row(row) for row in nearby],
                             rule="Exact code + overview-only temporal IoU>=0.3, fixed before candidate result",
                             semantic_candidates_require_human_review=True,
                             limitation="Loading a counter is not proof of counting; descriptions are not automatically relabeled."))
    return dict(status=result.get("status"), elapsed_sec=result.get("elapsed_sec"), coverage=result.get("coverage"),
                total_observations=len(observations), total_event_proposals=len(events),
                overview_event_counts=dict(Counter(row.get("action") for row in overview)),
                all_stage_event_counts=dict(Counter(row.get("action") for row in events)),
                phone_and_counting_proposals=[observation_row(row) for row in events if row.get("action") in INTERESTING_ACTIONS],
                rejection_count=len(rejections), json_or_schema_rejection_count=sum(
                    any(word in str(row.get("error", "")).lower() for word in ("json", "schema", "expected", "unknown action", "unknown property"))
                    for row in rejections), rejection_details=rejections,
                rule_verdict_counts=dict(Counter(row.get("verdict") for row in result.get("rules", []))),
                raw_model_claim_verdict_counts=dict(Counter(row.get("model_verdict") for row in result.get("model_rule_claims", []))),
                reference_comparison=compared,
                empty_json_windows_cannot_be_counted_from_observation_count=True)


def compare(baseline, candidate, plan_check, references):
    for result in (baseline, candidate):
        if result.get("organizer_answers_in_prompt") is not False:
            raise ValueError("Blind inference provenance is missing or contaminated")
        if result.get("cloud_ai_used") is not False:
            raise ValueError("Expected local-only inference provenance")
    required = ("model_id", "model_revision", "quantization", "runtime_release", "inference_config")
    if any(baseline.get(key) is None or candidate.get(key) is None for key in required):
        raise ValueError("Required model/runtime provenance is missing")
    equality = {key: dict(equal=baseline.get(key) == candidate.get(key), baseline=baseline.get(key), candidate=candidate.get(key))
                for key in required}
    if any(not value["equal"] for value in equality.values()):
        raise ValueError("Different model/runtime configuration; compare in a separate experiment")
    code_names = sorted(set(baseline.get("code_sha256", {})) | set(candidate.get("code_sha256", {})))
    changed = [name for name in code_names if baseline.get("code_sha256", {}).get(name) != candidate.get("code_sha256", {}).get(name)]
    relevant_changed = [name for name in changed if name in {"runner.py", "protocol.py", "schema.py"}]
    output_schema_explicit_equal = (bool(baseline.get("output_schema_version"))
                                    and baseline.get("output_schema_version") == candidate.get("output_schema_version"))
    left, right = summarize(baseline, references), summarize(candidate, references)
    return dict(schema_version="gguf_phone_fit_demo_comparison_v1", experiment_kind="posthoc_fit_demonstration",
                is_accuracy_estimate=False, independent_test=False, measured_accuracy=None,
                plan_comparison=plan_check, matched_runtime=equality,
                prompt_sha256=dict(baseline=baseline.get("prompt_sha256"), candidate=candidate.get("prompt_sha256")),
                candidate_instructions_source=candidate.get("instructions_source"),
                schema_provenance=dict(baseline=baseline.get("output_schema_version"), candidate=candidate.get("output_schema_version"),
                                       explicitly_same=output_schema_explicit_equal),
                changed_code_files=changed, recognition_protocol_code_changes=relevant_changed,
                isolated_prompt_ablation_established=False,
                baseline=left, candidate=right, human_fit_references=references,
                runtime_delta_sec=(right["elapsed_sec"] - left["elapsed_sec"]) if isinstance(right["elapsed_sec"], (int, float)) and isinstance(left["elapsed_sec"], (int, float)) else None,
                limitations=[
                    "One previously inspected fit video; this is a demonstration, not accuracy, recall or an independent test.",
                    "Same 32 overview frames and same available detail pool; adaptive detail-window selections can differ.",
                    "Original pilot predates explicit output_schema_version; runner/protocol/schema changes prevent a clean instruction-only causal claim.",
                    "phone_put_away 0–4s is a confirmed action, not proof of a violation.",
                    "cash_counter_loading 6.16–8.17s has no exact action code in the current scenario ontology. Do not substitute completed counting.",
                    "Accepted human boundaries may retain the original proposal boundaries; temporal matches are approximate.",
                    "More observations or more rule claims alone do not establish better recognition.",
                    "Frame sampling coverage is not continuous coverage; absent sampled actions do not prove absence."])


def write_text_report(report, path):
    lines = ["GGUF: подробная инструкция, posthoc демонстрация на fit-видео", "",
             "Это не accuracy и не независимый test. Положительные ручные действия не означают нарушение.",
             "Одинаковы 32 обзорных кадра; детальные окна могут различаться. Старый пилот имеет иную/неполную provenance схемы.", ""]
    for name, label in (("baseline", "Исходный пилот"), ("candidate", "Подробная инструкция")):
        item = report[name]
        coverage = item.get("coverage") or {}
        lines.append(f"{label}: {item['status']}, {item['elapsed_sec']} с; кадры {coverage.get('inspected_sampled_frames', 0)}/{coverage.get('total_sampled_frames', 0)}; наблюдений {item['total_observations']}; отклонений {item['rejection_count']}.")
        lines.append("Действия основного прохода: " + json.dumps(item["overview_event_counts"], ensure_ascii=False))
        lines.append("Телефон/пересчёт: " + str(len(item["phone_and_counting_proposals"])) + " предложений; это не число подтверждённых событий.")
        for check in item["reference_comparison"]:
            lines.append(f"  {check['reference_action']}: точных совпадений кода и интервала {len(check['exact_action_overview_temporal_matches'])}; код есть в сценарной таксономии: {check['scenario_action_code_exists']}.")
        lines.append("")
    lines.append("Ключевые ограничения:")
    lines.extend("- " + text for text in report["limitations"])
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/gemma_lab/adaptation_v1"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("runs/gemma_lab/uploads/f22640e7b6e34f69a6d6e55d7f314132"))
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    candidate_dir = (args.candidate_dir or args.run_root / "gguf_instructions_v3_pilot").resolve()
    baseline_job = args.baseline_dir / "job.json"
    baseline_plan = args.baseline_dir / "plan.json"
    human = args.run_root / "human_actions_fit.jsonl"
    protocol = args.run_root / "frozen_group_protocol.json"
    baseline, check, references = preflight(baseline_job, baseline_plan, candidate_dir, human, protocol)
    if args.preflight_only:
        print(json.dumps(dict(status="ready_for_result", plan_comparison=check,
                              human_references=len(references), baseline_summary=summarize(baseline, references)), ensure_ascii=False))
        return
    result_path = candidate_dir / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError("Candidate result has not been generated/synchronized. No comparison or accuracy fabricated.")
    candidate = read(result_path)
    if candidate.get("status") not in {"completed", "partial", "failed"}:
        raise ValueError("Candidate job is still running; wait for its terminal result")
    report = compare(baseline, candidate, check, references)
    report["input_files"] = {str(path): sha(path) for path in [baseline_job, baseline_plan, candidate_dir / "plan.json", result_path, human, protocol]}
    output = args.output or args.run_root / "gguf_instructions_v3_comparison"
    if output.exists():
        raise FileExistsError("Use a new comparison output path")
    output.mkdir(parents=True)
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    write_text_report(report, output / "REPORT.txt")
    print(json.dumps(dict(output=str(output), experiment_kind=report["experiment_kind"], measured_accuracy=None,
                          baseline_elapsed=report["baseline"]["elapsed_sec"], candidate_elapsed=report["candidate"]["elapsed_sec"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
