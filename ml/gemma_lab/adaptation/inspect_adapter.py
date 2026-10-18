"""CPU-only forensic inspection of a saved LoRA adapter, not a quality metric."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from .hf_backend import MODEL_ID, MODEL_REVISION
from .train import atomic_json, file_sha


def inspect_checkpoint(checkpoint: Path) -> dict:
    """Inspect completed local tensors without loading the base or any frames."""
    import torch
    from safetensors import safe_open
    checkpoint = checkpoint.resolve()
    if checkpoint.name.endswith(".tmp"):
        raise ValueError("Refusing an unfinished checkpoint directory")
    model_file = checkpoint / "adapter_model.safetensors"
    if model_file.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("Unexpectedly large checkpoint: inspect for full/embedding weights separately")
    config = json.loads((checkpoint / "adapter_config.json").read_text())
    state_path = checkpoint / "trainer_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    provenance_path = checkpoint / "training_provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    results, modules, problems = [], {}, []
    with safe_open(model_file, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            finite = bool(torch.isfinite(tensor).all())
            matches = re.fullmatch(r"(.+)\.lora_([AB])(?:\.default)?\.weight", name)
            if matches is None:
                problems.append("unexpected_tensor:" + name)
            else:
                modules.setdefault(matches.group(1), {})[matches.group(2)] = tensor
            nonzero = int(torch.count_nonzero(tensor).item())
            row = {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                   "device": str(tensor.device), "elements": tensor.numel(),
                   "finite": finite, "nonzero_elements": nonzero,
                   "raw_bytes_sha256": hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
            if finite:
                row["l2_norm"] = float(torch.linalg.vector_norm(tensor.double()).item())
            else:
                problems.append("nonfinite_tensor:" + name)
            results.append(row)
    module_results = []
    for name, pair in sorted(modules.items()):
        if set(pair) != {"A", "B"}:
            problems.append("incomplete_lora_pair:" + name)
            continue
        a, b = pair["A"], pair["B"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            problems.append("incompatible_lora_shapes:" + name)
            continue
        rank = int(a.shape[0])
        expected_rank = config.get("rank_pattern", {}).get(name, config.get("r"))
        if rank != expected_rank:
            problems.append("rank_mismatch:" + name)
        # ||BA||_F² = sum((A A^T) * (B^T B)); avoid constructing giant BA.
        ad, bd = a.double(), b.double()
        norm_squared = float(((ad @ ad.T) * (bd.T @ bd)).sum().item())
        alpha = config.get("alpha_pattern", {}).get(name, config.get("lora_alpha", rank))
        scaling = alpha / (math.sqrt(rank) if config.get("use_rslora") else rank)
        delta = math.sqrt(max(0.0, norm_squared)) * scaling if math.isfinite(norm_squared) else None
        module_results.append({"module": name, "rank": rank, "scaling": scaling,
                               "b_nonzero_elements": int(torch.count_nonzero(b).item()),
                               "effective_delta_frobenius_norm": delta,
                               "nonzero_effective_delta": delta is not None and delta > 0})
    actual_targets = {name.removeprefix("base_model.model.") for name in modules}
    configured_targets = set(config.get("target_modules", []))
    if actual_targets != configured_targets:
        problems.append("saved_modules_do_not_match_explicit_target_modules")
    if config.get("base_model_name_or_path") != MODEL_ID or config.get("revision") != MODEL_REVISION:
        problems.append("unexpected_base_identity")
    initialization = state.get("initialization") or provenance.get("experiment", {}).get("initialization")
    if initialization is None:
        initialization = {
            "seed": None, "scope": "not_attested_in_checkpoint_metadata",
            "caveat": "Recorded shuffle seed alone does not establish the original LoRA initialization seed; verify the executed source, not a later working copy.",
        }
    updates = state.get("optimizer_updates", provenance.get("optimizer_updates"))
    nonzero_b_modules = sum(row["b_nonzero_elements"] > 0 for row in module_results)
    effective_modules = sum(row["nonzero_effective_delta"] for row in module_results)
    if updates is not None and updates > 0 and effective_modules == 0:
        problems.append("successful_updates_claimed_but_no_effective_adapter_delta")
    default_zero_b = config.get("init_lora_weights", True) is True
    return {
        "schema_version": 1, "checkpoint": str(checkpoint),
        "adapter_sha256": file_sha(model_file), "adapter_bytes": model_file.stat().st_size,
        "config_sha256": file_sha(checkpoint / "adapter_config.json"),
        "base_model_id": config.get("base_model_name_or_path"), "base_model_revision": config.get("revision"),
        "optimizer_updates_recorded": updates, "cpu_only": True, "base_model_loaded": False,
        "tensor_count": len(results), "module_count": len(module_results),
        "parameter_count": sum(r["elements"] for r in results),
        "nonzero_b_modules": nonzero_b_modules, "nonzero_effective_delta_modules": effective_modules,
        "finite": all(r["finite"] for r in results),
        "default_zero_b_initialization": default_zero_b,
        "evidence_of_nonzero_update": default_zero_b and effective_modules > 0 and not problems,
        "accuracy_or_quality_inferred": False,
        "initialization": initialization,
        "resume_rng_state_restored": state.get("resume_rng_state_restored", False),
        "resume_reproducibility_caveat": "Optimizer/scaler continuation does not establish uninterrupted-run equivalence when dropout RNG state was not restored.",
        "problems": problems, "tensors": results, "modules": module_results,
        "baseline_contract": {
            "same_upstream_weights_and_revision": True,
            "runtime": "HF bitsandbytes NF4; frozen vision FP32; language compute FP16",
            "adapter_disabled_vs_enabled": True, "eval_dropout_disabled": True,
            "past_key_values_shared_between_examples": False,
            "production_gguf_quantization_equivalence_claimed": False,
            "runtime_contract_is_measured_by_inference_evaluation_not_this_cpu_inspection": True,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_checkpoint(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("adapter_sha256", "optimizer_updates_recorded", "finite",
                                                 "module_count", "nonzero_b_modules", "nonzero_effective_delta_modules",
                                                 "evidence_of_nonzero_update", "problems")}, ensure_ascii=False))
    if report["problems"]:
        raise SystemExit(2)
