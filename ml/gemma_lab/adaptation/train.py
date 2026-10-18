"""Bounded, local Gemma QLoRA training on audited fit groups only.

Example (the caller should also set an outer timeout with a checkpoint grace):
  timeout -k 10 340 .venv-gemma-adapt/bin/python -m ml.gemma_lab.adaptation.train \
    --dataset runs/gemma_lab/adaptation_v1/prepared/training_samples.jsonl \
    --model artifacts/gemma_adaptation/base --output runs/gemma_lab/adaptation_v1/pilot \
    --max-updates 2 --max-seconds 300 --save-optimizer

One pass through deterministically shuffled examples. --max-updates is the total
successful update count including a resumed checkpoint; successfully checkpointed
sample IDs are skipped on resume. Unknown/rejected/held-out sources are forbidden.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import gc
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

from .hf_backend import MODEL_ID, MODEL_REVISION

DEFAULT_LOCK = Path("/home/ubuntu/nuhack-training/.gpu-work.lock")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_rows(rows, protocol):
    """Independent training gate, even when dataset.py already audited the rows."""
    if not rows:
        raise ValueError("No eligible training examples")
    protocol_content = {k: v for k, v in protocol.items() if k != "protocol_sha256"}
    if protocol.get("protocol_sha256") != canonical_sha(protocol_content):
        raise ValueError("Frozen partition digest mismatch")
    seen = set()
    for row in rows:
        sid, gid = row.get("sample_id"), row.get("group_id")
        if not sid or sid in seen:
            raise ValueError("Missing or duplicate sample ID")
        seen.add(sid)
        if row.get("partition") != "fit" or protocol["group_membership"].get(gid) != "fit":
            raise ValueError(f"Training leakage: {sid} is outside fit groups")
        identity = protocol["source_identity"].get(gid, {})
        if identity.get("split") != "train" or identity.get("video_sha256") != row.get("video_sha256"):
            raise ValueError(f"Source identity mismatch: {sid}")
        expected = {("human_action", "human_confirmed"): 1.0,
                    ("weak_caption", "weak_unconfirmed"): 0.2}.get((row.get("kind"), row.get("trust_level")))
        weight = row.get("loss_weight")
        if expected is None or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight != expected:
            raise ValueError(f"Unsupported source/trust/weight: {sid}")
        if row.get("status") in ("quarantined", "rejected", "reject"):
            raise ValueError(f"Excluded source status: {sid}")
        if not isinstance(row.get("prompt"), str) or not isinstance(row.get("target"), str):
            raise ValueError(f"Missing prompt or assistant target: {sid}")
    return {"eligible_examples": len(rows), "eligible_groups": len({r["group_id"] for r in rows}),
            "eligible_by_kind": dict(Counter(r["kind"] for r in rows))}


@contextmanager
def gpu_lock(path=DEFAULT_LOCK):
    """Fail immediately when occupied; do not inspect or terminate other jobs."""
    with Path(path).open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("GPU_BUSY: shared lock is held by another job") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def assert_gpu_idle():
    check = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    foreign = [line.strip() for line in check.stdout.splitlines()
               if line.strip() and line.split(",", 1)[0].strip() != str(os.getpid())]
    if foreign:
        raise RuntimeError("GPU_BUSY: existing compute processes detected; none were stopped")


def single_example_collate(examples):
    if len(examples) != 1:
        raise ValueError("This bounded trainer uses batch size one")
    return examples[0]


class StopTraining(Exception):
    pass


def run(args):
    started = time.monotonic()
    output = args.output.resolve()
    if (output / "status.json").exists() and not args.resume:
        raise FileExistsError("Output already exists; use a new run directory or explicit --resume")
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = args.dataset.resolve()
    root = args.run_root.resolve() if args.run_root else dataset_path.parent.parent
    protocol_path = args.protocol.resolve() if args.protocol else root / "frozen_group_protocol.json"
    state = {"schema_version": 1, "status": "processing", "phase": "validating",
             "started_at": time.time(), "dataset": str(dataset_path), "model_id": MODEL_ID,
             "model_revision": MODEL_REVISION, "max_seconds": args.max_seconds,
             "max_updates": args.max_updates, "optimizer_updates": 0,
             "learning_rate": args.learning_rate, "max_tokens": args.max_tokens,
             "consumed_sample_ids": [], "consumed_records": [], "skipped_examples": [],
             "checkpoint": None, "resume_from": str(args.resume.resolve()) if args.resume else None}
    state["training_code_sha256"] = file_sha(Path(__file__))
    state["backend_code_sha256"] = file_sha(Path(__file__).with_name("hf_backend.py"))
    state["initialization"] = {
        "seed": args.seed, "scope": "torch_cpu_and_cuda_before_backend_loading_and_lora_creation",
        "source": "fresh_adapter", "bitwise_cuda_determinism_guaranteed": False,
    }
    state["run_rng_seed"] = args.seed
    state["resume_rng_state_restored"] = False
    backend = optimizer = scaler = None
    prior_handlers = {}
    lock_stack = ExitStack()
    update_in_progress = False
    def status(phase=None, **updates):
        if phase:
            state["phase"] = phase
        state.update(updates)
        state["used_by_kind"] = dict(Counter(r["kind"] for r in state["consumed_records"]))
        state["used_groups"] = len({r["group_id"] for r in state["consumed_records"]})
        state["elapsed_sec"] = time.monotonic() - started
        state["updated_at"] = time.time()
        atomic_json(output / "status.json", state)

    def stopped(signum, frame):
        raise StopTraining("time_budget" if signum == signal.SIGALRM else "termination_requested")

    def checkpoint():
        if backend is None or backend.updates == 0 or update_in_progress:
            return None
        path = output / "checkpoints" / f"update_{backend.updates:06d}"
        if path.is_dir() and (path / "trainer_state.json").exists():
            state["checkpoint"] = str(path)
            return path
        status("checkpointing")
        path.parent.mkdir(exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        if temporary.exists():
            raise FileExistsError("Incomplete checkpoint preserved: " + str(temporary))
        backend.save_adapter(temporary, {
            "dataset_sha256": state["dataset_sha256"], "protocol_sha256": state["protocol_sha256"],
            "seed": args.seed, "consumed_sample_ids": state["consumed_sample_ids"],
            "weighted_weak_supervision": True, "no_production_activation": True,
            "initialization": state["initialization"],
            "training_code_sha256": state["training_code_sha256"],
            "backend_code_sha256": state["backend_code_sha256"],
            "resume_rng_state_restored": False,
        })
        if args.save_optimizer:
            backend.torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()},
                               temporary / "optimizer.pt")
        state["optimizer_updates"] = backend.updates
        state["checkpoint"] = str(path)
        atomic_json(temporary / "trainer_state.json", state)
        temporary.replace(path)
        return path

    status()
    try:
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
            os.environ[name] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        private_cache = args.model.resolve().parent / "training_hf_cache"
        private_cache.mkdir(exist_ok=True)
        os.environ["HF_HOME"] = str(private_cache)
        rows = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
        protocol = json.loads(protocol_path.read_text())
        state.update(validate_rows(rows, protocol))
        state.update(dataset_sha256=file_sha(dataset_path), protocol_sha256=protocol["protocol_sha256"], seed=args.seed)
        if args.resume:
            resumed = json.loads((args.resume / "trainer_state.json").read_text())
            for key in ("dataset_sha256", "protocol_sha256", "seed", "model_id", "model_revision"):
                if resumed.get(key) != state[key]:
                    raise ValueError("Resume provenance mismatch: " + key)
            for key in ("optimizer_updates", "consumed_sample_ids", "consumed_records"):
                state[key] = resumed[key]
            state["initialization"] = resumed.get("initialization", {
                "seed": None, "scope": "not_attested_in_legacy_checkpoint_metadata",
                "source": "resumed_adapter", "shuffle_seed_does_not_attest_initial_lora_seed": True,
                "bitwise_cuda_determinism_guaranteed": False,
            })
            if len(set(state["consumed_sample_ids"])) != len(state["consumed_sample_ids"]):
                raise ValueError("Resume has duplicate consumed samples")
            if set(state["consumed_sample_ids"]) - {r["sample_id"] for r in rows}:
                raise ValueError("Resume contains unknown consumed samples")
        from .dataset import GemmaAdaptationDataset
        dataset = GemmaAdaptationDataset(dataset_path, run_root=root)
        order = list(range(len(rows)))
        random.Random(args.seed).shuffle(order)
        if len(dataset) != len(rows):
            raise ValueError("Dataset silently changed eligible row count")
        for signum in (signal.SIGALRM, signal.SIGTERM):
            prior_handlers[signum] = signal.signal(signum, stopped)
        signal.setitimer(signal.ITIMER_REAL, max(0.01, args.max_seconds - (time.monotonic() - started)))
        status("waiting_gpu_lock")
        lock_stack.enter_context(gpu_lock(args.gpu_lock))
        assert_gpu_idle()
        status("loading_model")
        from .hf_backend import load_backend
        import torch
        from torch.utils.data import DataLoader, Subset
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        backend = load_backend(args.model, training=True, adapter_path=args.resume)
        backend.updates = int(state["optimizer_updates"])
        backend.examples_used = len(state["consumed_sample_ids"])
        optimizer, scaler = backend.make_optimizer(args.learning_rate), backend.make_scaler()
        optimizer_file = args.resume / "optimizer.pt" if args.resume else None
        if optimizer_file and optimizer_file.exists():
            saved = backend.torch.load(optimizer_file, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(saved["optimizer"])
            scaler.load_state_dict(saved["scaler"])
            state["optimizer_resumed"] = True
        else:
            state["optimizer_resumed"] = False
        state["backend"] = backend.provenance()
        remaining_order = [i for i in order if rows[i]["sample_id"] not in state["consumed_sample_ids"]]
        loader = DataLoader(Subset(dataset, remaining_order), batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=single_example_collate)
        status("training")
        stop_reason = "dataset_exhausted"
        with (output / "consumption.jsonl").open("a", encoding="utf-8") as ledger:
            for offset, sample in enumerate(loader):
                index = remaining_order[offset]
                sid = rows[index]["sample_id"]
                if backend.updates >= args.max_updates:
                    stop_reason = "max_updates"
                    break
                if time.monotonic() - started >= args.max_seconds - args.checkpoint_grace:
                    stop_reason = "time_budget"
                    break
                metadata = sample["metadata"]
                if metadata["sample_id"] != sid:
                    raise ValueError("Dataset order changed after audit")
                status("training", current_sample_id=sid, current_group_id=metadata["group_id"])
                try:
                    encoded = backend.encode(sample["frames"], sample["prompt"], sample["target"],
                                             max_tokens=args.max_tokens)
                except ValueError as error:
                    result = {"updated": False, "reason": "encoding_rejected",
                              "error": str(error), "loss_weight": sample["loss_weight"]}
                else:
                    update_in_progress = True
                    result = backend.train_batch(encoded, optimizer, scaler,
                                                 loss_weight=sample["loss_weight"])
                record = {"sample_id": sid, "group_id": metadata["group_id"],
                          "video_sha256": metadata["video_sha256"], "kind": metadata["kind"],
                          "trust_level": metadata["trust_level"], "loss_weight": sample["loss_weight"],
                          "frame_sha256": [f["sha256"] for f in metadata["frames"]],
                          "prompt_sha256": hashlib.sha256(sample["prompt"].encode()).hexdigest(),
                          "target_sha256": hashlib.sha256(sample["target"].encode()).hexdigest(),
                          "provenance": metadata.get("provenance"), "result": result,
                          "recorded_at": time.time()}
                ledger.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                ledger.flush()
                os.fsync(ledger.fileno())
                if result["updated"]:
                    state["consumed_sample_ids"].append(sid)
                    state["consumed_records"].append(record)
                    state["optimizer_updates"] = backend.updates
                else:
                    state["skipped_examples"].append({"sample_id": sid, "reason": result.get("reason")})
                update_in_progress = False
                status("training", last_step=result)
                if not result["updated"]:
                    if result.get("reason") == "encoding_rejected":
                        continue
                    raise FloatingPointError("Nonfinite training update; pilot stopped, not counted as training")
                if backend.updates % args.checkpoint_every == 0:
                    checkpoint()
        # Disarm before the reserved checkpoint grace. The caller's outer
        # timeout remains a hard process bound even if a CUDA op stalls.
        signal.setitimer(signal.ITIMER_REAL, 0)
        if backend.updates == 0:
            raise RuntimeError("No successful finite updates; no trained checkpoint exists")
        checkpoint()
        completed = stop_reason != "time_budget"
        status("finished", status="completed" if completed else "partial", stop_reason=stop_reason,
               examples_actually_used=len(state["consumed_sample_ids"]),
               examples_not_used=len(rows) - len(state["consumed_sample_ids"]))
    except StopTraining as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            checkpoint()
        except Exception as checkpoint_error:
            state["checkpoint_error"] = f"{type(checkpoint_error).__name__}: {checkpoint_error}"
        status("finished", status="partial", stop_reason=str(error),
               incomplete_update_discarded=update_in_progress,
               examples_actually_used=len(state["consumed_sample_ids"]),
               examples_not_used=state.get("eligible_examples", 0) - len(state["consumed_sample_ids"]))
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            checkpoint()
        except Exception as checkpoint_error:
            state["checkpoint_error"] = f"{type(checkpoint_error).__name__}: {checkpoint_error}"
        status("failed", status="failed", error=f"{type(error).__name__}: {error}",
               examples_actually_used=len(state["consumed_sample_ids"]))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for signum, handler in prior_handlers.items():
            signal.signal(signum, handler)
        # Keep the common lock through error checkpointing and GPU cleanup.
        optimizer = scaler = None
        if backend is not None:
            backend.model = None
            backend = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            finally:
                lock_stack.close()
        else:
            lock_stack.close()
    print(json.dumps({k: state.get(k) for k in ("status", "stop_reason", "optimizer_updates",
                                              "checkpoint", "examples_actually_used", "elapsed_sec")}, ensure_ascii=False))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-updates", type=int, default=41)
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--checkpoint-grace", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--save-optimizer", action="store_true")
    parser.add_argument("--gpu-lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    if not 1 <= args.max_updates <= 1000 or not 30 <= args.max_seconds <= 1500:
        parser.error("Use a bounded pilot: 1–1000 updates, 30–1500 seconds")
    if not 1 <= args.checkpoint_every <= 50 or not 0 < args.checkpoint_grace < args.max_seconds:
        parser.error("Invalid checkpoint interval/grace")
    run(args)


if __name__ == "__main__":
    main()
