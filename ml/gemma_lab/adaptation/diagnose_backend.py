"""Bounded first-nonfinite diagnostic: scalar stats only, no pixels/text dumped."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
import signal
import time

from .train import DEFAULT_LOCK, assert_gpu_idle, atomic_json, gpu_lock


class FirstNonfinite(Exception):
    pass


def run(args):
    started = time.monotonic()
    result = {"status": "starting", "seed": args.seed, "variant": args.variant,
              "private_content_logged": False, "checked_modules": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    handles, backend = [], None

    def save(**updates):
        result.update(updates)
        result["elapsed_sec"] = time.monotonic() - started
        atomic_json(args.output, result)

    def expire(signum, frame):
        raise TimeoutError("Diagnostic time limit reached")

    prior = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[name] = "1"
    os.environ["HF_HOME"] = str(args.model.resolve().parent / "training_hf_cache")
    try:
        with gpu_lock(args.gpu_lock):
            try:
                assert_gpu_idle()
                import torch
                from .dataset import GemmaAdaptationDataset
                from .hf_backend import load_backend, answer_only_loss
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)
                dataset = GemmaAdaptationDataset(args.dataset)
                order = list(range(len(dataset)))
                random.Random(args.seed).shuffle(order)
                sample = dataset[order[0]]
                save(status="loading", sample_id=sample["metadata"]["sample_id"])
                backend = load_backend(args.model, training=True)
                if args.variant == "fp32_compute":
                    for module in backend.model.modules():
                        if module.__class__.__name__ == "Linear4bit":
                            module.compute_dtype = torch.float32
                            module.compute_type_is_set = True
                backend.model.train()

                def stats(tensor):
                    tensor = tensor.detach()
                    answer = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                              "device": str(tensor.device), "elements": tensor.numel()}
                    if not tensor.numel():
                        return answer
                    if tensor.is_floating_point():
                        finite = torch.isfinite(tensor)
                        count = int(finite.sum().item())
                        answer.update(finite_elements=count, nonfinite_elements=tensor.numel() - count)
                        if count:
                            selected = tensor[finite] if count != tensor.numel() else tensor
                            answer.update(min=float(selected.min().item()), max=float(selected.max().item()))
                    else:
                        answer.update(min=float(tensor.min().item()), max=float(tensor.max().item()))
                    return answer

                def tensors(value):
                    if torch.is_tensor(value):
                        yield value
                    elif isinstance(value, (tuple, list)):
                        for item in value:
                            yield from tensors(item)

                def nonfinite(value):
                    return any(t.is_floating_point() and not bool(torch.isfinite(t).all()) for t in tensors(value))

                # Read frozen nonquantized weights in chunks: never allocate a
                # full bool mask for the billion-parameter shared embedding.
                parameter_problems = []
                dtypes = {}
                for name, parameter in backend.model.named_parameters():
                    dtype = str(parameter.dtype)
                    dtypes[dtype] = dtypes.get(dtype, 0) + parameter.numel()
                    if not parameter.is_floating_point():
                        continue
                    flat = parameter.detach().reshape(-1)
                    for offset in range(0, flat.numel(), 4_000_000):
                        chunk = flat[offset:offset + 4_000_000]
                        if not bool(torch.isfinite(chunk).all()):
                            parameter_problems.append({"module": name, "offset": offset, "stats": stats(chunk)})
                            break
                save(status="forward", parameter_nonfinite=parameter_problems, parameter_dtype_elements=dtypes,
                     allocated_bytes=torch.cuda.memory_allocated(), peak_bytes=torch.cuda.max_memory_allocated())
                if parameter_problems:
                    raise FirstNonfinite("Nonfinite stored parameter before forward")

                def before(name):
                    def hook(module, values):
                        if nonfinite(values):
                            save(first_nonfinite={"module": name, "class": type(module).__name__, "where": "input",
                                                  "input": [stats(t) for t in tensors(values)]})
                            raise FirstNonfinite(name + " input")
                    return hook

                def after(name):
                    def hook(module, values, outputs):
                        result["checked_modules"] += 1
                        if nonfinite(outputs):
                            save(first_nonfinite={"module": name, "class": type(module).__name__, "where": "output",
                                                  "input": [stats(t) for t in tensors(values)],
                                                  "output": [stats(t) for t in tensors(outputs)],
                                                  "parameters": {n: stats(p) for n, p in module.named_parameters(recurse=False)}})
                            raise FirstNonfinite(name + " output")
                    return hook

                for name, module in backend.model.named_modules():
                    if not list(module.children()):
                        handles.append(module.register_forward_pre_hook(before(name)))
                        handles.append(module.register_forward_hook(after(name)))
                encoded = backend.encode(sample["frames"], sample["prompt"], sample["target"])
                inputs = backend._device_inputs(encoded)
                target = encoded["target_ids"].to(backend.device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=args.variant != "no_autocast"):
                    output = backend.model(**inputs, use_cache=False, logits_to_keep=target.shape[1] + 1)
                    loss = answer_only_loss(output.logits, target)
                save(status="finite" if bool(torch.isfinite(loss)) else "nonfinite_loss",
                     loss=float(loss) if bool(torch.isfinite(loss)) else None,
                     logits=stats(output.logits), peak_bytes=torch.cuda.max_memory_allocated())
            except FirstNonfinite as error:
                save(status="nonfinite_located", reason=str(error))
            except BaseException as error:
                save(status="failed", error=type(error).__name__ + ": " + str(error))
                raise
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                for handle in handles:
                    handle.remove()
                if backend:
                    backend.model = None
                    backend = None
                    gc.collect()
                    import torch
                    torch.cuda.empty_cache()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=7301)
    p.add_argument("--max-seconds", type=int, default=150)
    p.add_argument("--gpu-lock", type=Path, default=DEFAULT_LOCK)
    p.add_argument("--variant", choices=("fp16", "fp32_compute", "no_autocast"), default="fp16")
    run(p.parse_args())
