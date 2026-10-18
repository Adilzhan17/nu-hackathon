"""Run inside the GPU VM: python3 ml/gpu_preflight.py [--device cuda|mps|cpu].

Requires an installed PyTorch build appropriate for the machine's driver.
Does not download weights, upload data, change drivers, or start paid resources.
Checks forward/backward Conv3d support; this is NOT a full X3D benchmark.
"""

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    args = parser.parse_args()
    report = {"python": sys.version, "platform": platform.platform(), "device": args.device}
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        report["nvidia_smi"] = result.stdout.strip()
        report["nvidia_smi_exit_code"] = result.returncode
    try:
        import torch

        report.update(torch=torch.__version__, cuda_build=torch.version.cuda)
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; check GPU attachment, driver and PyTorch build")
            report["compute_capability"] = torch.cuda.get_device_capability()
            report["compiled_architectures"] = torch.cuda.get_arch_list()
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable")
        torch.set_num_threads(4)
        device = torch.device(args.device)
        model = torch.nn.Sequential(
            torch.nn.Conv3d(3, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(8, 8, 3, padding=1, groups=8),
            torch.nn.AdaptiveAvgPool3d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(8, 2),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters())
        inputs = torch.randn(2, 3, 13, 64, 64, device=device)
        targets = torch.tensor([0, 1], device=device)
        started = time.monotonic()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(inputs), targets)
            loss.backward()
            optimizer.step()
        if args.device == "cuda":
            torch.cuda.synchronize()
        elif args.device == "mps":
            torch.mps.synchronize()
        report["smoke_test_seconds"] = time.monotonic() - started
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite loss")
        report["passed"] = True
    except Exception as exc:
        report.update(passed=False, error=f"{type(exc).__name__}: {exc}")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
