"""Check hardware and dependencies before downloading model weights."""

import json
import platform
import shutil
from importlib.metadata import PackageNotFoundError, version

import torch


def main():
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "datasets"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed (datasets is optional)"
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "disk_free_gib": round(shutil.disk_usage(".").free / 2**30, 1),
    }
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        report.update(
            gpu=gpu.name,
            gpu_total_gib=round(total / 2**30, 2),
            gpu_free_gib=round(free / 2**30, 2),
            bf16=torch.cuda.is_bf16_supported(),
        )
        tensor = torch.ones(16, device="cuda", requires_grad=True)
        tensor.square().sum().backward()
        report["cuda_backward"] = bool(torch.all(tensor.grad == 2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not torch.cuda.is_available():
        print(
            "CUDA is unavailable. The CPU toy works; HF training requires --device cpu or a CUDA wheel and compatible NVIDIA driver."
        )


if __name__ == "__main__":
    main()
