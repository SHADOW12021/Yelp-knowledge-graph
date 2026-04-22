from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TorchDeviceInfo:
    device: str
    torch_version: str
    cuda_available: bool
    cuda_version: str | None
    device_name: str | None


def detect_torch_device() -> TorchDeviceInfo:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    device = "cuda" if cuda_available else "cpu"
    device_name = torch.cuda.get_device_name(0) if cuda_available else None
    return TorchDeviceInfo(
        device=device,
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        cuda_version=torch.version.cuda,
        device_name=device_name,
    )


def build_device_banner(context: str) -> str:
    info = detect_torch_device()
    name = info.device_name or "CPU"
    cuda = info.cuda_version or "none"
    return (
        f"[{context}] torch={info.torch_version} "
        f"device={info.device} "
        f"cuda_available={info.cuda_available} "
        f"cuda_version={cuda} "
        f"name={name}"
    )
