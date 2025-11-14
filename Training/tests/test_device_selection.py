from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pytest

from Media import device_utils


def _patch_resources(monkeypatch, ram_mb: Optional[int], gpu_vrams: Optional[List[Optional[int]]]) -> None:
    if ram_mb is not None:
        monkeypatch.setattr(device_utils, "_get_available_ram_mb", lambda: ram_mb)
    else:
        monkeypatch.setattr(device_utils, "_get_available_ram_mb", lambda: None)
    monkeypatch.setattr(device_utils, "_get_all_gpu_vram_mb", lambda: gpu_vrams)


@pytest.mark.parametrize(
    "name, expected_device, gpu_vrams",
    [
        ("Wan-AI/Wan2.2-TI2V-5B", "cuda:0", [24000]),
        ("dx8152/Qwen-Edit-2509-Multiple-angles", "cpu", [6000]),
    ],
)
def test_resolve_device_chooses_best_fit(name: str, expected_device: str, gpu_vrams: List[Optional[int]], monkeypatch) -> None:
    _patch_resources(monkeypatch, ram_mb=32000, gpu_vrams=gpu_vrams)
    assert device_utils.resolve_device(name) == expected_device


def test_resolve_device_falls_back_to_cpu_when_no_gpu(monkeypatch) -> None:
    _patch_resources(monkeypatch, ram_mb=32000, gpu_vrams=[])
    assert device_utils.resolve_device("enhanceaiteam/Flux-Uncensored-V2") == "cpu"


def test_resolve_device_raises_when_insufficient_ram(monkeypatch) -> None:
    _patch_resources(monkeypatch, ram_mb=4000, gpu_vrams=[24000])
    with pytest.raises(MemoryError):
        device_utils.resolve_device("second-state/FLUX.1-dev-GGUF")


def test_prefers_specific_cuda_index_when_requested(monkeypatch) -> None:
    _patch_resources(monkeypatch, ram_mb=64000, gpu_vrams=[10000, 25000])
    assert device_utils.resolve_device("Wan-AI/Wan2.2-TI2V-5B", preferred="cuda:1") == "cuda:1"


def test_requested_gpu_unavailable_defaults_to_cpu(monkeypatch) -> None:
    _patch_resources(monkeypatch, ram_mb=32000, gpu_vrams=None)
    assert device_utils.resolve_device("enhanceaiteam/Flux-Uncensored-V2", preferred="cuda:0") == "cpu"


def test_get_model_cache_path_creates_named_directory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HF_CACHE_DIR", raising=False)
    monkeypatch.setenv("HF_CACHE_DIR", str(tmp_path / "hf"))
    path = Path(device_utils.get_model_cache_path("Wan-AI/Wan2.2-TI2V-5B"))
    assert path.exists()
    assert path.name == "Wan-AI_Wan2.2-TI2V-5B"
    assert str(path).startswith(str(tmp_path / "hf"))